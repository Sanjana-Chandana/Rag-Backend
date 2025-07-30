import fitz
import re
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from pydantic import BaseModel
from typing import List
from chromadb import Client
from chromadb.config import Settings
from openai import AzureOpenAI 
import requests
import os
import uvicorn

# === Config ===
from dotenv import load_dotenv
import openai
load_dotenv()
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_CHAT_ENDPOINT = os.getenv("AZURE_CHAT_ENDPOINT")
openai.api_key = AZURE_API_KEY
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for localhost frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
client = Client(Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection("pdf_chunks")

# === Step 1: Embed Function ===
def get_embedding(text: str):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY
    }
    body = {"input": text}
    res = requests.post(AZURE_EMBEDDING_ENDPOINT, headers=headers, json=body)
    res.raise_for_status()
    return res.json()["data"][0]["embedding"]

# === Step 2: PDF Processing with Agentic Chunking ===
def process_pdf(file: UploadFile):
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, file.filename)
    with open(pdf_path, "wb") as f:
        f.write(file.file.read())

    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    doc.close()
    shutil.rmtree(temp_dir)

    # === Use GPT-4o to generate agentic chunks ===
    chat_client = AzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_CHAT_ENDPOINT,
        api_version="2024-08-01-preview"
    )

    prompt = f"""
    You are an intelligent assistant helping to process text for semantic search.

    Split the following document into semantically meaningful sections (called chunks), each focusing on a coherent idea. Each chunk should be around 100-200 words long and make sense when read independently.
    Return the chunks as a numbered list in plain text format.
    Document:
    {full_text}"""

    try:
        response = chat_client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant trained in text chunking for retrieval."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        chunked_output = response.choices[0].message.content.strip()
        chunks = chunked_output.split("\n\n")  

    except Exception as e:
        print("Chunking error:", e)
        return

    embeddings = []
    metadatas = []
    ids = []

    for idx, chunk in enumerate(chunks):
        if chunk.strip():
            try:
                vector = get_embedding(chunk)
                embeddings.append(vector)
                metadatas.append({"source": file.filename})
                ids.append(f"{file.filename}_chunk_{idx}")
            except Exception as e:
                print("Embedding error:", e)

    if embeddings:
        collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
        print(f"Added {len(embeddings)} agentically chunked embeddings to Chroma.")
        print(f"Total chunks in collection: {collection.count()}")
    else:
        print(f"No valid chunks embedded for: {file.filename}")

@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    count = 0
    for file in files:
        process_pdf(file)
        count += 1
    return {"message": f"Uploaded and indexed {count} PDF(s)."}

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: AskRequest):
    try:
        chunks_used_in_answer = []

        query_embedding = get_embedding(req.question)
        results = collection.query(query_embeddings=[query_embedding], n_results=5)

        semantic_chunks = results["documents"][0]
        semantic_sources = results["metadatas"][0]

        keyword_matches = []
        keyword_sources = []
        matched_keywords = set()

        question_words = set(re.findall(r'\w+', req.question.lower()))

        all_chunks = collection.get(include=["documents", "metadatas"])
        best_overlap = 0
        best_chunk = None
        best_metadata = None

        for doc, meta in zip(all_chunks["documents"], all_chunks["metadatas"]):
            doc_words = set(re.findall(r'\w+', doc.lower()))
            intersection = question_words.intersection(doc_words)
            overlap_score = len(intersection)

            if intersection:
                keyword_matches.append(doc)
                keyword_sources.append(meta)
                matched_keywords.update(intersection)

            if overlap_score > best_overlap:
                best_overlap = overlap_score
                best_chunk = doc
                best_metadata = meta

        combined_chunks = semantic_chunks + [c for c in keyword_matches if c not in semantic_chunks]
        combined_sources = semantic_sources + [s for i, s in enumerate(keyword_sources) if keyword_matches[i] not in semantic_chunks]

        if not combined_chunks:
            return {
                "answer": "I cannot answer that based on the uploaded PDFs.",
                "source_pdf": None,
                "matched_keywords": []
            }

        context = "\n---\n".join(combined_chunks)
        source_files = list(set([meta['source'] for meta in semantic_sources]))

        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_API_KEY
        }
        body = {
            "messages": [
                {"role": "system", "content": "Check if the question is related to any of the uploaded PDFs and give priority to answering questions from the PDFs. However, if the question is not related to uploading PDFs, then say 'The provided question is not related to the uploaded PDFs, but a generic response is:', and then provide a general response"},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question}"}
            ],
            "model": "gpt-4o"
        }

        res = requests.post(AZURE_CHAT_ENDPOINT, headers=headers, json=body)
        res.raise_for_status()
        reply = res.json()["choices"][0]["message"]["content"]

        is_generic = not chunks_used_in_answer and (
            reply.strip().lower().startswith("the provided question is not related") or
            len(matched_keywords) == 0
        )

        attribution_prompt = f"""
        You are a helpful assistant that identifies which context chunks were actually used to generate an answer.
        Below is the context (from a document), and the final answer generated. Your job is to return only the specific context chunks that directly contributed to the answer.
        ### CONTEXT CHUNKS:
        {chr(10).join(f"[{i+1}] {chunk}" for i, chunk in enumerate(combined_chunks))}
        ### ANSWER:
        {reply}
        Now return the relevant chunks from the context that contain information actually used to form the answer. Return them as plain text list — one chunk per item. Ignore unrelated chunks."""

        try:
            response = requests.post(
                AZURE_CHAT_ENDPOINT,
                headers=headers,
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that matches answers to relevant source chunks."},
                        {"role": "user", "content": attribution_prompt}
                    ]
                }
            )
            response.raise_for_status()
            extracted_chunks = response.json()["choices"][0]["message"]["content"].strip().split("\n")
            seen = set()
            for chunk_text in extracted_chunks:
                chunk_text_clean = chunk_text.strip().lstrip("-•1234567890. ").strip()
                for chunk, meta in zip(combined_chunks, combined_sources):
                    if chunk_text_clean in chunk and chunk not in seen:
                        seen.add(chunk)
                        chunks_used_in_answer.append({
                            "chunk": chunk,
                            "source": meta
                        })
                        break
        except Exception as e:
            print("Chunk attribution error:", e)
            chunks_used_in_answer = []

        return {
            "answer": reply,
            "source_pdf": ["Generic response"] if is_generic else source_files,
            "matched_keywords": [] if is_generic else list(matched_keywords),
            "answer_chunk": {
                "chunk": best_chunk,
                "source": best_metadata
            } if best_chunk and not is_generic else None,
            "chunks_used_in_answer": chunks_used_in_answer if not is_generic else []
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)