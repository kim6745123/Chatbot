import os
from pathlib import Path
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

# FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Chroma
import chromadb

# OpenAI (new SDK)
from openai import OpenAI

# utils
import time, traceback

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RAG Backend (Chroma + OpenAI) - with debug endpoints")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev 서버 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---- load .env ----
proj_root = Path(__file__).resolve().parents[1]
INPUT_MD_PATH = proj_root / "output" / "2026_수시_output.md"
load_dotenv(proj_root / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

# ---- OpenAI client ----
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Chroma client (new-style) ----
chroma_client = chromadb.Client()
COLLECTION_NAME = "anyang_docs"

# Try to get or create collection
try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
except Exception:
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

# ---- Input directory & chunking settings ----
# INPUT_MD_DIR = proj_root / "output"   # 기본: doc_parser/output
# MD_GLOB = "*.md"
md_files = [INPUT_MD_PATH]

CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP = 100

def split_into_chunks(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_MAX_CHARS:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_MAX_CHARS
        chunk = text[start:end].strip()
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks

def read_md_files() -> List[tuple]:
    """
    Reads a single markdown file (INPUT_MD_PATH) and returns list of (filename, block_text).
    Splits input by two-or-more newlines to get paragraph-like blocks.
    """
    results = []
    if not INPUT_MD_PATH.exists():
        print(f"Input file not found: {INPUT_MD_PATH}")
        return results

    try:
        txt = INPUT_MD_PATH.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Failed to read {INPUT_MD_PATH}: {e}")
        return results

    parts = [seg.strip() for seg in re.split(r'\n{2,}', txt) if seg.strip()]
    for seg in parts:
        results.append((INPUT_MD_PATH.name, seg))
    return results

# ---- Populate Chroma (one-time if empty) ----
def populate_chroma_if_empty():
    # Check if collection has items
    try:
        cnt = collection.count()
    except Exception:
        # fallback: try a safe query to detect non-emptiness
        try:
            res = collection.query(query_texts=[""], n_results=1, include=["documents"])
            cnt = len(res.get("documents", [[]])[0])
        except Exception:
            cnt = 0

    if cnt and cnt > 0:
        print("Chroma already has data, skip populate.")
        return

    docs = read_md_files()
    ids, texts, metadatas = [], [], []
    idx = 0
    for source_filename, block in docs:
        chunks = split_into_chunks(block)
        for ci, chunk in enumerate(chunks, start=1):
            idx += 1
            ids.append(f"{source_filename}__{idx}")
            texts.append(chunk)
            metadatas.append({"source": source_filename, "chunk_idx": ci})
    if not texts:
        print("No markdown files to index.")
        return

    # embed in batches via OpenAI embeddings
    embeddings = []
    BATCH = 32
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = openai_client.embeddings.create(model="text-embedding-3-large", input=batch)
        for item in resp.data:
            embeddings.append(item.embedding)

    # add to collection
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    try:
        chroma_client.persist()
    except Exception:
        pass
    print(f"Populated Chroma with {len(texts)} chunks from {len(docs)} md files.")

# call populate at startup
print("Starting server... populating Chroma if empty.")
populate_chroma_if_empty()

# ---- Query + LLM synthesis endpoint ----
class QueryReq(BaseModel):
    query: str
    top_k: int = 5
    llm_model: str = "gpt-4o-mini"
    max_context_chars: int = 3000

def build_prompt(query: str, docs: List[Dict[str,Any]], max_chars: int = 3000) -> str:
    header = (
        "다음은 검색 결과 문단들입니다. 주어진 문단들을 참고하여 한국어로 명확하고 간결한 답변을 작성하세요.\n"
        "요구사항:\n"
        "1) 사용자 질문에 대한 핵심 답변을 2-4문장(간결)으로 작성하세요.\n"
        "2) 답변 밑에 출처 목록을 '출처:'로 작성하고 각 항목에 문서 id 또는 파일명을 표기하세요.\n"
        "3) 문맥에서 불확실한 정보는 추측하지 말고 '출처에서 직접 확인하세요' 같은 문구를 사용하세요.\n\n"
    )
    body = f"사용자 질문: {query}\n\n검색 결과:\n"
    ctx = ""
    for i, d in enumerate(docs, start=1):
        meta = d.get("metadata") or {}
        label = meta.get("source") or d.get("id")
        piece = f"--- 문단 {i} (출처: {label}) ---\n{d.get('document')}\n\n"
        if len(ctx) + len(piece) > max_chars:
            break
        ctx += piece
    prompt = header + body + ctx + "\n\n위 정보를 바탕으로 답변하세요:"
    return prompt

def run_chroma_query_and_format(query: str, top_k: int):
    # embed query
    q_emb_resp = openai_client.embeddings.create(model="text-embedding-3-large", input=[query])
    q_emb = q_emb_resp.data[0].embedding

    # request documents, metadatas, distances (do NOT request "ids" in include for some chroma versions)
    res = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas", "distances"])

    docs = []
    ids = res.get("ids", [[]])[0] if "ids" in res else []
    docs_list = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0] if "metadatas" in res else []
    dists = res.get("distances", [[]])[0] if "distances" in res else []

    length = max(len(ids), len(docs_list), len(metas), len(dists))
    for i in range(length):
        doc_id = ids[i] if i < len(ids) else f"unknown_{i}"
        doc_text = docs_list[i] if i < len(docs_list) else ""
        meta = metas[i] if i < len(metas) else {}
        docs.append({
            "id": doc_id,
            "document": doc_text,
            "metadata": meta
        })

    return docs, dists

@app.post("/query_synth")
def query_synth(req: QueryReq):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    try:
        docs, dists = run_chroma_query_and_format(req.query, req.top_k)
    except Exception as e:
        tb = traceback.format_exc()
        print("Error during search:", tb)
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

    if not docs:
        return {"answer": "관련 정보를 찾지 못했습니다.", "sources": []}

    prompt = build_prompt(req.query, docs, max_chars=req.max_context_chars)

    # call OpenAI response-generation
    try:
        resp = openai_client.responses.create(
            model=req.llm_model,
            input=prompt,
            max_output_tokens=512,
            temperature=0.0
        )
    except Exception as e:
        tb = traceback.format_exc()
        print("OpenAI response error:", tb)
        raise HTTPException(status_code=500, detail=f"OpenAI response error: {e}")

    # Extract text robustly
    text = None
    if hasattr(resp, "output_text"):
        text = resp.output_text
    else:
        try:
            if isinstance(resp.data, list) and len(resp.data) > 0:
                first = resp.data[0]
                text = getattr(first, "text", None) or first.get("text", None) or str(first)
            else:
                text = str(resp)
        except Exception:
            text = str(resp)

    sources = [{"id": d["id"], "source": (d["metadata"].get("source") if d["metadata"] else None)} for d in docs]

    return {"answer": text.strip() if text else "요약을 생성하지 못했습니다.", "sources": sources, "raw_search_distances": dists}

# -----------------------
# Debug endpoints
# -----------------------
@app.get("/debug_index")
def debug_index():
    """
    Returns simple diagnostics about the chroma collection:
    - count (if available)
    - up to 5 sample documents (preview)
    """
    # collection exist?
    info = {"collection_name": COLLECTION_NAME}
    try:
        cnt = None
        try:
            cnt = collection.count()
        except Exception:
            # fallback: try a safe query
            try:
                res = collection.query(query_texts=[""], n_results=1, include=["documents"])
                cnt = len(res.get("documents", [[]])[0])
            except Exception:
                cnt = None
        info["count"] = cnt
    except Exception as e:
        info["count_error"] = str(e)

    # sample docs
    sample = {}
    try:
        # use collection.get if supported
        res = collection.get(include=["ids", "documents", "metadatas"], limit=5)
        sample["ids"] = res.get("ids", [])
        sample["documents_preview"] = [d[:400] for d in res.get("documents", [])]
        sample["metadatas"] = res.get("metadatas", [])
    except Exception as e:
        sample["error"] = str(e)
    info["sample"] = sample
    return info

@app.post("/reindex_force")
def reindex_force():
    """
    WARNING: This will delete the collection and rebuild from files.
    Use only for debugging/reindexing.
    """
    try:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        # recreate
        _ = chroma_client.create_collection(name=COLLECTION_NAME)
        # repopulate (force): call populate logic that doesn't skip on non-empty (we temporarily reuse function)
        populate_chroma_if_empty()  # current function checks emptiness; since we deleted, it'll populate
        return {"status": "reindexed"}
    except Exception as e:
        tb = traceback.format_exc()
        print("Reindex error:", tb)
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")

# ---- run server if invoked directly ----
if __name__ == "__main__":
    import uvicorn
    print("Uvicorn starting at http://0.0.0.0:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
