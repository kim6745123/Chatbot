# doc_parser/src/embedding_test.py
# 실행: python -m src.embedding_test  (또는 python src\embedding_test.py)
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import time

print("=== embedding_test start ===")
print("cwd:", Path.cwd())
print("python:", sys.executable)

# load .env
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print("Loaded .env from", env_path)
else:
    print("No .env found at", env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    print("OPENAI_API_KEY found (will enable OpenAI embedding).")
else:
    print("OPENAI_API_KEY not found. OpenAI embedding will be skipped.")

# import heavy libs inside try to show helpful errors
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("ERROR: sentence-transformers not available. Install with `pip install sentence-transformers`")
    print("Exception:", e)
    sys.exit(1)

# Optional OpenAI
USE_OPENAI = bool(OPENAI_API_KEY)
if USE_OPENAI:
    try:
        from openai import OpenAI
    except Exception as e:
        print("ERROR: openai package not installed. Install with `pip install openai` to use OpenAI embeddings.")
        print("Exception:", e)
        USE_OPENAI = False

# test sentence
TEST_SENTENCE = "인공지능(AI)을 이용해 전공 과제를 수행하는 것에 관해 어떻게 생각하나요?"

def try_local_model(name):
    print(f"\n--- Loading local model: {name} ---")
    t0 = time.time()
    model = SentenceTransformer(name)
    t1 = time.time()
    print(f"Loaded model '{name}' in {t1-t0:.2f}s")
    # embed
    t2 = time.time()
    emb = model.encode(TEST_SENTENCE)
    t3 = time.time()
    print(f"Embedding done in {t3-t2:.2f}s; vector length: {len(emb)}")
    # show first 8 values
    print("sample:", [float(x) for x in emb[:8]])
    return emb, model

def try_openai_embedding(api_key):
    print("\n--- Using OpenAI embeddings ---")
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print("Failed to create OpenAI client:", e)
        return None
    try:
        t0 = time.time()
        resp = client.embeddings.create(model="text-embedding-3-large", input=TEST_SENTENCE)
        t1 = time.time()
        emb = resp.data[0].embedding
        print(f"OpenAI embedding done in {t1-t0:.2f}s; vector length: {len(emb)}")
        print("sample:", emb[:8])
        return emb
    except Exception as e:
        print("OpenAI embedding request failed:", e)
        return None

def cosine_sim(a, b):
    # simple numpy-based cosine
    import numpy as np
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if a.shape != b.shape:
        return None
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(a, b) / (na * nb))

def main():
    results = {}
    # local models to test
    local_models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "jhgan/ko-sroberta-multitask"
    ]

    for m in local_models:
        try:
            emb, model_obj = try_local_model(m)
            results[m] = {"emb": emb}
        except Exception as e:
            print(f"Failed model {m}: {e}")
            results[m] = {"error": str(e)}

    openai_emb = None
    if USE_OPENAI:
        openai_emb = try_openai_embedding(OPENAI_API_KEY)
        if openai_emb is not None:
            results["openai/text-embedding-3-large"] = {"emb": openai_emb}
        else:
            results["openai/text-embedding-3-large"] = {"error": "openai request failed"}

    # similarity table
    print("\n=== Cosine similarity between embeddings ===")
    keys = [k for k in results.keys() if "emb" in results[k]]
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            k1 = keys[i]; k2 = keys[j]
            s = cosine_sim(results[k1]["emb"], results[k2]["emb"])
            print(f"{k1}  <->  {k2}   = {s:.4f}")

    print("\n=== embedding_test end ===")

if __name__ == "__main__":
    main()
