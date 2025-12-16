#chroma.py
from chromadb import PersistentClient
from pathlib import Path

# 저장 경로 설정 (원하는 폴더에 저장 가능)
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
CHROMA_DIR.mkdir(exist_ok=True)

class ChromaManager:
    def __init__(self, collection_name="anyang_docs"):
        # ✅ 영구 저장 클라이언트로 변경
        self.client = PersistentClient(path=str(CHROMA_DIR))

        # ✅ 기존 컬렉션 불러오기 (없으면 생성)
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            self.collection = self.client.create_collection(collection_name)
    
    def add_documents(self, ids, docs, embeddings, metadatas):
        self.collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def query(self, query_embedding, top_k=5):
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return res
    
    def count(self):
        try:
            return self.collection.count()
        except Exception:
            return 0

    def persist(self):
        try:
            self.client.persist()
        except Exception:
            pass

