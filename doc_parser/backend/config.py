import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent.parent

# .env 로드
load_dotenv(BASE_DIR / ".env")

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

# 데이터 폴더
DATA_DIR = BASE_DIR / "output"

# 임베딩 및 LLM 설정
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"
BATCH_SIZE = 32
