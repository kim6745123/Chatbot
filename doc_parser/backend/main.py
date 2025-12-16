#main.py
from fastapi import FastAPI, HTTPException, Body
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from .models import UserRequest, UserResponse, ChatRequest, ChatResponse
from .rag_engine import query_and_answer, index_all_documents, competition_handler
from .auth import login, signup
import uuid
from pydantic import BaseModel
from typing import List
from .utils.parser import parse_competition_query
from .config import LLM_MODEL
from openai import OpenAI

client = OpenAI()

class MessageItem(BaseModel):
    role: str
    content: str

class SaveMessageRequest(BaseModel):
    userId: str
    chatId: str
    messages: List[MessageItem]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting...")
    # 서버 시작 시 문서 인덱싱
    index_all_documents()
    yield
    print("Server shutting down...")

app = FastAPI(title="RAG Backend (Chroma + OpenAI)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # 로컬 개발용
        "https://chatbot-56i4idcwf-seulbees-projects.vercel.app",  # 지금 prod 도메인(예시)
    ],
    allow_origin_regex=r"^https:\/\/.*\.vercel\.app$",  # ✅ Vercel 프리뷰/프로덕션 도메인 전부 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 회원가입
@app.post("/api/signup")
def api_signup(req: UserRequest):
    success = signup(req.email, req.password)
    if not success:
        raise HTTPException(status_code=400, detail="이미 가입된 계정")
    return {"message": "회원가입 성공"}

# 로그인
@app.post("/api/login")
def api_login(req: UserRequest):
    user = login(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="로그인 실패")
    return user

# 채팅
@app.post("/api/chat")
def api_chat(req: ChatRequest):
    chat_id = req.chatId or str(uuid.uuid4())

    # 1) 경쟁률 처리
    comp_result = competition_handler(req.question)
    if comp_result:
        comp_result["chat_id"] = chat_id
        data = comp_result["content"]

        # 텍스트 형태
        if comp_result["type"] == "text":
            # 단일 수치인지 체크
            years = list(data.keys())
            values = list(data.values())

            # --------------------------
            # 유형 A: 단일 연도 경쟁률 요청
            # --------------------------
            if len(years) == 1:
                year = years[0]
                val  = values[0]

                val_str = "정보 없음" if val is None else str(val)

                prompt = f"""
                다음 경쟁률 데이터를 바탕으로 1~2줄로 자연스럽게 설명해줘.

                - {year}년 경쟁률: {val_str}

                너무 길게 말하지 말고, 최대 1~2문장만.
                """

                resp = client.responses.create(
                    model=LLM_MODEL,
                    input=prompt,
                    max_output_tokens=80
                )

                comp_result["answer"] = resp.output_text
                return comp_result

            # --------------------------
            # 유형 B: 여러 연도 → 그래프 없는 텍스트 요청
            # --------------------------
            detail_text = "\n".join([f"- {y}년: {data[y]}" for y in years])

            prompt = f"""
            다음 대학 경쟁률 데이터를 분석해줘:
            {detail_text}

            간단히 추세나 특징을 요약해서 2~3문장 이내로 설명해줘.
            """

            resp = client.responses.create(
                model=LLM_MODEL,
                input=prompt,
                max_output_tokens=120
            )

            comp_result["answer"] = resp.output_text
            return comp_result

         # 그래프 요청
        if comp_result["type"] == "graph":
            # graph일 때 수치는 content가 아니라 values에 들어 있음
            data = comp_result["values"]

            detail_text = "\n".join([f"- {y}년: {data[y]}" for y in data])

            prompt = f"""
            다음 경쟁률 데이터를 기반으로 그래프를 해석해서
            간단히 2~3줄로 설명해줘.

            {detail_text}

            너무 길게 설명하지 말 것.
            """

            resp = client.responses.create(
                model=LLM_MODEL,
                input=prompt,
                max_output_tokens=80
            )

            comp_result["summary"] = resp.output_text
            return comp_result

    # 일반 RAG
    answer = query_and_answer(req.question)
    return { "type": "text", "answer": answer, "chat_id": chat_id }


