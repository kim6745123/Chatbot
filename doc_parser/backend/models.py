# models.py
from pydantic import BaseModel
from typing import Optional, List

class UserRequest(BaseModel):
    email: str
    password: str
    userId: Optional[str] = None
    chatId: Optional[str] = None
    question: Optional[str] = None 

class UserResponse(BaseModel):
    email: str

class ChatRequest(BaseModel):
    userId: str
    chatId: Optional[str] = None
    question: str

class ChatResponse(BaseModel):
    answer: str

class MessageItem(BaseModel):
    role: str
    content: str

class SaveMessageRequest(BaseModel):
    userId: str
    chatId: str
    messages: List[MessageItem]

