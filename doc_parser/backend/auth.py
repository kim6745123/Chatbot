# auth.py
from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import declarative_base, sessionmaker

# --------------------
# DB 설정
# --------------------
Base = declarative_base()
engine = create_engine("sqlite:///backend/users.db", echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

# --------------------
# User 테이블
# --------------------
class User(Base):
    __tablename__ = "users"
    email = Column(String, primary_key=True, index=True)
    password = Column(String)

# 테이블 생성
Base.metadata.create_all(bind=engine)

# --------------------
# 회원가입
# --------------------
def signup(email: str, password: str) -> bool:
    # 이미 존재하면 가입 실패
    existing_user = session.query(User).filter_by(email=email).first()
    if existing_user:
        return False
    # 새 사용자 생성
    new_user = User(email=email, password=password)
    session.add(new_user)
    session.commit()
    return True

# --------------------
# 로그인
# --------------------
def login(email: str, password: str):
    user = session.query(User).filter_by(email=email, password=password).first()
    if not user:
        return None
    return {"email": user.email}
