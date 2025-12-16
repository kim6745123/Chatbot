#parser.py
import re
from datetime import datetime

ADMISSION_KEYWORDS = [
    "기회균형전형",
    "고른기회전형",
    "특성화고교졸업자전형",
    "실기우수자전형",
    "아리학생부교과전형",
    "아리학생부면접전형",
    "아리학생부종합전형Ⅰ",
    "아리학생부종합전형Ⅱ",
    "농어촌학생전형",
    "학교장추천전형",
    "체육특기자전형",
    "특성화고등을졸업한재직자전형"
]

def parse_competition_query(query: str):
    parsed = {
        "years": [],
        "university": None,
        "major": None,
        "admission": None,
        "wants_graph": False
    }

    # 연도 추출
    years = re.findall(r"(20\d{2})", query)
    parsed["years"] = list(map(int, years))

    if not parsed["years"]:
        m = re.search(r"(최근|지난)\s*(\d+)\s*년|(\d+)\s*개년", query)
        if m:
            n = int(m.group(2) or m.group(3))  # 최근3년 / 3개년
            this_year = datetime.now().year     # 서버 실행 시점 기준
            parsed["years"] = list(range(this_year - (n - 1), this_year + 1))

    # 대학명 추출
    uni_match = re.search(r"([가-힣]+대학?|[가-힣]+대학교)", query)
    parsed["university"] = uni_match.group() if uni_match else None

    if parsed["university"] is None:
        parsed["university"] = "안양대"

    # 학과 추출
    major_match = re.search(r"([\w가-힣]+학과)", query)
    parsed["major"] = major_match.group() if major_match else None

    # 전형 추출 (ADMISSION_KEYWORDS 우선)
    for kw in ADMISSION_KEYWORDS:
        if kw in query:
            parsed['admission'] = kw
            break

    # 그래프 요청 여부
    if "그래프" in query or "추이" in query:
        parsed['wants_graph'] = True

    return parsed

