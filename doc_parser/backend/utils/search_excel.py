import os
import pandas as pd
import re

def normalize(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", "", text).lower()

def find_competition_ratio(year, university, major, admission):
    """
    경쟁률을 .xlsx 또는 .md 파일에서 검색하는 함수
    """

    uni = normalize(university)
    maj = normalize(major)
    adm = normalize(admission)

    # ---------------------------
    # 1) 우선 .xlsx 시도
    # ---------------------------
    xlsx_path = f"output/{university}_{year}.xlsx"
    if os.path.exists(xlsx_path):
        try:
            df = pd.read_excel(xlsx_path)
            return search_dataframe(df, uni, maj, adm)
        except:
            pass  # 파싱 오류 → md 파일로 fallback

    # ---------------------------
    # 2) .md 파일 fallback
    # ---------------------------
    md_path = f"output/{university}_{year}.md"
    if os.path.exists(md_path):
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                md_text = f.read()

            df = parse_md_table(md_text)
            return search_dataframe(df, uni, maj, adm)
        except Exception as e:
            print("MD parsing error:", e)
            return None

    return None


def parse_md_table(md_text: str):
    """
    Markdown 표를 파싱하여 pandas DataFrame으로 변환
    """

    # 표 구간만 추출
    lines = [line.strip() for line in md_text.splitlines() if line.strip().startswith("|")]

    # 헤더 + 데이터
    table_str = "\n".join(lines)

    # pandas read_csv를 이용한 파싱
    from io import StringIO
    table_str = re.sub(r"\|\s*---.*\n", "", table_str)  # 구분선 제거

    df = pd.read_csv(StringIO(table_str), sep="|", engine="python")

    # 불필요한 빈 column 제거
    df = df.dropna(axis=1, how="all")

    # 양쪽 공백 제거
    df.columns = [c.strip() for c in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    return df


def search_dataframe(df, uni, maj, adm):
    """
    DataFrame을 기반으로 경쟁률을 검색
    """

    # 컬럼명 정규화
    df.columns = [normalize(c) for c in df.columns]

    # 예상되는 컬럼명
    col_university = [c for c in df.columns if "대학" in c][0]
    col_major = [c for c in df.columns if "모집단위명" in c or "학과" in c][0]
    col_admission = [c for c in df.columns if "전형" in c][0]
    col_ratio = [c for c in df.columns if "경쟁률" in c][0]

    # 모든 텍스트 컬럼 정규화
    df["_uni"] = df[col_university].apply(normalize)
    df["_maj"] = df[col_major].apply(normalize)
    df["_adm"] = df[col_admission].apply(normalize)

    # 기본 필터
    mask = (df["_uni"] == uni) & (df["_maj"] == maj)

    if adm:
        # admission이 부분만 있어도 매칭되게
        mask = mask & (df["_adm"].str.contains(adm))

    result = df[mask]

    if len(result) == 0:
        return None

    ratio_value = result.iloc[0][col_ratio]
    try:
        return float(ratio_value)
    except:
        return ratio_value
