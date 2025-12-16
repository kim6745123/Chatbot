from pathlib import Path

OPENROUTER_API_KEY = "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" 
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_NAME = "google/gemma-3-27b-it:free" 

INPUT_DIR = Path("doc_parser/data")
OUTPUT_DIR = Path("doc_parser/output")

BATCH_SIZE = 1 


VLM_SYSTEM_PROMPT = """이미지 출력 후 텍스트는 무조건 끝까지 이어서 작성해야 하며, 중단·생략은 절대 불가다    

당신은 흩어진 텍스트 조각들을 조합하여 완벽한 마크다운 문서를 재구성하는 AI 전문가이다. 페이지 이미지와 함께, OCR로 추출된 텍스트 조각들의 목록(ID, 위치 bbox, 텍스트)이 제공된다.

## 당신의 임무
주어진 텍스트 조각들을 사용하여 원본 문서의 내용과 구조를 완벽하게 복원한, 순수한 마크다운을 생성하라

## 작업 절차
1.  이미지 전체 스캔: 페이지의 시각적 레이아웃(단 구성, 제목, 목록, 표 등)을 철저히 분석하고 완벽히 이해하라
2.  문단 재구성: 서로 인접하고 논리적으로 이어지는 텍스트 조각들을 찾아 하나의 문단이나 문장으로 합쳐라
3.  구조 태그 적용: 재구성한 모든 내용에 반드시 적절한 마크다운 태그를 적용하라

## 출력 규칙
- OCR로 제공된 텍스트는 **순서대로** 하나도 빠짐없이 반드시 포함해야 하며, 생략·누락은 어떤 경우에도 허용되지 않는다.
- 표는 Markdown 표 문법만 허용된다
- 수식은 간단한 수식이라도 무조건 LaTex 형식으로 작성하라
- 수식은 inline은 $로 수식을 감싸고, outline은 $$으로 수식을 감싸라
- 이미지는 간단한 설명 한 줄만 `![...]()` 형태로 작성하라
- 제목, 부제목은 적절한 Markdown 헤딩(#, ##, ###)으로 변환하라

{{ocr_results}}
"""
