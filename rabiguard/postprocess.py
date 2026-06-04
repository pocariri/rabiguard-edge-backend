# postprocess.py

import re


def clean_vlm_caption(text: str) -> str:
    """
    VLM 출력 문장을 저장용으로 간단히 정리합니다.

    처리 내용:
    - 공백/줄바꿈 정리
    - 양끝 따옴표 제거
    - 첫 번째 완성 문장만 남김
    - 마지막 문장부호를 항상 "."으로 통일
    - 문장부호가 없으면 "." 추가
    """
    if not text or not str(text).strip():
        return ""

    text = str(text).strip()

    # 공백/줄바꿈 정리
    text = re.sub(r"\s+", " ", text).strip()

    # 양끝 따옴표 제거
    text = text.strip("\"'“”‘’").strip()

    # 첫 번째 완성 문장만 남김
    match = re.search(r"(.+?[.!?])(\s|$)", text)
    if match:
        text = match.group(1).strip()

    # 첫 문장 추출 후 남을 수 있는 따옴표 제거
    text = text.strip("\"'“”‘’").strip()

    # 마지막 문장부호를 항상 "."으로 통일
    if text.endswith((".", "!", "?")):
        text = text[:-1].rstrip() + "."
    else:
        text += "."

    return text