# postprocess.py

import re


def remove_gender_age_words(text: str) -> str:
    """
    VLM 출력에서 성별/연령 추정 표현을 중립 표현으로 치환합니다.

    예:
    - a woman -> a person
    - two men -> two people
    - boys/girls/children -> people
    - young/old/elderly -> 제거
    """
    if not text or not str(text).strip():
        return ""

    result = str(text).strip()

    # 연령 추정 형용사 제거
    age_adjective_patterns = [
        r"\byoung\s+",
        r"\bold\s+",
        r"\belderly\s+",
        r"\bmiddle-aged\s+",
        r"\bteenage\s+",
    ]

    for pattern in age_adjective_patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)

    # 복수 표현 먼저 처리
    plural_patterns = [
        (r"\btwo women\b", "two people"),
        (r"\btwo men\b", "two people"),
        (r"\bthree women\b", "three people"),
        (r"\bthree men\b", "three people"),
        (r"\bseveral women\b", "several people"),
        (r"\bseveral men\b", "several people"),
        (r"\bmultiple women\b", "multiple people"),
        (r"\bmultiple men\b", "multiple people"),
        (r"\bwomen\b", "people"),
        (r"\bmen\b", "people"),
        (r"\bgirls\b", "people"),
        (r"\bboys\b", "people"),
        (r"\bchildren\b", "people"),
        (r"\badults\b", "people"),
        (r"\bfemales\b", "people"),
        (r"\bmales\b", "people"),
    ]

    # 단수 표현 처리
    singular_patterns = [
        (r"\ba woman\b", "a person"),
        (r"\ba man\b", "a person"),
        (r"\ba girl\b", "a person"),
        (r"\ba boy\b", "a person"),
        (r"\ba child\b", "a person"),
        (r"\ban adult\b", "a person"),
        (r"\ba female\b", "a person"),
        (r"\ba male\b", "a person"),
        (r"\bwoman\b", "person"),
        (r"\bman\b", "person"),
        (r"\bgirl\b", "person"),
        (r"\bboy\b", "person"),
        (r"\bchild\b", "person"),
        (r"\badult\b", "person"),
        (r"\bfemale\b", "person"),
        (r"\bmale\b", "person"),
    ]

    for pattern, replacement in plural_patterns + singular_patterns:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # 어색한 중복 표현 정리
    result = re.sub(
        r"\ba person and a person\b",
        "two people",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(r"\ba person person\b", "a person", result, flags=re.IGNORECASE)
    result = re.sub(r"\bpeople people\b", "people", result, flags=re.IGNORECASE)
    result = re.sub(r"\s+", " ", result).strip()

    return result


def capitalize_first_letter(text: str) -> str:
    """
    문장 첫 글자를 대문자로 보정합니다.
    """
    if not text:
        return ""

    return text[0].upper() + text[1:] if len(text) > 1 else text.upper()


def clean_vlm_caption(text: str) -> str:
    """
    VLM 출력 문장을 저장용으로 간단히 정리합니다.

    처리 내용:
    - 공백/줄바꿈 정리
    - 양끝 따옴표 제거
    - 첫 번째 완성 문장만 남김
    - 성별/연령 추정 표현 제거
    - 마지막 문장부호를 항상 "."으로 통일
    - 문장부호가 없으면 "." 추가
    - 문장 첫 글자 대문자 보정
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

    # 성별/연령 표현 제거
    text = remove_gender_age_words(text)

    # 문장 첫 글자 대문자 보정
    text = capitalize_first_letter(text)

    # 마지막 문장부호를 항상 "."으로 통일
    if text.endswith((".", "!", "?")):
        text = text[:-1].rstrip() + "."
    else:
        text += "."

    return text