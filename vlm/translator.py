# translator.py

from deep_translator import GoogleTranslator


def translate_to_korean(text: str) -> str:
    """영어 문장을 한국어로 단순 번역"""
    if not text or not text.strip():
        return ""

    try:
        return GoogleTranslator(source="en", target="ko").translate(text)
    except Exception as e:
        print(f"[Translation Error] {e}")
        return text