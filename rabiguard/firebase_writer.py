# firebase_writer.py

from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore

try:
    from config import FIREBASE_KEY_PATH
except ImportError:
    from .config import FIREBASE_KEY_PATH


def init_firestore():
    """
    Firestore client 초기화.
    이미 firebase_admin 앱이 초기화되어 있으면 재사용합니다.
    """
    if not firebase_admin._apps:
        if not FIREBASE_KEY_PATH.exists():
            raise FileNotFoundError(f"Firebase key file not found: {FIREBASE_KEY_PATH}")

        cred = credentials.Certificate(str(FIREBASE_KEY_PATH))
        firebase_admin.initialize_app(cred)

    return firestore.client()


def save_vlm_result_to_firestore(
    english_text: str,
    korean_text: str,
    image_path: str,
    collection_name: str = "vlm_events",
    zone_id: str = "",
    track_id: int | str = "",
    person_depth: float | None = None,
    zone_depth: float | None = None,
):
    """
    VLM 분석 결과를 Firestore에 저장합니다.
    """
    db = init_firestore()

    data = {
        "created_at": datetime.now(timezone.utc),
        "english_text": english_text,
        "korean_text": korean_text,
        "image_path": image_path,
        "zone_id": zone_id,
        "track_id": track_id,
        "person_depth": person_depth,
        "zone_depth": zone_depth,
    }

    write_time, doc_ref = db.collection(collection_name).add(data)

    return doc_ref.id