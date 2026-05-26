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

    저장 구조:
    vlm_events/{zone_id}/events/{event_doc_id}
    """

    db = init_firestore()

    if not zone_id:
        zone_id = "unknown_zone"

    now = datetime.now(timezone.utc)

    # 구역별 문서 참조
    zone_doc_ref = db.collection(collection_name).document(str(zone_id))

    # 구역별 events 서브컬렉션에 이벤트 기록 저장
    event_data = {
        "created_at": now,
        "english_text": english_text,
        "korean_text": korean_text,
        "image_path": image_path,
        "zone_id": str(zone_id),
        "track_id": track_id,
        "person_depth": person_depth,
        "zone_depth": zone_depth,
    }

    # 자동 ID 대신 직접 문서 ID 생성
    event_doc_id = f"{now.strftime('%Y%m%d_%H%M%S_%f')}_track_{track_id}"

    # 생성한 문서 ID로 저장
    event_doc_ref = zone_doc_ref.collection("events").document(event_doc_id)
    event_doc_ref.set(event_data)

    return event_doc_ref.id