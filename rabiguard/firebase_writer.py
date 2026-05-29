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

    zone_doc_ref = db.collection(collection_name).document(str(zone_id))

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

    event_doc_id = f"{now.strftime('%Y%m%d_%H%M%S')}_track_{track_id}"

    event_doc_ref = zone_doc_ref.collection("events").document(event_doc_id)
    event_doc_ref.set(event_data)

    return event_doc_ref.id


def save_zones_to_firestore(zones_data, collection_name="zones"):
    """
    추출된 Zone 정보를 Firestore에 저장합니다.
    기존에 동일한 ID를 가진 문서는 덮어씌워집니다.
    """
    db = init_firestore()
    batch = db.batch()

    for zone_id, data in zones_data.items():
        doc_ref = db.collection(collection_name).document(zone_id)
        batch.set(doc_ref, data)

    batch.commit()
    print(f"✅ [Firestore] {len(zones_data)}개의 구역이 '{collection_name}' 컬렉션에 저장되었습니다.")
