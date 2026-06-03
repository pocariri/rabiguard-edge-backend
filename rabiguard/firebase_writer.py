# firebase_writer.py

from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore, messaging

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

def get_fcm_token():
    db = init_firestore()
    doc = db.collection("fcm_tokens").document("device").get()
    if doc.exists:
        return doc.to_dict().get("token")
    return None

def send_fcm_notification(title: str, body: str):
    token = get_fcm_token()
    if not token:
        print("FCM 토큰 없음, 알림 전송 스킵")
        return
    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
        token=token,
    )
    response = messaging.send(message)
    print(f"FCM 알림 전송 완료: {response}")

def save_vlm_result_to_firestore(
    english_text: str,
    korean_text: str,
    image_path: str,
    collection_name: str = "vlm_events",
    zone_id: str = "",
    track_id: int | str = "",
    person_depth: float | None = None,
    zone_depth: float | None = None,
    status: str = "completed",
):
    """
    VLM 분석 결과를 Firestore에 저장합니다.

    저장 구조:
    vlm_events/{zone_id}/events/{event_doc_id}
    
    status:
    - completed: VLM 분석 및 번역까지 정상 완료
    - vlm_failed: 사람 감지는 되었지만 VLM 분석 실패, fallback 문장 저장
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
        "status": status,
    }

    event_doc_id = f"{now.strftime('%Y%m%d_%H%M%S')}_track_{track_id}"

    event_doc_ref = zone_doc_ref.collection("events").document(event_doc_id)
    event_doc_ref.set(event_data)

    # FCM 푸시 알림 전송
    send_fcm_notification(
        title="⚠️ 보안 알림",
        body=korean_text,
    )
    
    return event_doc_ref.id


def clear_firestore_collection(collection_name: str = "auto_zones"):
    """
    Firestore 컬렉션의 기존 문서를 모두 삭제합니다.

    자동구역 객체 감지를 실행할 때마다
    기존 자동구역 데이터를 초기화하기 위해 사용합니다.
    """
    db = init_firestore()

    docs = db.collection(collection_name).stream()

    batch = db.batch()
    deleted_count = 0

    for doc in docs:
        batch.delete(doc.reference)
        deleted_count += 1

        # Firestore batch 작업 제한을 고려하여 450개마다 commit
        if deleted_count % 450 == 0:
            batch.commit()
            batch = db.batch()

    # 남은 삭제 작업 commit
    if deleted_count % 450 != 0:
        batch.commit()

    print(f"🧹 [Firestore] '{collection_name}' 컬렉션 초기화 완료: {deleted_count}개 문서 삭제")


def save_zones_to_firestore(
    zones_data,
    collection_name: str = "auto_zones",
    reset_before_save: bool = False,
):
    """
    추출된 Zone 정보를 Firestore에 저장합니다.

    reset_before_save=True이면 저장 전에 기존 collection 문서를 모두 삭제합니다.
    실행할 때마다 새로 인식된 객체만 Firestore에 남기고 싶을 때 사용합니다.
    """
    db = init_firestore()

    if reset_before_save:
        clear_firestore_collection(collection_name)

    batch = db.batch()
    saved_count = 0

    for zone_id, data in zones_data.items():
        doc_ref = db.collection(collection_name).document(str(zone_id))
        batch.set(doc_ref, data)
        saved_count += 1

        # 저장도 450개마다 commit
        if saved_count % 450 == 0:
            batch.commit()
            batch = db.batch()

    if saved_count % 450 != 0:
        batch.commit()

    if reset_before_save:
        print(
            f"✅ [Firestore] 기존 데이터를 초기화한 뒤 "
            f"{len(zones_data)}개의 구역이 '{collection_name}' 컬렉션에 저장되었습니다."
        )
    else:
        print(f"✅ [Firestore] {len(zones_data)}개의 구역이 '{collection_name}' 컬렉션에 저장되었습니다.")