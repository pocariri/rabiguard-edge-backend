# firebase_writer.py

import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone
from pathlib import Path


FIREBASE_KEY_PATH = Path("/home/rafour/workspace/jongmin/rafour-app/firebase_key.json")


def init_firestore():
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
    collection_name: str = "vlm_test_results",
):
    db = init_firestore()

    data = {
        "created_at": datetime.now(timezone.utc),
        "english_text": english_text,
        "korean_text": korean_text,
        "image_path": image_path,
    }

    # 중요: add() 반환 순서는 write_time, doc_ref
    write_time, doc_ref = db.collection(collection_name).add(data)

    return doc_ref.id
