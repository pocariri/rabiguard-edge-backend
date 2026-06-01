# firestore_listener.py

import time

try:
    from .config import zone_config_queue, stop_event
    from .firebase_writer import init_firestore
except ImportError:
    from config import zone_config_queue, stop_event
    from firebase_writer import init_firestore


def on_zone_snapshot(col_snapshot, changes, read_time):
    """
    Firestore zones 컬렉션 변경 사항을 감지해서
    zone_config_queue로 전달합니다.
    """
    for change in changes:
        try:
            doc = change.document
            zone_id = doc.id

            if change.type.name == "ADDED":
                zone_config_queue.put(
                    {
                        "action": "update",
                        "zone_id": zone_id,
                        "data": doc.to_dict(),
                    }
                )
                print(f"➕ [Firestore] Zone added: {zone_id}")

            elif change.type.name == "MODIFIED":
                zone_config_queue.put(
                    {
                        "action": "update",
                        "zone_id": zone_id,
                        "data": doc.to_dict(),
                    }
                )
                print(f"🔄 [Firestore] Zone modified: {zone_id}")

            elif change.type.name == "REMOVED":
                zone_config_queue.put(
                    {
                        "action": "delete",
                        "zone_id": zone_id,
                    }
                )
                print(f"❌ [Firestore] Zone removed: {zone_id}")

        except Exception as e:
            print(f"⚠️ [Firestore Listener Change Error] {e}")


def start_firestore_listener(collection_name="zones"):
    """
    Firestore zones 컬렉션을 실시간 감시합니다.
    """
    print("🔵 [Firestore Listener] 초기화 시작...")

    try:
        db = init_firestore()

        query_watch = db.collection(collection_name).on_snapshot(on_zone_snapshot)

        print(f"✅ [Firestore Listener] '{collection_name}' 컬렉션 감시 시작")

        while not stop_event.is_set():
            time.sleep(0.5)

        try:
            query_watch.unsubscribe()
            print("🔴 [Firestore Listener] 감시 종료")
        except Exception as e:
            print(f"⚠️ [Firestore Listener unsubscribe warning] {e}")

    except Exception as e:
        print(f"⚠️ [Firestore Listener Error] {e}")
        print("⚠️ Firestore listener 없이 기본 Zone_A1 구역으로만 실행됩니다.")
