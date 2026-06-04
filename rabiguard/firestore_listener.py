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


def start_firestore_listener():
    """
    manual_zones와 auto_zones 컬렉션을 동시에 실시간 감시합니다.
    """
    manual_col = "manual_zones"
    auto_col = "auto_zones"
    
    print(f"🔵 [Firestore Listener] '{manual_col}' 및 '{auto_col}' 감시 시작...")

    try:
        db = init_firestore()

        # 두 컬렉션에 대해 각각 리스너 등록
        manual_watch = db.collection(manual_col).on_snapshot(on_zone_snapshot)
        auto_watch = db.collection(auto_col).on_snapshot(on_zone_snapshot)

        print(f"✅ [Firestore Listener] 두 컬렉션 모두 감시 중")

        while not stop_event.is_set():
            time.sleep(0.5)

        try:
            manual_watch.unsubscribe()
            auto_watch.unsubscribe()
            print("🔴 [Firestore Listener] 모든 감시 종료")
        except Exception as e:
            print(f"⚠️ [Firestore Listener unsubscribe warning] {e}")

    except Exception as e:
        print(f"⚠️ [Firestore Listener Error] {e}")
        print("⚠️ Firestore listener 없이 작동 중일 수 있습니다.")
