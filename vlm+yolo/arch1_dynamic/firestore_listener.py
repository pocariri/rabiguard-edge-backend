import threading
import firebase_admin
from firebase_admin import credentials, firestore
from .config import zone_config_queue, stop_event

def start_firestore_listener():
    """
    Phase 2: Firestore 비동기 리스너 스레드 구현
    네트워크 I/O만 담당하므로 메인 비전 루프(CPU)에 영향을 주지 않습니다.
    """
    print("🌐 [Firestore Listener] 초기화 중...")
    
    # Firebase 초기화 (실제 서비스 계정 키 필요 시 credential 경로 수정)
    try:
        firebase_admin.get_app()
    except ValueError:
        try:
            # 기본 애플리케이션 자격 증명 시도
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred)
        except Exception as e:
            print(f"⚠️ [Firestore] 초기화 실패 (키 확인 필요): {e}")
            print("⚠️ [Firestore] 더미 모드로 전환합니다 (수동으로 큐에 데이터를 넣어야 합니다).")
            return

    db = firestore.client()
    col_ref = db.collection('Zones')

    def on_snapshot(col_snapshot, changes, read_time):
        for change in changes:
            doc = change.document
            data = doc.to_dict()
            
            if change.type.name in ['ADDED', 'MODIFIED']:
                payload = {
                    "action": "update",
                    "zone_id": doc.id,
                    "data": {
                        "polygon": data.get("polygon", []),
                        "enter_threshold_sec": data.get("enter_threshold_sec", 2.0),
                        "min_people": data.get("min_people", 1),
                        "is_active": data.get("is_active", True)
                    }
                }
            elif change.type.name == 'REMOVED':
                payload = {
                    "action": "delete",
                    "zone_id": doc.id
                }
            else:
                continue
                
            # Phase 1: 표준화된 페이로드를 큐에 푸시
            zone_config_queue.put(payload)
            print(f"📥 [Firestore] 큐에 이벤트 푸시 완료: {change.type.name} -> {doc.id}")

    # 리스너 부착
    watch = col_ref.on_snapshot(on_snapshot)
    print("✅ [Firestore Listener] 구독 시작 완료!")

    # 메인 스레드가 종료될 때까지 대기
    while not stop_event.is_set():
        stop_event.wait(1.0)
        
    watch.unsubscribe()
    print("🛑 [Firestore Listener] 종료 완료")
