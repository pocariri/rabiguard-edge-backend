import json
import time
import os
from .config import zone_config_queue, stop_event

def start_local_listener(config_path="zones_config.json"):
    """
    로컬 JSON 파일을 감시하여 구역 설정을 업데이트하는 리스너.
    파일이 수정되면 자동으로 큐에 이벤트를 푸시합니다.
    """
    print(f"📁 [Local Listener] '{config_path}' 감시 시작...")
    
    last_mtime = 0
    
    while not stop_event.is_set():
        if not os.path.exists(config_path):
            # 파일이 없으면 기본 예시 파일 생성
            example_config = {
                "Zone_Local_1": {
                    "polygon": [[100, 100], [540, 100], [540, 380], [100, 380]],
                    "enter_threshold_sec": 2.0,
                    "min_people": 1,
                    "is_active": True
                }
            }
            with open(config_path, "w") as f:
                json.dump(example_config, f, indent=4)
            print(f"📝 [Local Listener] 기본 설정 파일을 생성했습니다: {config_path}")

        try:
            mtime = os.path.getmtime(config_path)
            if mtime > last_mtime:
                last_mtime = mtime
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                
                # 기존 구역 삭제 또는 갱신을 위해 현재 메모리 상태와 비교할 수도 있지만,
                # 여기서는 파일에 있는 모든 구역을 'update' 액션으로 보냅니다.
                for zone_id, data in config_data.items():
                    payload = {
                        "action": "update",
                        "zone_id": zone_id,
                        "data": data
                    }
                    zone_config_queue.put(payload)
                
                print(f"📥 [Local Listener] {len(config_data)}개의 구역 설정 로드 완료")
        except Exception as e:
            print(f"⚠️ [Local Listener] 오류 발생: {e}")

        # 2초마다 파일 변경 확인
        time.sleep(2.0)

    print("🛑 [Local Listener] 종료")
