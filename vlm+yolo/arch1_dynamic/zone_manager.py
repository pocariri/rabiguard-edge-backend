import time
import cv2
import numpy as np
import queue
from datetime import datetime

from .config import DEPTH_SIMILARITY_THRESHOLD, SAVE_DIR, vlm_queue

# -----------------------------------------------------------------------
# 유틸리티: Depth 연산
# -----------------------------------------------------------------------
def get_roi_depth(depth_map, x1, y1, x2, y2):
    if depth_map is None: return 0.0
    h, w = depth_map.shape
    scale_y, scale_x = h / 480.0, w / 640.0
    tx1, ty1 = int(x1 * scale_x), int(y1 * scale_y)
    tx2, ty2 = int(x2 * scale_x), int(y2 * scale_y)
    ty1, ty2 = max(0, ty1), min(h, ty2)
    tx1, tx2 = max(0, tx1), min(w, tx2)
    if tx1 >= tx2 or ty1 >= ty2: return 0.0
    
    roi_depth_values = depth_map[ty1:ty2, tx1:tx2].flatten()
    roi_depth_values = roi_depth_values[roi_depth_values > 0.1]
    if len(roi_depth_values) == 0: return 0.0
    return float(np.mean(roi_depth_values))

# -----------------------------------------------------------------------
# Phase 3: 개별 구역(Zone) 클래스
# -----------------------------------------------------------------------
class Zone:
    def __init__(self, zone_id, data):
        self.zone_id = zone_id
        self.update(data)
        self.tracker_state = {}  # { track_id: {"enter_time": float, "notified": bool} }

    def update(self, data):
        """Firestore Payload 데이터를 바탕으로 속성 갱신"""
        self.polygon = np.array(data["polygon"], np.int32) if data["polygon"] else np.array([])
        self.enter_threshold_sec = float(data.get("enter_threshold_sec", 2.0))
        self.min_people = int(data.get("min_people", 1))
        self.is_active = bool(data.get("is_active", True))

# -----------------------------------------------------------------------
# Phase 3: 구역 관리자(ZoneManager) 클래스
# -----------------------------------------------------------------------
class ZoneManager:
    def __init__(self):
        self.zones = {}  # { zone_id: Zone_Object }

    def process_queue_events(self, payload):
        """큐에서 가져온 설정을 내부 메모리 상태에 반영합니다."""
        action = payload.get("action")
        zone_id = payload.get("zone_id")
        
        if action == "update":
            if zone_id in self.zones:
                self.zones[zone_id].update(payload["data"])
                print(f"🔄 [ZoneManager] 구역 갱신됨: {zone_id}")
            else:
                self.zones[zone_id] = Zone(zone_id, payload["data"])
                print(f"➕ [ZoneManager] 구역 추가됨: {zone_id}")
        elif action == "delete":
            if zone_id in self.zones:
                del self.zones[zone_id]
                print(f"❌ [ZoneManager] 구역 삭제됨: {zone_id}")

    def check_zones(self, results, depth_map, frame_raw, color_conv, w_orig, h_orig):
        """매 프레임마다 여러 구역을 일괄 검사합니다."""
        
        # 감지된 객체가 없으면 모든 구역의 타이머(tracker_state)를 초기화
        if results[0].boxes is None or results[0].boxes.id is None:
            for z in self.zones.values():
                z.tracker_state.clear()
            return

        boxes_small = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        # 1. 320 해상도 좌표를 원본으로 복원하고 중앙점 계산 (전처리)
        persons = []
        for box, track_id in zip(boxes_small, track_ids):
            x1 = int(box[0] * (w_orig / 320))
            y1 = int(box[1] * (h_orig / 320))
            x2 = int(box[2] * (w_orig / 320))
            y2 = int(box[3] * (h_orig / 320))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            persons.append({
                "id": track_id, 
                "bbox": (x1, y1, x2, y2), 
                "center": (cx, cy)
            })

        current_time = time.time()

        # 2. 각 구역(Zone)별로 독립적으로 검사 진행
        for zone_id, zone in self.zones.items():
            if not zone.is_active or len(zone.polygon) == 0:
                zone.tracker_state.clear()
                continue

            # 이 구역 폴리곤 안에 있는 사람(점) 필터링
            people_in_zone = [p for p in persons if cv2.pointPolygonTest(zone.polygon, p["center"], False) >= 0]
            current_ids_in_zone = set(p["id"] for p in people_in_zone)

            # 진입 시간 기록 (신규 진입자)
            for p in people_in_zone:
                t_id = p["id"]
                if t_id not in zone.tracker_state:
                    zone.tracker_state[t_id] = {"enter_time": current_time, "notified": False}
                    print(f"⚠️ [{zone_id}] ID {t_id} 진입 감지")

            # 구역을 벗어난 사람의 타이머 삭제
            for d_id in list(zone.tracker_state.keys() - current_ids_in_zone):
                del zone.tracker_state[d_id]

            # 3. 구역 이벤트 조건 검사 (최소 인원 이상일 때만 시간 검사)
            if len(people_in_zone) >= zone.min_people:
                for p in people_in_zone:
                    t_id = p["id"]
                    state = zone.tracker_state[t_id]

                    # 체류 시간 충족 여부 확인
                    if current_time - state["enter_time"] >= zone.enter_threshold_sec and not state["notified"]:
                        x1, y1, x2, y2 = p["bbox"]
                        
                        # Depth 유사성 검사
                        p_depth = get_roi_depth(depth_map, x1, y1, x2, y2)
                        rx, ry, rw, rh = cv2.boundingRect(zone.polygon)
                        z_depth = get_roi_depth(depth_map, rx, ry, rx+rw, ry+rh)

                        if abs(p_depth - z_depth) <= DEPTH_SIMILARITY_THRESHOLD:
                            state["notified"] = True  # 중복 알림 방지
                            print(f"🚨 [{zone_id}] ID {t_id} 조건 충족! (체류:{zone.enter_threshold_sec}s, 인원:{len(people_in_zone)}명)")
                            
                            # VLM을 위해 BGR 변환 및 오버레이 그리기
                            frame_bgr = cv2.cvtColor(frame_raw, color_conv)
                            ctx = frame_bgr.copy()
                            cv2.rectangle(ctx, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.polylines(ctx, [zone.polygon], True, (255, 0, 0), 2)
                            
                            # 디버그용 이미지 저장
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            cv2.imwrite(str(SAVE_DIR / f"{zone_id}_ID_{t_id}_{timestamp}_orig.jpg"), frame_bgr)
                            cv2.imwrite(str(SAVE_DIR / f"{zone_id}_ID_{t_id}_{timestamp}_over.jpg"), ctx)

                            # VLM 큐에 전송 (Payload 형식으로 포장)
                            try:
                                vlm_queue.put_nowait({
                                    "image": ctx,
                                    "track_id": t_id,
                                    "p_depth": p_depth,
                                    "z_depth": z_depth,
                                    "zone_id": zone_id
                                })
                            except queue.Full:
                                pass
