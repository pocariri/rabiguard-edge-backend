# zone_manager.py

import time
import queue
from datetime import datetime

import cv2
import numpy as np

try:
    from .config import DEPTH_SIMILARITY_THRESHOLD, SAVE_DIR, vlm_queue
except ImportError:
    from config import DEPTH_SIMILARITY_THRESHOLD, SAVE_DIR, vlm_queue


# ------------------------------------------------------------
# Depth utility
# ------------------------------------------------------------

def get_roi_depth(depth_map, x1, y1, x2, y2):
    """
    원본 640x480 좌표 기준 bbox 영역의 평균 depth 계산.
    depth_map은 보통 320x256 형태로 들어온다고 가정합니다.
    """
    if depth_map is None:
        return 0.0

    h, w = depth_map.shape

    scale_y = h / 480.0
    scale_x = w / 640.0

    tx1 = int(x1 * scale_x)
    ty1 = int(y1 * scale_y)
    tx2 = int(x2 * scale_x)
    ty2 = int(y2 * scale_y)

    ty1 = max(0, ty1)
    ty2 = min(h, ty2)
    tx1 = max(0, tx1)
    tx2 = min(w, tx2)

    if tx1 >= tx2 or ty1 >= ty2:
        return 0.0

    roi_depth_values = depth_map[ty1:ty2, tx1:tx2].flatten()
    roi_depth_values = roi_depth_values[roi_depth_values > 0.1]

    if len(roi_depth_values) == 0:
        return 0.0

    return float(np.mean(roi_depth_values))


# ------------------------------------------------------------
# Zone class
# ------------------------------------------------------------

class Zone:
    def __init__(self, zone_id, data):
        self.zone_id = zone_id
        self.tracker_state = {}
        self.update(data)

    def update(self, data):
        """
        Firestore payload 데이터를 바탕으로 구역 속성을 갱신합니다.
        """
        polygon = data.get("polygon", [])

        if polygon:
            self.polygon = np.array(polygon, np.int32)
        else:
            self.polygon = np.array([])

        self.enter_threshold_sec = float(data.get("enter_threshold_sec", 2.0))
        self.min_people = int(data.get("min_people", 1))
        self.is_active = bool(data.get("is_active", True))


# ------------------------------------------------------------
# ZoneManager class
# ------------------------------------------------------------

class ZoneManager:
    def __init__(self):
        self.zones = {}

    def process_queue_events(self, payload):
        """
        Firestore listener가 넣어준 queue payload를 내부 zone 상태에 반영합니다.
        """
        action = payload.get("action")
        zone_id = payload.get("zone_id")

        if not action or not zone_id:
            return

        if action == "update":
            data = payload.get("data", {})

            if zone_id in self.zones:
                self.zones[zone_id].update(data)
                print(f"🔄 [ZoneManager] 구역 갱신됨: {zone_id}")
            else:
                self.zones[zone_id] = Zone(zone_id, data)
                print(f"➕ [ZoneManager] 구역 추가됨: {zone_id}")

        elif action == "delete":
            if zone_id in self.zones:
                del self.zones[zone_id]
                print(f"❌ [ZoneManager] 구역 삭제됨: {zone_id}")

    def check_zones(self, results, depth_map, frame_raw, color_conv, w_orig, h_orig):
        """
        YOLO 결과를 바탕으로 각 구역별 진입/체류/Depth 조건을 검사합니다.
        조건 충족 시 VLM queue로 이벤트 이미지를 전달합니다.
        """
        if not self.zones:
            return

        # 감지 객체가 없으면 모든 zone의 tracker_state 초기화
        if results[0].boxes is None or results[0].boxes.id is None:
            for zone in self.zones.values():
                zone.tracker_state.clear()
            return

        boxes_small = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        persons = []

        for box, track_id in zip(boxes_small, track_ids):
            x1 = int(box[0] * (w_orig / 320))
            y1 = int(box[1] * (h_orig / 320))
            x2 = int(box[2] * (w_orig / 320))
            y2 = int(box[3] * (h_orig / 320))

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            persons.append(
                {
                    "id": int(track_id),
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                }
            )

        current_time = time.time()

        for zone_id, zone in self.zones.items():
            if not zone.is_active or len(zone.polygon) == 0:
                zone.tracker_state.clear()
                continue

            people_in_zone = [
                p for p in persons
                if cv2.pointPolygonTest(zone.polygon, p["center"], False) >= 0
            ]

            current_ids_in_zone = set(p["id"] for p in people_in_zone)

            # 신규 진입자 기록
            for person in people_in_zone:
                t_id = person["id"]

                if t_id not in zone.tracker_state:
                    zone.tracker_state[t_id] = {
                        "enter_time": current_time,
                        "notified": False,
                    }
                    print(f"⚠️ [{zone_id}] ID {t_id} 진입 감지")

            # 구역 이탈 객체 제거
            for disappeared_id in list(set(zone.tracker_state.keys()) - current_ids_in_zone):
                print(f"🔵 [{zone_id}] ID {disappeared_id} 이탈")
                del zone.tracker_state[disappeared_id]

            # 최소 인원 조건 확인
            if len(people_in_zone) < zone.min_people:
                continue

            for person in people_in_zone:
                t_id = person["id"]
                state = zone.tracker_state[t_id]

                elapsed = current_time - state["enter_time"]

                if elapsed < zone.enter_threshold_sec:
                    continue

                if state["notified"]:
                    continue

                x1, y1, x2, y2 = person["bbox"]

                p_depth = get_roi_depth(depth_map, x1, y1, x2, y2)

                rx, ry, rw, rh = cv2.boundingRect(zone.polygon)
                z_depth = get_roi_depth(depth_map, rx, ry, rx + rw, ry + rh)

                if p_depth <= 0 or z_depth <= 0:
                    print(
                        f"⚠️ [{zone_id}] Depth 부족: "
                        f"ID={t_id}, person={p_depth:.2f}, zone={z_depth:.2f}"
                    )
                    state["enter_time"] = current_time
                    continue

                diff = abs(p_depth - z_depth)

                if diff > DEPTH_SIMILARITY_THRESHOLD:
                    print(
                        f"❌ [{zone_id}] Depth 검증 실패: "
                        f"ID={t_id}, person={p_depth:.2f}, "
                        f"zone={z_depth:.2f}, diff={diff:.2f}"
                    )
                    state["enter_time"] = current_time
                    continue

                state["notified"] = True

                print(
                    f"✅ [{zone_id}] ID {t_id} 조건 충족! "
                    f"체류={zone.enter_threshold_sec}s, "
                    f"인원={len(people_in_zone)}명, "
                    f"depth diff={diff:.2f}"
                )

                # frame_raw를 BGR로 변환
                frame_bgr = cv2.cvtColor(frame_raw, color_conv)

                # VLM용 context 이미지 생성
                ctx = frame_bgr.copy()
                cv2.rectangle(ctx, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.polylines(ctx, [zone.polygon], True, (255, 0, 0), 2)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                orig_path = SAVE_DIR / f"{zone_id}_ID_{t_id}_{timestamp}_orig.jpg"
                over_path = SAVE_DIR / f"{zone_id}_ID_{t_id}_{timestamp}_over.jpg"

                cv2.imwrite(str(orig_path), frame_bgr)
                cv2.imwrite(str(over_path), ctx)

                try:
                    if vlm_queue.full():
                        try:
                            vlm_queue.get_nowait()
                            vlm_queue.task_done()
                        except Exception:
                            pass

                    vlm_queue.put_nowait(
                        {
                            "image": ctx,
                            "track_id": t_id,
                            "p_depth": p_depth,
                            "z_depth": z_depth,
                            "zone_id": zone_id,
                            "image_path": str(over_path),
                            "original_image_path": str(orig_path),
                            "people_count": len(people_in_zone),
                            "enter_threshold_sec": zone.enter_threshold_sec,
                        }
                    )

                except queue.Full:
                    print("⚠️ [VLM Queue] 이전 이벤트 처리 중이라 현재 이벤트는 건너뜀")