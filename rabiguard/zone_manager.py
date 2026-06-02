# zone_manager.py

import time
import queue
import json
from datetime import datetime

import cv2
import numpy as np
import uuid

try:
    from .config import DEPTH_SIMILARITY_THRESHOLD, SAVE_DIR, vlm_queue, ZONES_CONFIG_PATH
except ImportError:
    from config import DEPTH_SIMILARITY_THRESHOLD, SAVE_DIR, vlm_queue, ZONES_CONFIG_PATH


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


def get_polygon_depth(depth_map, polygon):
    """
    다각형(Polygon) 영역 내의 평균 depth 계산.
    polygon은 원본 640x480 좌표계 기준입니다.
    """
    if depth_map is None or len(polygon) == 0:
        return 0.0

    h, w = depth_map.shape

    # 원본(640x480) -> Depth 맵 크기로 스케일링
    scale_y = h / 480.0
    scale_x = w / 640.0
    scaled_poly = (polygon * [scale_x, scale_y]).astype(np.int32)

    # 마스크 생성 및 다각형 채우기
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [scaled_poly], 255)

    # 마스크 영역의 유효한(>0.1) depth 값만 추출
    roi_depth_values = depth_map[mask > 0]
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
        polygon_raw = data.get("polygon", [])

        if polygon_raw:
            # Firestore의 [{"x": 1, "y": 2}, ...] 형식을 [[1, 2], ...] 형식으로 변환
            if len(polygon_raw) > 0 and isinstance(polygon_raw[0], dict):
                polygon = [[p.get("x", 0), p.get("y", 0)] for p in polygon_raw]
            else:
                polygon = polygon_raw
            
            self.polygon = np.array(polygon, np.int32)
        else:
            self.polygon = np.array([])

        self.enter_threshold_sec = float(data.get("enter_threshold_sec", 2.0))
        self.min_people = int(data.get("min_people", 1))
        self.is_active = bool(data.get("is_active", True))

    def to_dict(self):
        """
        객체 정보를 딕셔너리로 변환합니다. (파일 저장용)
        """
        return {
            "polygon": self.polygon.tolist() if self.polygon.size > 0 else [],
            "enter_threshold_sec": self.enter_threshold_sec,
            "min_people": self.min_people,
            "is_active": self.is_active
        }


# ------------------------------------------------------------
# ZoneManager class
# ------------------------------------------------------------

class ZoneManager:
    def __init__(self):
        self.zones = {}
        self.snapshot_event_queue = None

    def save_to_file(self):
        """
        현재 구역 정보를 zones_config.json 파일로 저장합니다.
        """
        try:
            data = {z_id: z.to_dict() for z_id, z in self.zones.items()}
            with open(ZONES_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"💾 [ZoneManager] 구역 설정이 파일에 저장되었습니다: {ZONES_CONFIG_PATH}")
        except Exception as e:
            print(f"⚠️ [ZoneManager] 파일 저장 중 오류 발생: {e}")

    def process_queue_events(self, payload):
        """
        Firestore listener가 넣어준 queue payload를 내부 zone 상태에 반영하고 파일로 저장합니다.
        """
        action = payload.get("action")
        zone_id = payload.get("zone_id")

        if not action or not zone_id:
            return

        changed = False
        if action == "update":
            data = payload.get("data", {})

            if zone_id in self.zones:
                self.zones[zone_id].update(data)
                print(f"🔄 [ZoneManager] 구역 갱신됨: {zone_id}")
            else:
                self.zones[zone_id] = Zone(zone_id, data)
                print(f"➕ [ZoneManager] 구역 추가됨: {zone_id}")
            changed = True

        elif action == "delete":
            if zone_id in self.zones:
                del self.zones[zone_id]
                print(f"❌ [ZoneManager] 구역 삭제됨: {zone_id}")
                changed = True

        if changed:
            self.save_to_file()

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
                z_depth = get_polygon_depth(depth_map, zone.polygon)

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

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # 고유 이벤트 ID 생성
                event_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"

                print(
                    f"✅ [{zone_id}] ID {t_id} 조건 충족! EventID: {event_id}"
                )

                # frame_raw를 BGR로 변환
                frame_bgr = cv2.cvtColor(frame_raw, color_conv)

                # 구역(Zone) 주변 크롭 좌표 계산 (패딩 포함)
                rx, ry, rw, rh = cv2.boundingRect(zone.polygon)
                pad_x = 20
                pad_y = 20
                h, w = frame_bgr.shape[:2]

                crop_x1 = max(0, rx - pad_x)
                crop_y1 = max(0, ry - pad_y)
                crop_x2 = min(w, rx + rw + pad_x)
                crop_y2 = min(h, ry + rh + pad_y)

                # VLM용 클린 크롭 이미지 생성
                ctx_cropped = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy()

                # 디버그용 오버레이 이미지 생성 (로그 확인용)
                ctx_overlay = ctx_cropped.copy()
                cv2.rectangle(
                    ctx_overlay,
                    (x1 - crop_x1, y1 - crop_y1),
                    (x2 - crop_x1, y2 - crop_y1),
                    (0, 0, 255), 2
                )
                poly_offset = zone.polygon - [crop_x1, crop_y1]
                cv2.polylines(ctx_overlay, [poly_offset.astype(np.int32)], True, (255, 0, 0), 2)

                orig_path = SAVE_DIR / f"{zone_id}_ID_{t_id}_{timestamp}_orig.jpg"
                over_path = SAVE_DIR / f"{zone_id}_ID_{t_id}_{timestamp}_over.jpg"

                cv2.imwrite(str(orig_path), frame_bgr)
                cv2.imwrite(str(over_path), ctx_overlay)

                # 스냅샷 캡처 요청 (main.py의 전역 버퍼 사용)
                if self.snapshot_event_queue:
                    try:
                        # main.py에서 정의될 전역 변수들에 접근
                        import main
                        with main.snapshot_lock:
                            before_frames = list(main.snapshot_ring_buffer)
                        
                        self.snapshot_event_queue.put(
                            {"event_id": event_id, "before_frames": before_frames}
                        )
                    except Exception as e:
                        print(f"⚠️ [Snapshot Queue Error] {e}")

                try:
                    if vlm_queue.full():
                        try:
                            vlm_queue.get_nowait()
                            vlm_queue.task_done()
                        except Exception:
                            pass

                    vlm_queue.put_nowait(
                        {
                            "image": ctx_cropped,
                            "track_id": t_id,
                            "p_depth": p_depth,
                            "z_depth": z_depth,
                            "zone_id": zone_id,
                            "event_id": event_id,
                            "image_path": str(over_path),
                            "original_image_path": str(orig_path),
                            "people_count": len(people_in_zone),
                            "enter_threshold_sec": zone.enter_threshold_sec,
                        }
                    )

                except queue.Full:
                    print("⚠️ [VLM Queue] 이전 이벤트 처리 중이라 현재 이벤트는 건너뜀")
