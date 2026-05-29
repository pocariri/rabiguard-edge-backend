# dynamic_roi_extractor.py

import cv2
import json
import numpy as np
import sys
from pathlib import Path

# 직접 실행 시 상위 패키지 인식을 위한 경로 추가
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from .config import (
        YOLOE_MODEL_PATH,
        TARGET_OBJECTS_PATH,
        ZONES_CONFIG_PATH,
        EXTRACTED_ROIS_DIR,
    )
    from .firebase_writer import save_zones_to_firestore
except (ImportError, ValueError):
    from config import (
        YOLOE_MODEL_PATH,
        TARGET_OBJECTS_PATH,
        ZONES_CONFIG_PATH,
        EXTRACTED_ROIS_DIR,
    )
    from firebase_writer import save_zones_to_firestore

try:
    from ultralytics import YOLOE
except ImportError:
    from ultralytics import YOLO as YOLOE


class DynamicROIExtractor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = YOLOE_MODEL_PATH

        print(f"📦 [ROI Extractor] 모델 로드 중: {model_path}")
        self.model = YOLOE(str(model_path))
        self.target_size = (640, 480)  # arch1_headless 좌표계 기준

        # 필터링할 객체 목록 로드
        self.target_list_path = TARGET_OBJECTS_PATH
        self.target_objects = self.load_target_objects()

    def load_target_objects(self):
        """target_objects.txt 파일을 읽어 필터링 목록을 생성합니다."""
        if not self.target_list_path.exists():
            default_tags = [
                "bed",
                "chair",
                "couch",
                "dining table",
                "refrigerator",
                "tv",
                "toilet",
            ]

            with open(self.target_list_path, "w", encoding="utf-8") as f:
                f.write("\n".join(default_tags))

            print(f"📝 [ROI Extractor] 기본 필터 목록 생성됨: {self.target_list_path}")
            return set(default_tags)

        with open(self.target_list_path, "r", encoding="utf-8") as f:
            tags = {line.strip().lower() for line in f if line.strip()}

        print(f"📋 [ROI Extractor] {len(tags)}개의 타겟 객체 로드 완료")
        return tags

    def extract_candidates(self, frame):
        """프레임에서 target_objects에 포함된 ROI 후보만 추출합니다."""
        frame_resized = cv2.resize(frame, self.target_size)
        results = self.model.predict(frame_resized, verbose=False)

        candidates = []

        if not results or results[0].masks is None:
            return candidates

        masks = results[0].masks
        boxes = results[0].boxes
        names = self.model.names

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            class_name = names[cls_id].lower()

            # 필터링 목록에 있는지 확인
            if class_name not in self.target_objects:
                continue

            polygon = masks.xy[i].astype(np.int32)
            epsilon = 0.01 * cv2.arcLength(polygon, True)
            approx_polygon = cv2.approxPolyDP(polygon, epsilon, True).reshape(-1, 2)

            # 크롭 및 이미지 처리
            mask = np.zeros(self.target_size[::-1], dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)

            bgra = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = mask

            x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())

            crop_img = bgra[
                max(0, y1 - 10):min(480, y2 + 10),
                max(0, x1 - 10):min(640, x2 + 10),
            ].copy()

            obj_id = f"{class_name}_{i}"

            candidates.append(
                {
                    "id": obj_id,
                    "class_name": class_name,
                    "points": approx_polygon.tolist(),
                    "crop": crop_img,
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return candidates

    def save_original_image(self, frame, save_path=None):
        """
        자동구역 객체 인식에 사용된 원본 프레임을 저장합니다.
        SSH/headless 환경에서 orig.jpg로 확인할 수 있습니다.
        """
        if save_path is None:
            save_path = EXTRACTED_ROIS_DIR / "orig.jpg"
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)

        original_resized = cv2.resize(frame, self.target_size)
        cv2.imwrite(str(save_path), original_resized)

        print(f"🖼️ [ROI Extractor] 원본 이미지 저장 완료: {save_path}")

    def save_detected_result_image(self, frame, candidates, save_path=None):
        """
        SSH/headless 환경에서 확인할 수 있도록
        원본 프레임 위에 bbox, polygon, 객체명을 표시한 이미지를 저장합니다.
        """
        if save_path is None:
            save_path = EXTRACTED_ROIS_DIR / "detected_result.jpg"
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)

        display = cv2.resize(frame, self.target_size)

        for cand in candidates:
            class_name = cand["class_name"]
            x1, y1, x2, y2 = cand["bbox"]
            points = np.array(cand["points"], dtype=np.int32)

            # 바운딩 박스 표시
            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # polygon ROI 영역 표시
            cv2.polylines(
                display,
                [points],
                isClosed=True,
                color=(255, 0, 0),
                thickness=2,
            )

            label = class_name

            text_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )

            text_w, text_h = text_size
            label_x = x1
            label_y = max(25, y1 - 10)

            # 라벨 배경
            cv2.rectangle(
                display,
                (label_x, label_y - text_h - 8),
                (label_x + text_w + 6, label_y + 4),
                (0, 255, 0),
                -1,
            )

            # 라벨 텍스트
            cv2.putText(
                display,
                label,
                (label_x + 3, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        cv2.imwrite(str(save_path), display)
        print(f"🖼️ [ROI Extractor] 인식 결과 이미지 저장 완료: {save_path}")

    def save_to_config(self, candidates, config_path=None):
        """추출된 후보들을 zones_config.json 형식으로 저장합니다."""
        if config_path is None:
            config_path = ZONES_CONFIG_PATH

        config = {}

        for cand in candidates:
            config[cand["id"]] = {
                "polygon": cand["points"],
                "enter_threshold_sec": 2.0,
                "min_people": 1,
                "is_active": True,
                "class_name": cand["class_name"],
            }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        print(f"💾 [ROI Extractor] {len(candidates)}개의 구역을 '{config_path}'에 저장했습니다.")

    def save_to_firebase(
        self,
        candidates,
        collection_name="auto_zones",
        reset_before_save=True,
    ):
        """
        추출된 후보들을 Firebase Firestore에 저장합니다.

        reset_before_save=True:
        - 저장 전에 기존 auto_zones 컬렉션 문서를 삭제합니다.
        - 실행할 때마다 새로 인식된 객체만 Firestore에 남습니다.
        """
        zones_data = {}

        for cand in candidates:
            # Firestore는 중첩 배열(Nested Array)을 지원하지 않으므로 객체 리스트로 변환
            polygon_dicts = [
                {
                    "x": int(p[0]),
                    "y": int(p[1]),
                }
                for p in cand["points"]
            ]

            zones_data[cand["id"]] = {
                "polygon": polygon_dicts,
                "enter_threshold_sec": 2.0,
                "min_people": 1,
                "is_active": True,
                "class_name": cand["class_name"],
            }

        save_zones_to_firestore(
            zones_data,
            collection_name=collection_name,
            reset_before_save=reset_before_save,
        )


if __name__ == "__main__":
    extractor = DynamicROIExtractor()

    print("📸 [ROI Extractor] 카메라 프레임 캡처 중...")

    cap = cv2.VideoCapture(0)

    # 카메라 자동 노출 안정화를 위해 몇 프레임 버림
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if ret:
        out_dir = EXTRACTED_ROIS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. 자동구역 인식에 사용한 원본 이미지 저장
        extractor.save_original_image(
            frame,
            save_path=out_dir / "orig.jpg",
        )

        # 2. 자동구역 후보 추출
        rois = extractor.extract_candidates(frame)

        if rois:
            # 3. 인식 결과 이미지 저장
            extractor.save_detected_result_image(
                frame,
                rois,
                save_path=out_dir / "detected_result.jpg",
            )

            # 4. Firestore 저장
            # 실행할 때마다 기존 auto_zones 컬렉션을 초기화한 뒤 새 객체만 저장
            extractor.save_to_firebase(
                rois,
                collection_name="auto_zones",
                reset_before_save=True,
            )

            # 5. 로컬 JSON 백업 저장
            extractor.save_to_config(rois)

        else:
            print(f"⚠️ [ROI Extractor] 타겟 목록({extractor.target_objects})에 해당하는 객체를 찾지 못했습니다.")

            # 감지 객체가 없어도 detected_result.jpg는 원본과 동일하게 저장
            extractor.save_detected_result_image(
                frame,
                [],
                save_path=out_dir / "detected_result.jpg",
            )

            # 감지 객체가 없는 경우에도 Firestore auto_zones를 비움
            extractor.save_to_firebase(
                [],
                collection_name="auto_zones",
                reset_before_save=True,
            )

    else:
        print("❌ [ROI Extractor] 카메라를 읽을 수 없습니다.")