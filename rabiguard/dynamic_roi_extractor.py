# dynamic_roi_extractor.py

import cv2
import json
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# 직접 실행 시 상위 패키지 인식을 위한 경로 추가
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from .config import (
        YOLOE_MODEL_PATH,
        TARGET_OBJECTS_PATH,
        EXTRACTED_ROIS_DIR,
    )
    from .firebase_writer import save_zones_to_firestore
except (ImportError, ValueError):
    from config import (
        YOLOE_MODEL_PATH,
        TARGET_OBJECTS_PATH,
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

    def bbox_to_rect_points(self, x1, y1, x2, y2):
        """
        bbox 좌표를 zone_manager가 기존 polygon처럼 사용할 수 있는
        사각형 꼭짓점 4개로 변환합니다.

        순서:
        좌상단 -> 우상단 -> 우하단 -> 좌하단
        """
        return [
            [int(x1), int(y1)],
            [int(x2), int(y1)],
            [int(x2), int(y2)],
            [int(x1), int(y2)],
        ]

    def calculate_iou(self, box1, box2):
        """두 bbox 간의 IoU(Intersection over Union)를 계산합니다."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 교차 영역 좌표 계산
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        # 교차 영역 면적
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # 각 bbox 면적
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Union 면적 (합집합)
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0
        return inter_area / union_area

    def filter_candidates(self, candidates, iou_threshold=0.4):
        """
        면적이 큰 순서대로 정렬한 후, 겹치는 구역을 제거합니다.
        (Greedy Non-Maximum Suppression 변형)
        """
        if not candidates:
            return []

        # 1. 면적(Area) 기준으로 내림차순 정렬 (큰 것부터)
        candidates.sort(
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
            reverse=True
        )

        final_rois = []
        for cand in candidates:
            keep = True
            for accepted in final_rois:
                # 2. 이미 선택된 구역(더 큰 구역)과 겹치는지 확인
                iou = self.calculate_iou(cand["bbox"], accepted["bbox"])
                if iou > iou_threshold:
                    print(f"🚫 [ROI Extractor] {cand['id']} 제거됨 (겹침 비율: {iou:.2f} with {accepted['id']})")
                    keep = False
                    break
            
            if keep:
                final_rois.append(cand)
        
        return final_rois

    def extract_candidates(self, frame):
        """프레임에서 target_objects에 포함된 ROI 후보만 추출합니다."""
        frame_resized = cv2.resize(frame, self.target_size)
        results = self.model.predict(frame_resized, verbose=False)

        candidates = []

        if not results or results[0].boxes is None:
            return candidates

        boxes = results[0].boxes
        names = self.model.names
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            class_name = names[cls_id].lower()

            # 필터링 목록에 있는지 확인
            if class_name not in self.target_objects:
                continue

            x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())

            # 좌표가 이미지 범위를 벗어나지 않게 보정
            x1 = max(0, min(639, x1))
            y1 = max(0, min(479, y1))
            x2 = max(0, min(639, x2))
            y2 = max(0, min(479, y2))

            # bbox를 사각형 polygon 좌표로 변환
            rect_points = self.bbox_to_rect_points(x1, y1, x2, y2)

            # bbox crop 저장용 이미지
            crop_img = frame_resized[
                max(0, y1 - 10):min(480, y2 + 10),
                max(0, x1 - 10):min(640, x2 + 10),
            ].copy()

            obj_id = f"{timestamp}_{class_name}_{i}"

            candidates.append(
                {
                    "id": obj_id,
                    "class_name": class_name,
                    "points": rect_points,          # 기존 polygon 필드에 들어갈 사각형 꼭짓점
                    "crop": crop_img,
                    "bbox": [x1, y1, x2, y2],      # 앱/UI/추후 bbox mode용
                    "mode": "bbox",
                }
            )

        # 겹침 및 크기 기반 필터링 적용 (면적 순 정렬 + IoU 기반 중복 제거)
        filtered_candidates = self.filter_candidates(candidates)
        print(f"🎯 [ROI Extractor] 필터링 결과: {len(candidates)} -> {len(filtered_candidates)}개")

        return filtered_candidates

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
        원본 프레임 위에 bbox와 객체명을 표시한 이미지를 저장합니다.
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

            # bbox 자동구역 표시
            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = f"{class_name}"

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
                "mode": "bbox",
                "polygon": polygon_dicts,
                "enter_threshold_sec": 20.0,
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
