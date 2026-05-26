import cv2
import json
import numpy as np
import os
from pathlib import Path
from datetime import datetime

try:
    from .config import YOLOE_MODEL_PATH, TARGET_OBJECTS_PATH, ZONES_CONFIG_PATH, EXTRACTED_ROIS_DIR
    from .firebase_writer import save_zones_to_firestore
except ImportError:
    from config import YOLOE_MODEL_PATH, TARGET_OBJECTS_PATH, ZONES_CONFIG_PATH, EXTRACTED_ROIS_DIR
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
        self.target_size = (640, 480) # arch1_headless 좌표계 기준
        
        # 필터링할 객체 목록 로드
        self.target_list_path = TARGET_OBJECTS_PATH
        self.target_objects = self.load_target_objects()

    def load_target_objects(self):
        """target_objects.txt 파일을 읽어 필터링 목록을 생성합니다."""
        if not self.target_list_path.exists():
            default_tags = ["bed", "chair", "couch", "dining table", "refrigerator", "tv", "toilet"]
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

            conf = float(boxes.conf[i].item())
            polygon = masks.xy[i].astype(np.int32)
            epsilon = 0.01 * cv2.arcLength(polygon, True)
            approx_polygon = cv2.approxPolyDP(polygon, epsilon, True).reshape(-1, 2)

            # 크롭 및 이미지 처리
            mask = np.zeros(self.target_size[::-1], dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            bgra = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = mask
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
            crop_img = bgra[max(0, y1-10):min(480, y2+10), max(0, x1-10):min(640, x2+10)].copy()

            obj_id = f"{class_name}_{i}"
            candidates.append({
                "id": obj_id,
                "class_name": class_name,
                "points": approx_polygon.tolist(),
                "crop": crop_img,
                "bbox": [x1, y1, x2, y2]
            })
        return candidates

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
                "class_name": cand["class_name"]
            }
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print(f"💾 [ROI Extractor] {len(candidates)}개의 구역을 '{config_path}'에 저장했습니다.")

    def save_to_firebase(self, candidates, collection_name="zones"):
        """추출된 후보들을 Firebase Firestore에 저장합니다."""
        zones_data = {}
        for cand in candidates:
            zones_data[cand["id"]] = {
                "polygon": cand["points"],
                "enter_threshold_sec": 2.0,
                "min_people": 1,
                "is_active": True,
                "class_name": cand["class_name"]
            }
        
        save_zones_to_firestore(zones_data, collection_name=collection_name)

if __name__ == "__main__":
    extractor = DynamicROIExtractor()
    print("📸 [ROI Extractor] 카메라 프레임 캡처 중...")
    cap = cv2.VideoCapture(0)
    for _ in range(5): cap.read()
    ret, frame = cap.read()
    cap.release()

    if ret:
        rois = extractor.extract_candidates(frame)
        if rois:
            # Firebase에 저장
            extractor.save_to_firebase(rois)
            
            # 로컬 파일로도 백업 저장
            extractor.save_to_config(rois)
            
            out_dir = EXTRACTED_ROIS_DIR
            out_dir.mkdir(parents=True, exist_ok=True)
            for roi in rois:
                cv2.imwrite(str(out_dir / f"{roi['id']}.png"), roi['crop'])
            print(f"🖼️ [ROI Extractor] 개별 구역 이미지가 '{out_dir}'에 저장되었습니다.")
        else:
            print(f"⚠️ [ROI Extractor] 타겟 목록({extractor.target_objects})에 해당하는 객체를 찾지 못했습니다.")
    else:
        print("❌ [ROI Extractor] 카메라를 읽을 수 없습니다.")
