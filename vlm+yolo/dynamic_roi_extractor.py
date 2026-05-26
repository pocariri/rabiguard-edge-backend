import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLOE
except ImportError:
    from ultralytics import YOLO as YOLOE

class DynamicROIExtractor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = Path(__file__).parent.parent / "yoloe_tests" / "yoloe-26n-seg-pf.pt"
        
        print(f"📦 [ROI Extractor] 모델 로드 중: {model_path}")
        self.model = YOLOE(str(model_path))
        self.target_size = (640, 480) # arch1_headless 좌표계 기준

    def extract_candidates(self, frame, env_tag_list=None):
        """프레임에서 ROI 후보를 추출합니다."""
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
            if env_tag_list and class_name not in env_tag_list:
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
                "points": approx_polygon.tolist(), # JSON 저장을 위해 list 변환
                "crop": crop_img,
                "bbox": [x1, y1, x2, y2]
            })
        return candidates

    def save_to_config(self, candidates, config_path="zones_config.json"):
        """추출된 후보들을 zones_config.json 형식으로 저장합니다."""
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

if __name__ == "__main__":
    extractor = DynamicROIExtractor()
    print("📸 [ROI Extractor] 카메라 프레임 캡처 중...")
    cap = cv2.VideoCapture(0)
    # 카메라 안정화를 위해 몇 프레임 건너뜀
    for _ in range(5): cap.read()
    ret, frame = cap.read()
    cap.release()

    if ret:
        # 1. 구역 추출
        rois = extractor.extract_candidates(frame)
        if rois:
            # 2. 설정 파일에 저장 (이 시점에 main_local이 감지하고 업데이트함)
            extractor.save_to_config(rois)
            
            # 3. 디버그용 크롭 이미지 저장
            out_dir = Path("_outputs/extracted_rois")
            out_dir.mkdir(parents=True, exist_ok=True)
            for roi in rois:
                cv2.imwrite(str(out_dir / f"{roi['id']}.png"), roi['crop'])
            print(f"🖼️ [ROI Extractor] 개별 구역 이미지가 '{out_dir}'에 저장되었습니다.")
        else:
            print("⚠️ [ROI Extractor] 감지된 구역이 없습니다.")
    else:
        print("❌ [ROI Extractor] 카메라를 읽을 수 없습니다.")
