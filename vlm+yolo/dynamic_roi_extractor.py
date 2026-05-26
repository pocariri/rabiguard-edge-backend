import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLOE
except ImportError:
    # YOLOE 임포트 실패 시 일반 YOLO로 대체 시도 (환경에 따라 다를 수 있음)
    from ultralytics import YOLO as YOLOE

class DynamicROIExtractor:
    def __init__(self, model_path=None):
        if model_path is None:
            # 기본 경로 설정
            model_path = Path(__file__).parent.parent / "yoloe_tests" / "yoloe-26n-seg-pf.pt"
        
        print(f"📦 [ROI Extractor] 모델 로드 중: {model_path}")
        self.model = YOLOE(str(model_path))
        self.target_size = (640, 480) # arch1_headless 좌표계 기준

    def extract_candidates(self, frame, env_tag_list=None):
        """
        프레임에서 ROI 후보를 추출합니다.
        
        Args:
            frame: 카메라에서 캡처한 BGR 이미지
            env_tag_list: 필터링할 클래스 이름 리스트 (None이면 모든 클래스)
            
        Returns:
            list: [{'id': str, 'points': np.array, 'crop': img_bgra}, ...]
        """
        h_orig, w_orig = frame.shape[:2]
        
        # 1. arch1_headless 좌표계(640x480)와 일치시키기 위해 리사이즈
        frame_resized = cv2.resize(frame, self.target_size)
        
        # 2. 세그멘테이션 추론
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

            # 환경 태그 필터링
            if env_tag_list and class_name not in env_tag_list:
                continue

            conf = float(boxes.conf[i].item())
            
            # 다각형 좌표 (640x480 좌표계)
            polygon = masks.xy[i].astype(np.int32)
            
            # 다각형 단순화 (좌표 개수 줄이기)
            epsilon = 0.01 * cv2.arcLength(polygon, True)
            approx_polygon = cv2.approxPolyDP(polygon, epsilon, True).reshape(-1, 2)

            # --- 배경 투명 처리된 이미지 생성 (BGRA) ---
            # 마스크 생성
            mask = np.zeros(self.target_size[::-1], dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)

            # BGRA 변환 및 마스크 적용
            bgra = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = mask

            # 바운딩 박스 기준으로 크롭
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
            # 패딩 추가
            padding = 10
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(640, x2 + padding), min(480, y2 + padding)
            
            crop_img = bgra[y1:y2, x1:x2].copy()

            obj_id = f"{class_name}_{i}_{datetime.now().strftime('%H%M%S')}"
            
            candidates.append({
                "id": obj_id,
                "class_name": class_name,
                "confidence": conf,
                "points": approx_polygon, # 640x480 좌표계의 다각형
                "crop": crop_img,          # 배경 투명 처리된 크롭 이미지
                "bbox": [x1, y1, x2, y2]   # 640x480 좌표계의 박스
            })

        return candidates

# 사용 예시 (테스트용)
if __name__ == "__main__":
    extractor = DynamicROIExtractor()
    # 더미 카메라 입력 예시
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        rois = extractor.extract_candidates(frame)
        for roi in rois:
            print(f"✅ 추출된 구역: {roi['id']}, 좌표 수: {len(roi['points'])}")
            # 이미지 저장 테스트
            cv2.imwrite(f"_outputs/{roi['id']}.png", roi['crop'])
    cap.release()
