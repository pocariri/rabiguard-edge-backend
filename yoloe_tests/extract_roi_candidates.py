import cv2
import json
import os
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLOE
except ImportError:
    print("❌ ultralytics 패키지가 필요합니다. pip install ultralytics 명령어로 설치해주세요.")
    import sys
    sys.exit(1)

def load_environment_tags(env_file):
    """지정된 환경 파일에서 허용된 객체 태그 목록을 불러옵니다."""
    tags = set()
    try:
        with open(env_file, 'r') as f:
            for line in f:
                tag = line.strip().lower()
                if tag:
                    tags.add(tag)
    except FileNotFoundError:
        print(f"❌ 환경 파일을 찾을 수 없습니다: {env_file}")
    return tags

def propose_roi_from_image(image_path, env_name, model_path="yoloe-26n-seg-pf.pt"):
    # 환경별 파일 경로 매핑
    env_files = {
        "1": "retail_and_convenience_store.txt",
        "2": "food_and_beverage.txt",
        "3": "smart_home_and_elderly_care.txt",
        "4": "industrial_and_logistics.txt",
        "5": "education_and_daycare.txt"
    }

    if env_name not in env_files:
        print("❌ 잘못된 환경 선택입니다. 1~5 사이의 숫자를 입력하세요.")
        return

    env_file_path = Path(__file__).parent / "roi_obj_lists" / env_files[env_name]
    valid_tags = load_environment_tags(env_file_path)
    
    if not valid_tags:
        return

    print(f"📦 YOLOE 모델 로드 중: {model_path}")
    try:
        model = YOLOE(model_path)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 이미지 읽기
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ 이미지를 읽을 수 없습니다: {image_path}")
        return

    print(f"🔍 '{env_files[env_name]}' 환경에 맞는 객체를 탐색합니다...")
    
    # 예측 수행
    results = model.predict(frame, verbose=False)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    proposals_dir = Path(__file__).parent / f"proposals_{timestamp}"
    proposals_dir.mkdir(exist_ok=True)
    
    proposals_info = []

    if results and results[0].masks is not None:
        masks = results[0].masks
        boxes = results[0].boxes
        names = model.names

        height, width = frame.shape[:2]
        padding = 20 # 상하좌우 여유 공간 (픽셀)

        vis_img = frame.copy()

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            class_name = names[cls_id].lower()

            # 감지된 객체가 선택한 환경의 허용된 목록에 있는지 확인
            if class_name in valid_tags:
                conf = float(boxes.conf[i].item())
                
                # Bounding Box 좌표 (패딩 추가 및 이미지 경계 초과 방지)
                orig_x1, orig_y1, orig_x2, orig_y2 = map(int, boxes.xyxy[i].cpu().numpy())
                x1 = max(0, orig_x1 - padding)
                y1 = max(0, orig_y1 - padding)
                x2 = min(width, orig_x2 + padding)
                y2 = min(height, orig_y2 + padding)
                
                # Polygon 좌표 추출 및 단순화
                polygon = masks.xy[i]
                polygon = np.array(polygon, dtype=np.int32)
                epsilon = 0.015 * cv2.arcLength(polygon, True)
                approx_polygon = cv2.approxPolyDP(polygon, epsilon, True)
                roi_points = approx_polygon.reshape(-1, 2).tolist()

                # --- 전체 시각화 이미지에 바운딩 박스와 다각형 그리기 ---
                cv2.polylines(vis_img, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                cv2.putText(vis_img, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # --- Segmentation 마스크를 이용해 배경을 투명하게 잘라내기 ---
                # 1. 빈 캔버스(마스크) 생성 및 객체의 외곽선 내부를 흰색(255)으로 채우기
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [polygon], 255)

                # 2. 원본 이미지에 투명도(Alpha) 채널(BGRA) 추가
                bgra_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                
                # 3. 투명도 채널에 마스크를 적용하여 배경을 투명하게 처리
                bgra_frame[:, :, 3] = mask

                # 4. 바운딩 박스 크기만큼 객체 부분만 크롭
                crop_img = bgra_frame[y1:y2, x1:x2].copy()
                
                # 파일 이름 생성 (클래스명_인덱스) - 투명도를 지원하는 PNG로 확장자 변경
                obj_id = f"{class_name.replace(' ', '_')}_{i}"
                img_filename = f"{obj_id}.png"
                img_path = proposals_dir / img_filename
                
                if crop_img.size > 0:
                    cv2.imwrite(str(img_path), crop_img)

                # 메타데이터 수집
                proposal_data = {
                    "id": obj_id,
                    "class_name": class_name,
                    "confidence": round(conf, 3),
                    "image_file": img_filename,
                    "bounding_box": [x1, y1, x2, y2],
                    "polygon_points": roi_points
                }
                proposals_info.append(proposal_data)
                
                print(f"✅ 발견됨: {class_name} (신뢰도: {conf:.2f}) -> {img_filename} 저장 완료")

    else:
        print("⚠️ 객체가 감지되지 않았습니다.")

    # 결과 JSON 및 시각화 이미지 저장
    if proposals_info:
        json_path = proposals_dir / "proposals.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(proposals_info, f, indent=4, ensure_ascii=False)
            
        vis_path = proposals_dir / "visualization.jpg"
        cv2.imwrite(str(vis_path), vis_img)
        
        print(f"\n🎉 총 {len(proposals_info)}개의 ROI 후보 객체를 추출했습니다.")
        print(f"📂 저장 경로: {proposals_dir}")
        print(f"📄 요약 정보: {json_path}")
        print(f"🖼️ 시각화 이미지: {vis_path}")
    else:
        print("\n⚠️ 해당 환경에 적합한 객체를 화면에서 찾지 못했습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ROI candidates from an image based on environment.")
    parser.add_argument("image_path", type=str, help="Path to the input image file")
    parser.add_argument("--env", type=str, required=True, choices=["1", "2", "3", "4", "5"],
                        help="Environment type: 1(Retail), 2(F&B), 3(Home/Care), 4(Industrial), 5(Education)")
    
    args = parser.parse_args()
    propose_roi_from_image(args.image_path, args.env)
