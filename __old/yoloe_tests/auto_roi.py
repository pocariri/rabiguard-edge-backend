import cv2
import json
import numpy as np
from pathlib import Path

# ultralytics 패키지에서 YOLOE 임포트
try:
    from ultralytics import YOLOE
except ImportError:
    print("❌ ultralytics 패키지가 필요합니다. pip install ultralytics 명령어로 설치해주세요.")
    import sys
    sys.exit(1)

def auto_roi_live(model_path="yoloe-26l-seg-pf.pt", camera_id=0):
    print(f"📦 YOLOE 모델 로드 중: {model_path}")
    try:
        # Prompt Free(PF) 모델 초기화
        model = YOLOE(model_path)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        print("모델 파일이 있는지 확인해주세요.")
        return

    print(f"🎥 카메라({camera_id}) 스트리밍 시작...")
    print("="*50)
    print("📍 [자동 ROI 설정 모드]")
    print("YOLOE 모델이 화면에서 가장 적합한(눈에 띄는) 객체를 자동으로 찾아냅니다.")
    print("- [s] 키: 현재 감지된 초록색 영역을 ROI로 저장하고 종료하기")
    print("- [q] 키: 저장하지 않고 종료")
    print("="*50)

    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    current_roi_points = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ 카메라 프레임을 읽을 수 없습니다.")
                break
                
            display_img = frame.copy()
            
            # YOLOE Prompt Free 모델 예측 (verbose=False로 콘솔 출력 방지)
            results = model.predict(frame, verbose=False)
            
            # 세그멘테이션 마스크가 감지되었는지 확인
            if results and results[0].masks is not None:
                masks = results[0].masks
                boxes = results[0].boxes
                
                if len(boxes) > 0:
                    # 화면에서 가장 큰 객체(면적 기준)를 주된 관심 영역으로 간주
                    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes.xyxy.cpu().numpy()]
                    largest_idx = np.argmax(areas)
                    
                    # 해당 마스크의 테두리 좌표 추출
                    polygon = masks.xy[largest_idx]
                    polygon = np.array(polygon, dtype=np.int32)
                    
                    # 다각형 형태 단순화 (점의 개수 줄이기 - Douglas-Peucker 알고리즘)
                    epsilon = 0.015 * cv2.arcLength(polygon, True)
                    approx_polygon = cv2.approxPolyDP(polygon, epsilon, True)
                    
                    # [N, 1, 2] -> [N, 2] 로 변환 후 파이썬 리스트로 변경
                    current_roi_points = approx_polygon.reshape(-1, 2).tolist()
                    
                    # 시각화: 테두리 그리기 및 반투명 초록색 채우기
                    if len(current_roi_points) >= 3:
                        cv2.polylines(display_img, [np.array(current_roi_points)], True, (0, 255, 0), 3)
                        
                        overlay = display_img.copy()
                        cv2.fillPoly(overlay, [np.array(current_roi_points)], (0, 255, 0))
                        cv2.addWeighted(overlay, 0.4, display_img, 0.6, 0, display_img)
                        
                        cv2.putText(display_img, "Object Found! Press 's' to save ROI", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                current_roi_points = None
                cv2.putText(display_img, "No salient object detected...", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            cv2.imshow("YOLOE Auto ROI Setup", display_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            # 'q' 키: 종료
            if key == ord('q'):
                print("🛑 저장하지 않고 종료합니다.")
                break
                
            # 's' 키: 저장
            elif key == ord('s'):
                if current_roi_points is not None and len(current_roi_points) >= 3:
                    # rafour-app 최상단 디렉토리에 roi_config.json 저장
                    config_path = Path(__file__).resolve().parent.parent / "roi_config.json"
                    
                    with open(config_path, "w") as f:
                        json.dump({"roi_polygon": current_roi_points}, f)
                        
                    print(f"\n✅ ROI 폴리곤 성공적으로 저장됨! (점 개수: {len(current_roi_points)})")
                    print(f"저장 위치: {config_path}\n")
                    
                    # 저장 성공 시각적 피드백
                    cv2.putText(display_img, "SAVED!", (display_img.shape[1]//2 - 70, display_img.shape[0]//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    cv2.imshow("YOLOE Auto ROI Setup", display_img)
                    cv2.waitKey(1500) # 1.5초간 대기
                    break
                else:
                    print("⚠️ 유효한 객체가 감지되지 않아 저장할 수 없습니다.")
                    
    except KeyboardInterrupt:
        print("\n🛑 강제 종료됨")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    auto_roi_live()
