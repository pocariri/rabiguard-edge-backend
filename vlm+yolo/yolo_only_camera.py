import os
import sys
import time
import queue
import threading
from pathlib import Path

import cv2
import numpy as np

# NCNN 최적화 환경 변수
os.environ["OMP_NUM_THREADS"] = "4" # YOLO 단독 구동이므로 코어 전체 할당

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ 'pip install ultralytics ncnn' 패키지가 필요합니다.")
    sys.exit(1)

# -----------------------------------------------------------------------
# GStreamer Pipeline 설정 (Raspberry Pi 5 + USB Camera용)
# -----------------------------------------------------------------------
def get_gst_pipeline(width=640, height=480, fps=30):
    return (
        f"v4l2src device=/dev/video0 ! "
        f"video/x-raw, width={width}, height={height}, framerate={fps}/1 ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=True"
    )

# -----------------------------------------------------------------------
# YOLO 감지 스레드
# -----------------------------------------------------------------------
def yolo_worker(frame_queue, model):
    print("✅ YOLO 감지 스레드 시작!")
    last_time = time.time()
    frame_count = 0

    while True:
        try:
            # 가장 최신 프레임만 가져오기 (LIFO 처럼 동작하도록 큐 비우기)
            frame = None
            while not frame_queue.empty():
                frame = frame_queue.get_nowait()
            
            if frame is None:
                time.sleep(0.01)
                continue

            # YOLO 추론 (사람 클래스 0번만 감지)
            results = model.track(frame, persist=True, classes=[0], verbose=False, imgsz=320)
            
            frame_count += 1
            now = time.time()
            
            # 감지 결과 분석 및 출력
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                count = len(results[0].boxes)
                ids = results[0].boxes.id.int().cpu().numpy().tolist() if results[0].boxes.id is not None else ["Unknown"]
                print(f"👤 [DETECTED] 현재 {count}명의 사람이 감지됨 (ID: {ids})")
            else:
                # 아무도 없을 때는 주기적으로 '정상' 상태 출력 (선택 사항)
                if frame_count % 30 == 0:
                    print("➖ 감지된 사람 없음")

            # FPS 계산
            if now - last_time >= 2.0:
                print(f"📊 [Inference Speed] {frame_count / (now - last_time):.1f} FPS")
                frame_count, last_time = 0, now

        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")
            continue

def main():
    print("🚀 [YOLO26n Only Monitoring] 시작")
    
    # 1. 모델 경로 설정
    current_dir = Path(__file__).parent.parent.resolve()
    model_path = current_dir / "yolo26n_ncnn_model"
    
    if not model_path.exists():
        print(f"❌ 모델을 찾을 수 없습니다: {model_path}")
        return

    # 2. 모델 로드
    print(f"📦 모델 로딩 중: {model_path}")
    model = YOLO(str(model_path), task="detect")
    
    # 3. 카메라 및 큐 설정
    cap = cv2.VideoCapture(get_gst_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다. /dev/video0 연결을 확인하세요.")
        return

    frame_queue = queue.Queue(maxsize=1)
    
    # 4. 워커 스레드 실행
    worker = threading.Thread(target=yolo_worker, args=(frame_queue, model), daemon=True)
    worker.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 최신 프레임을 워커에게 전달
            try:
                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put_nowait(frame)
            except:
                pass
                
    except KeyboardInterrupt:
        print("\n🛑 중단 요청됨")
    finally:
        cap.release()
        print("✅ 종료 완료")

if __name__ == "__main__":
    main()
