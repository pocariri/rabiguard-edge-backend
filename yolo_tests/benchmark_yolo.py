#  YOLO 객체 감지(사람 인식 + ByteTrack 추적) 성능만을 측정하여, ONNX
#   백엔드와 NCNN 백엔드의 초당 프레임 수(FPS)를 동일한 조건에서 비교할 수 있는 벤치마크 스크립트(benchmark_yolo.py)를
#   작성했습니다.

#   주요 기능 및 특징
#    1. 동일 환경 통제: 인자로 지정한 동일한 비디오 소스(웹캠 또는 녹화된 .mp4 영상 등)와 동일한 측정 시간(duration)을
#       기준으로 두 모델을 순차적으로 테스트합니다. (테스트 시간보다 동영상이 짧을 경우 자동으로 되감아 무한 반복
#       처리합니다)
#    2. 화면 렌더링 분리: GUI 렌더링 자체가 성능 측정의 병목이 될 수 있으므로 기본적으로는 백그라운드 추론만 수행하며,
#       화면 렌더링을 보고 싶다면 --show 옵션을 붙여 켤 수 있습니다.
#    3. 자동 모델 변환: 만약 yolov8n.onnx 파일이 존재하지 않는다면, yolov8n.pt를 다운로드 받아 ONNX 포맷으로 즉석에서
#       export 한 뒤 벤치마크를 진행하도록 구성했습니다. (만약 이 마저도 실패할 경우 .pt 모델로 fallback하여 NCNN과
#       비교합니다)

#   실행 방법

#   기본적으로 웹캠(0번 인덱스)을 대상으로 각 모델당 15초씩 벤치마크를 수행하려면 아래 명령어를 입력하세요:
#    1 python3 benchmark_yolo.py

#   특정 동영상 파일(test_video.mp4)을 대상으로, 30초 동안, 테스트 화면을 보면서 벤치마크하려면:

#    1 python3 benchmark_yolo.py --video test_video.mp4 --duration 30 --show

#   테스트가 완료되면 터미널 화면에 각 모델의 처리 프레임 수와 평균 FPS, 그리고 어느 쪽이 몇 퍼센트(%) 더 빠른지에 대한
#   요약 리포트가 출력됩니다.

import sys
import time
import argparse
from pathlib import Path
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics 패키지가 필요합니다. 'pip install ultralytics ncnn onnxruntime' 명령어로 설치해주세요.")
    sys.exit(1)

def run_benchmark(model_path, video_source, duration=15, show=False):
    """
    주어진 모델과 비디오 소스로 벤치마크를 실행합니다.
    """
    print(f"\n[{model_path}] 🚀 모델 로딩 중...")
    try:
        model = YOLO(model_path, task="detect")
    except Exception as e:
        print(f"❌ 모델 로드 실패 ({model_path}): {e}")
        return None
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"❌ 동영상/카메라 소스를 열 수 없습니다: {video_source}")
        return None

    print(f"[{model_path}] ⏱️ {duration}초 동안 성능 측정을 진행합니다...")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # 동영상이 끝나면 처음으로 되감기 (무한 반복)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        
        current_time = time.time()
        elapsed = current_time - start_time
        if elapsed > duration:
            break
            
        # 객체 감지 및 ByteTrack 추적 (사람(0)만 대상, 화면 출력 제외)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)
        frame_count += 1
        
        # 화면 출력이 활성화된 경우에만 렌더링 (성능 저하의 원인이 될 수 있음)
        if show:
            res_frame = results[0].plot()
            cv2.imshow("Benchmark", res_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    end_time = time.time()
    actual_duration = end_time - start_time
    cap.release()
    if show:
        cv2.destroyAllWindows()
        
    fps = frame_count / actual_duration
    print("-" * 45)
    print(f"✅ 결과 요약: {model_path}")
    print(f"처리한 프레임: {frame_count} frames")
    print(f"실제 소요 시간: {actual_duration:.2f} sec")
    print(f"평균 FPS: {fps:.2f} fps")
    print("-" * 45)
    
    return fps


def main():
    parser = argparse.ArgumentParser(description="ONNX vs NCNN YOLO 성능 비교 벤치마크 (VLM 제외)")
    parser.add_argument("--video", type=str, default="0", help="테스트할 비디오 파일 경로 (또는 웹캠 인덱스 0)")
    parser.add_argument("--duration", type=int, default=15, help="테스트 진행 시간 (초)")
    parser.add_argument("--onnx_model", type=str, default="yolov8n.onnx", help="ONNX 모델 경로 (기본: yolov8n.onnx)")
    parser.add_argument("--ncnn_model", type=str, default="yolo26n_ncnn_model", help="NCNN 모델 폴더 경로")
    parser.add_argument("--show", action="store_true", help="추론 화면 표시 (체크 시 렌더링 부하로 FPS 하락 가능)")
    
    args = parser.parse_args()
    
    video_src = int(args.video) if args.video.isdigit() else args.video
    
    print("=" * 55)
    print(" 🚀 YOLO 백엔드 성능 비교 (ONNX vs NCNN)")
    print("=" * 55)
    
    # 1. ONNX 모델 존재 여부 확인 및 자동 변환
    onnx_path = Path(args.onnx_model)
    if not onnx_path.exists():
        print(f"⚠️ 지정된 ONNX 모델({args.onnx_model})이 없습니다. PyTorch 모델에서 변환을 시도합니다.")
        pt_path = args.onnx_model.replace('.onnx', '.pt')
        try:
            pt_model = YOLO(pt_path)
            print(f"[{pt_path}] ONNX 형식으로 변환 중...")
            pt_model.export(format="onnx")
        except Exception as e:
            print(f"⚠️ ONNX 변환 실패: {e}")
            print(f"👉 대신 {pt_path} (PyTorch)를 사용하여 벤치마크를 진행합니다.")
            args.onnx_model = pt_path # 실패 시 PyTorch 모델로 fallback

    # 2. ONNX (또는 PyTorch) 벤치마크 실행
    fps_onnx = run_benchmark(args.onnx_model, video_src, args.duration, args.show)
    
    # 3. NCNN 벤치마크 실행
    fps_ncnn = run_benchmark(args.ncnn_model, video_src, args.duration, args.show)
    
    # 4. 결과 종합
    if fps_onnx and fps_ncnn:
        print("\n🏆 최종 성능 비교 결과 🏆")
        print(f"ONNX/PT 모델 ({args.onnx_model}): {fps_onnx:.2f} FPS")
        print(f"NCNN 모델 ({args.ncnn_model}): {fps_ncnn:.2f} FPS")
        
        diff = fps_ncnn - fps_onnx
        percent = (diff / fps_onnx) * 100
        if diff > 0:
            print(f"🔥 NCNN 백엔드가 {diff:.2f} FPS ({percent:.1f}%) 더 빠릅니다.")
        else:
            print(f"🔥 ONNX 백엔드가 {-diff:.2f} FPS ({-percent:.1f}%) 더 빠릅니다.")

if __name__ == "__main__":
    main()
