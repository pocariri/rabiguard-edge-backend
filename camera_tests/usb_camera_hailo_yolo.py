import sys
import queue
import threading
from functools import partial
from pathlib import Path
import numpy as np
import json
import cv2

# 1. hailo-apps 모듈을 사용할 수 있도록 시스템 경로에 추가
# 스크립트가 위치한 경로를 기준으로 hailo-apps 폴더를 찾도록 수정
hailo_apps_dir = (Path(__file__).parent / "hailo-apps").resolve()
print(f"DEBUG: Resolving hailo-apps directory to: {hailo_apps_dir}")
print(f"DEBUG: Does hailo-apps exist? {hailo_apps_dir.exists()}")

if str(hailo_apps_dir) not in sys.path:
    sys.path.insert(0, str(hailo_apps_dir))

print(f"DEBUG: sys.path is: {sys.path[:3]}") # 상위 3개 경로만 확인

# hailo-apps의 핵심 컴포넌트들 가져오기
from hailo_apps.python.core.common.hailo_inference import HailoInfer
from hailo_apps.python.core.common.toolbox import InputContext, VisualizationSettings, init_input_source, preprocess, visualize
from hailo_apps.python.standalone_apps.object_detection.object_detection_post_process import inference_result_handler
from hailo_apps.python.core.common.defines import MAX_INPUT_QUEUE_SIZE, MAX_OUTPUT_QUEUE_SIZE

def main():
    print("🚀 Hailo-10H NPU 기반 실시간 객체 인식 초기화 중...")

    # [모델 파일 설정]
    # 주의: YOLO26 모델을 Hailo 컴파일러로 변환한 .hef 파일이 필요합니다.
    hef_path = "./yolo26n.hef"
    
    # hailo-apps에서 제공하는 기본 객체 인식용 설정 파일 사용
    config_path = str(hailo_apps_dir / "hailo_apps/python/standalone_apps/object_detection/config.json")
    
    if not Path(hef_path).exists():
        print(f"❌ 에러: 모델 파일을 찾을 수 없습니다: {hef_path}")
        print("💡 팁: 'yolo26n.hef' 파일이 현재 디렉토리에 존재하는지 확인해주세요.")
        return

    # 설정 파일 로드
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # 2. 카메라 소스 및 컨텍스트 초기화
    # /dev/video0 은 기본 USB 카메라를 의미합니다.
    input_context = InputContext(input_src="/dev/video0", batch_size=1)
    input_context = init_input_source(input_context)

    # 3. 화면 출력/저장 설정
    # HDMI 디스플레이에 출력하기 위해 no_display=False 로 설정합니다.
    viz_settings = VisualizationSettings(
        output_dir="./outputs",
        save_stream_output=False,
        no_display=False # GUI 창 띄우기
    )

    # 4. Hailo NPU 추론기 초기화
    hailo_inference = HailoInfer(hef_path, batch_size=1)
    height, width, _ = hailo_inference.get_input_shape()

    # 5. 비동기 데이터 큐 및 쓰레드 종료 이벤트 생성
    input_queue = queue.Queue(MAX_INPUT_QUEUE_SIZE)
    output_queue = queue.Queue(MAX_OUTPUT_QUEUE_SIZE)
    stop_event = threading.Event()

    # 6. 전처리 쓰레드 (카메라 영상 -> 크기 조절 -> 큐)
    preprocess_thread = threading.Thread(
        target=preprocess,
        args=(input_context, input_queue, width, height, None, stop_event)
    )

    # 7. NPU 추론 쓰레드 (큐에서 영상 가져와 NPU 연산)
    def infer_worker():
        while not stop_event.is_set():
            next_batch = input_queue.get()
            if not next_batch: break
            
            input_batch, preprocessed_batch = next_batch
            
            # 추론 완료 시 호출될 콜백 함수
            def callback(completion_info, bindings_list, in_batch):
                if completion_info.exception:
                    print(f"추론 에러: {completion_info.exception}")
                    return
                for i, bindings in enumerate(bindings_list):
                    result = {name: np.expand_dims(bindings.output(name).get_buffer(), axis=0) 
                              for name in bindings._output_names}
                    output_queue.put((in_batch[i], result))

            job = hailo_inference.run(preprocessed_batch, partial(callback, in_batch=input_batch))
            job.wait(1000) 
            
        hailo_inference.close()
        output_queue.put(None) # 종료 신호

    infer_thread = threading.Thread(target=infer_worker)

    # 8. 파이프라인 시작
    preprocess_thread.start()
    infer_thread.start()

    print("🚀 실시간 카메라 추론을 시작합니다. (Hailo-10H NPU)")
    print("화면이 띄워진 상태에서 'q' 키를 누르거나 터미널에서 Ctrl+C를 누르면 종료됩니다.")
    
    # 9. 후처리 및 시각화 (OpenCV 창 관리)
    # 메인 쓰레드에서 실행되어 모델 예측 결과를 이미지에 그리고 화면에 출력합니다.
    post_process_fn = partial(inference_result_handler, labels=None, config_data=config_data, tracker=None, draw_trail=False)
    
    try:
        visualize(input_context, viz_settings, output_queue, post_process_fn, fps_tracker=None, stop_event=stop_event)
    except KeyboardInterrupt:
        print("\n사용자에 의해 종료되었습니다.")
    finally:
        stop_event.set()
        preprocess_thread.join()
        infer_thread.join()
        print("✅ 프로그램이 안전하게 종료되었습니다.")

if __name__ == "__main__":
    main()
