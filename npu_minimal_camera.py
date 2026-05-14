import sys
import queue
import threading
from functools import partial
from pathlib import Path
import numpy as np
import json

# 1. hailo-apps 모듈을 사용할 수 있도록 시스템 경로에 추가
hailo_apps_dir = Path("./hailo-apps").resolve()
if str(hailo_apps_dir) not in sys.path:
    sys.path.insert(0, str(hailo_apps_dir))

# hailo-apps의 핵심 컴포넌트들 가져오기
from hailo_apps.python.core.common.hailo_inference import HailoInfer
from hailo_apps.python.core.common.toolbox import InputContext, VisualizationSettings, init_input_source, preprocess, visualize
from hailo_apps.python.standalone_apps.object_detection.object_detection_post_process import inference_result_handler
from hailo_apps.python.core.common.defines import MAX_INPUT_QUEUE_SIZE, MAX_OUTPUT_QUEUE_SIZE

def main():
    print("🚀 Hailo NPU 기반 최소형 객체 검출 파이프라인 초기화 중...")

    # [필수 설정 값] 환경에 맞게 변경해야 할 부분
    # 주의: hailo-apps 환경 셋업(setup_env.sh 등) 및 HEF 모델 다운로드가 선행되어야 합니다.
    hef_path = str(hailo_apps_dir / "local_resources" / "yolov8m.hef") 
    config_path = str(hailo_apps_dir / "hailo_apps/python/standalone_apps/object_detection/config.json")
    
    if not Path(hef_path).exists():
        print(f"❌ HEF 모델 파일을 찾을 수 없습니다: {hef_path}")
        print("💡 팁: 'hailo-apps/local_resources/' 폴더에 .hef 파일이 있는지 확인하세요.")
        return

    # 설정 파일 로드
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # 2. 카메라 소스 및 컨텍스트 초기화 (라즈베리파이 카메라의 경우 /dev/video0 일 수 있습니다)
    # input_src="/dev/video0" 으로 하면 USB/기본 카메라가 열립니다.
    input_context = InputContext(input_src="/dev/video0", batch_size=1)
    input_context = init_input_source(input_context)

    # 3. 화면 출력/저장 설정
    # 화면 출력이 불가능한 환경(SSH 등)이라면 no_display=True 로 설정해야 에러가 나지 않습니다.
    viz_settings = VisualizationSettings(
        output_dir="./outputs",
        save_stream_output=False, # True로 하면 동영상 파일로 저장됩니다.
        no_display=False # 창을 띄워서 확인
    )

    # 4. Hailo NPU 추론기 초기화
    hailo_inference = HailoInfer(hef_path, batch_size=1)
    height, width, _ = hailo_inference.get_input_shape()

    # 5. 비동기 큐(Queue) 및 이벤트 생성
    input_queue = queue.Queue(MAX_INPUT_QUEUE_SIZE)
    output_queue = queue.Queue(MAX_OUTPUT_QUEUE_SIZE)
    stop_event = threading.Event()

    # 6. 쓰레드 설정: 전처리 (카메라 -> 크기 조절 -> 큐)
    preprocess_thread = threading.Thread(
        target=preprocess,
        args=(input_context, input_queue, width, height, None, stop_event)
    )

    # 7. 쓰레드 설정: 추론 (NPU 연산)
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
            job.wait(1000) # 단순화를 위해 동기적으로 대기 (실제론 비동기로 큐에 쌓는 것이 성능에 좋음)
            
        hailo_inference.close()
        output_queue.put(None) # 종료 신호

    infer_thread = threading.Thread(target=infer_worker)

    # 8. 파이프라인 시작!
    preprocess_thread.start()
    infer_thread.start()

    print("✅ 파이프라인 실행 중! (종료하려면 'q'를 누르시거나 터미널에서 Ctrl+C)")
    
    # 9. 후처리 및 시각화 (메인 쓰레드에서 실행 - OpenCV 창을 띄우기 위함)
    # inference_result_handler가 텐서에서 박스를 뽑아내고 이미지를 그려줍니다.
    post_process_fn = partial(inference_result_handler, labels=None, config_data=config_data, tracker=None, draw_trail=False)
    
    try:
        visualize(input_context, viz_settings, output_queue, post_process_fn, fps_tracker=None, stop_event=stop_event)
    except KeyboardInterrupt:
        print("\n사용자에 의해 종료됩니다.")
    finally:
        stop_event.set()
        preprocess_thread.join()
        infer_thread.join()
        print("종료 완료.")

if __name__ == "__main__":
    main()
