import queue
import threading
from pathlib import Path

# Phase 1: 스레드 안전 큐 생성
zone_config_queue = queue.Queue()
vlm_queue = queue.LifoQueue(maxsize=1)

# 글로벌 종료 이벤트
stop_event = threading.Event()

# 환경 변수 및 임계값
DEPTH_SIMILARITY_THRESHOLD = 0.5
YOLO_INPUT_SIZE = 320

# 경로 설정
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
MODEL_PATH = ROOT_DIR / "yolo26n_ncnn_model"
SAVE_DIR = ROOT_DIR / "_outputs" / "vlm_captures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
