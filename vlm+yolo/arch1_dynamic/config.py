import queue
import threading
from pathlib import Path

# ------------------------------------------------------------
# Thread-safe queues
# ------------------------------------------------------------

zone_config_queue = queue.Queue()
vlm_queue = queue.LifoQueue(maxsize=1)

# Global stop event
stop_event = threading.Event()

# ------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------

DEPTH_SIMILARITY_THRESHOLD = 0.5
YOLO_INPUT_SIZE = 320

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

# 현재 파일 위치:
# rabiguard-edge-backend/vlm+yolo/arch1_dynamic/config.py
ARCH1_DYNAMIC_DIR = Path(__file__).resolve().parent
VLM_YOLO_DIR = ARCH1_DYNAMIC_DIR.parent
ROOT_DIR = VLM_YOLO_DIR.parent

MODEL_PATH = ROOT_DIR / "yolo26n_ncnn_model"

SAVE_DIR = ROOT_DIR / "_outputs" / "vlm_captures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Firebase key path
FIREBASE_KEY_PATH = ROOT_DIR / "firebase_key.json"