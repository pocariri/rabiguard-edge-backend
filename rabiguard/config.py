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
# rabiguard-edge-backend/rabiguard/config.py
RABIGURAD_EDGE_BACKEND_DIR = Path(__file__).resolve().parent
MODEL_DIR = "./models"

MODEL_PATH = MODEL_DIR / "yolo26n_ncnn_model"
YOLOE_MODEL_PATH = MODEL_DIR"

SAVE_DIR = ROOT_DIR / "_outputs" / "vlm_captures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Firebase key path
FIREBASE_KEY_PATH = ROOT_DIR / "firebase_key.json"