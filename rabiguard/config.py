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

# Snapshot Recording Settings
SNAPSHOT_FPS = 2  # 1초에 2장 저장
SNAPSHOT_BEFORE_SEC = 10
SNAPSHOT_AFTER_SEC = 10
SNAPSHOT_BUFFER_SIZE = (SNAPSHOT_BEFORE_SEC + SNAPSHOT_AFTER_SEC) * SNAPSHOT_FPS

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

# 현재 파일 위치: rabiguard/config.py
RABIGUARD_DIR = Path(__file__).resolve().parent

# 프로젝트 루트 위치: rafour-app/
ROOT_DIR = RABIGUARD_DIR.parent

MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "yolo26n_ncnn_model"
YOLOE_MODEL_PATH = MODEL_DIR / "yoloe-26n-seg-pf.pt"

SAVE_DIR = ROOT_DIR / "_outputs" / "vlm_captures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Snapshot 저장 경로
SNAPSHOT_DIR = ROOT_DIR / "_outputs" / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# ROI Extractor 관련 경로
TARGET_OBJECTS_PATH = RABIGUARD_DIR / "target_objects.txt"
ZONES_CONFIG_PATH = RABIGUARD_DIR / "zones_config.json"
EXTRACTED_ROIS_DIR = ROOT_DIR / "_outputs" / "extracted_rois"

# Firebase key path
FIREBASE_KEY_PATH = ROOT_DIR / "firebase_key.json"
