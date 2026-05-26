import os
import sys
import time
import queue
import threading
from pathlib import Path

import cv2
import numpy as np

# NCNN 및 OpenMP 최적화 환경 변수 설정
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE,v4l2slh265dec:NONE,v4l2slh264dec:NONE,v4l2h265dec:NONE,v4l2h264dec:NONE"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.fonts=false"

# YOLO NCNN
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics 패키지가 필요합니다.")
    sys.exit(1)

# Hailo GStreamer Imports
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

hailo_apps_dir = (Path.home() / "hailo-apps").resolve()
if str(hailo_apps_dir) not in sys.path:
    sys.path.insert(0, str(hailo_apps_dir))

try:
    from hailo_apps.python.pipeline_apps.depth.depth_pipeline import GStreamerDepthApp
    from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
    from hailo_apps.python.core.common.buffer_utils import get_numpy_from_buffer
    import hailo
    from hailo_apps.python.core.common.core import resolve_hef_path
    from hailo_apps.python.core.common.defines import VLM_CHAT_APP, SHARED_VDEVICE_GROUP_ID, HAILO10H_ARCH
    from hailo_platform import VDevice
    from hailo_platform.genai import VLM
    # 최적화된 파이프라인 구성을 위한 추가 임포트
    from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
        INFERENCE_PIPELINE,
        INFERENCE_PIPELINE_WRAPPER,
        USER_CALLBACK_PIPELINE,
    )
except ImportError as e:
    print(f"❌ Hailo 라이브러리 임포트 실패: {e}")
    sys.exit(1)


# -----------------------------------------------------------------------
# GStreamer StructureWrapper Bug Fix (For GStreamer 1.26.2)
# -----------------------------------------------------------------------
def get_caps_from_pad_fixed(pad):
    caps = pad.get_current_caps()
    if not caps: return None, None, None
    structure = caps.get_structure(0)
    if not structure: return None, None, None
    real_structure = getattr(structure, '_StructureWrapper__structure', structure)
    try:
        return real_structure.get_value("format"), real_structure.get_value("width"), real_structure.get_value("height")
    except AttributeError:
        return None, None, None


# -----------------------------------------------------------------------
# Headless Optimized App Class
# -----------------------------------------------------------------------
class HeadlessDepthApp(GStreamerDepthApp):
    """
    Truly headless version of GStreamerDepthApp.
    - Removes hailooverlay and display-related elements to save CPU.
    - Reduces internal buffering (bypass queue) to minimize latency.
    """
    def get_pipeline_string(self):
        source_pipeline = self.get_source_pipeline()
        
        depth_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            name="depth_inference",
        )
        
        # 최적화: bypass_max_size_buffers를 20에서 2로 줄여 지연시간(Lag) 최소화
        depth_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(
            depth_pipeline, bypass_max_size_buffers=2, name="inference_wrapper_depth"
        )
        
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        
        # HEADLESS: 디스플레이 관련 요소를 모두 제거하고 fakesink로 직접 연결
        # hailooverlay와 videoconvert 단계를 생략하여 CPU 리소스를 절약합니다.
        pipeline_str = (
            f"{source_pipeline} ! "
            f"{depth_pipeline_wrapper} ! "
            f"{user_callback_pipeline} ! "
            f"fakesink sync=false"
        )
        print("✅ [PIPELINE] Headless mode initialized (Display disabled, Latency minimized)")
        return pipeline_str


# -----------------------------------------------------------------------
# 전역 설정 및 스레드 이벤트
# -----------------------------------------------------------------------
ROI_POLYGON = np.array([[100, 100], [540, 100], [540, 380], [100, 380]], np.int32)
ENTER_THRESHOLD_SEC = 2.0
DEPTH_SIMILARITY_THRESHOLD = 0.5

# 이미지 저장 경로 설정
SAVE_DIR = Path(__file__).parent.parent / "_outputs" / "vlm_captures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

stop_event = threading.Event()
vlm_queue = queue.LifoQueue(maxsize=1)

# -----------------------------------------------------------------------
# 유틸리티 함수
# -----------------------------------------------------------------------
def get_roi_depth(depth_map, x1, y1, x2, y2):
    if depth_map is None: return 0.0
    h, w = depth_map.shape
    scale_y, scale_x = h / 480.0, w / 640.0
    tx1, ty1 = int(x1 * scale_x), int(y1 * scale_y)
    tx2, ty2 = int(x2 * scale_x), int(y2 * scale_y)
    ty1, ty2 = max(0, ty1), min(h, ty2)
    tx1, tx2 = max(0, tx1), min(w, tx2)
    if tx1 >= tx2 or ty1 >= ty2: return 0.0
    roi_depth_values = depth_map[ty1:ty2, tx1:tx2].flatten()
    roi_depth_values = roi_depth_values[roi_depth_values > 0.1]
    if len(roi_depth_values) == 0: return 0.0
    return float(np.mean(roi_depth_values))

# -----------------------------------------------------------------------
# VLM Worker Thread (NPU)
# -----------------------------------------------------------------------
def vlm_worker_thread():
    print("🤖 [VLM Worker] 초기화 시작...")
    hef_path = resolve_hef_path(None, app_name=VLM_CHAT_APP, arch=HAILO10H_ARCH)
    if not hef_path: return
    vdevice = None
    vlm = None
    try:
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        vdevice = VDevice(params)
        vlm = VLM(vdevice, str(hef_path))
        print("✅ [VLM Worker] VLM 초기화 완료!")
        while not stop_event.is_set():
            try:
                item = vlm_queue.get(timeout=0.5)
            except queue.Empty: continue
            if item is None: break
            context_img, track_id, p_depth, r_depth = item
            
            # [SAVE] VLM 분석 대상 이미지 저장 (박스/ROI 포함)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = SAVE_DIR / f"vlm_event_ID{track_id}_{timestamp}.jpg"
            cv2.imwrite(str(save_path), context_img)
            print(f"📸 [VLM] 이벤트 이미지 저장됨: {save_path}")

            vlm_img = cv2.resize(context_img, (336, 336))
            vlm_img = cv2.cvtColor(vlm_img, cv2.COLOR_BGR2RGB)
            prompt = [
                {"role": "system", "content": [{"type": "text", "text": "You are an AI assistant that monitors CCTV and describes the situation."}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"A person (depth: {p_depth:.2f}m) is detected inside the restricted zone (depth: {r_depth:.2f}m). Please summarize what the person is doing in one short sentence."}]}
            ]
            print(f"\n🧠 [VLM] ID {track_id} 객체 분석 중...")
            try:
                response = vlm.generate_all(prompt=prompt, frames=[vlm_img], temperature=0.1, max_generated_tokens=30)
                clean_text = response.split(". [{'type'")[0].split("<|im_end|>")[0].strip()
                print("="*60 + f"\n🚨 [VLM 알림] ID {track_id}: {clean_text}\n" + "="*60)
            except Exception as e: print(f"⚠️ [VLM Error] {e}")
            finally: vlm_queue.task_done()
    except Exception as e: print(f"❌ [VLM Worker] {e}")
    finally:
        if vlm: vlm.release()
        if vdevice: vdevice.release()

# -----------------------------------------------------------------------
# Callback Class & YOLO Worker
# -----------------------------------------------------------------------
class HeadlessAppCallback(app_callback_class):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.total_frames = 0
        self.status_start_time = time.time()
        self.status_frame_count = 0
        
        # Caps 정보 캐싱
        self.caps_info = None
        
        self.tracker_state = {}
        self.yolo_queue = queue.Queue(maxsize=1)
        self.yolo_ready = True  # 동기화 플래그

        # [SNAPSHOT] 정기적 스냅샷 설정 (10초 간격)
        self.last_snapshot_time = 0
        self.snapshot_interval = 10.0
        self.snapshot_dir = SAVE_DIR.parent / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.yolo_thread = threading.Thread(target=self.yolo_worker, daemon=True)
        self.yolo_thread.start()
        
    def yolo_worker(self):
        print("✅ [YOLO Worker] 시작!")
        last_yolo_time = time.time()
        yolo_count = 0
        
        while not stop_event.is_set():
            self.yolo_ready = True
            try:
                data = self.yolo_queue.get(timeout=0.5)
            except queue.Empty: continue
            if data is None: break
            
            frame_raw, depth_raw, fmt = data
            try:
                h_orig, w_orig = frame_raw.shape[:2]
                color_conv = cv2.COLOR_RGB2BGR if fmt == "RGB" else cv2.COLOR_RGBA2BGR

                # 전처리: 해상도를 320x320으로 축소
                frame_small_rgb = cv2.resize(frame_raw, (320, 320), interpolation=cv2.INTER_LINEAR)
                frame_small_bgr = cv2.cvtColor(frame_small_rgb, color_conv)
                
                depth_map = None
                if depth_raw is not None:
                    depth_map = np.array(depth_raw, dtype=np.float32).reshape((256, 320))
                
                # NCNN 추론
                results = self.model.track(frame_small_bgr, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)
                
                yolo_count += 1
                now = time.time()
                
                # [SAVE] 정기적 시스템 스냅샷 저장 (박스 및 ROI 포함)
                if now - self.last_snapshot_time >= self.snapshot_interval:
                    self.last_snapshot_time = now
                    snapshot_draw = cv2.cvtColor(frame_raw, color_conv)
                    
                    # ROI 구역 그리기
                    cv2.polylines(snapshot_draw, [ROI_POLYGON], True, (255, 0, 0), 2)
                    
                    # 모든 탐지된 객체 박스 그리기
                    if results[0].boxes is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        track_ids = results[0].boxes.id.int().cpu().numpy() if results[0].boxes.id is not None else [None] * len(boxes)
                        for box, tid in zip(boxes, track_ids):
                            x1 = int(box[0] * (w_orig / 320))
                            y1 = int(box[1] * (h_orig / 320))
                            x2 = int(box[2] * (w_orig / 320))
                            y2 = int(box[3] * (h_orig / 320))
                            cv2.rectangle(snapshot_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            if tid is not None:
                                cv2.putText(snapshot_draw, f"ID:{tid}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    snapshot_path = self.snapshot_dir / f"snapshot_{timestamp}.jpg"
                    cv2.imwrite(str(snapshot_path), snapshot_draw)
                    print(f"📷 [SYSTEM] 분석 스냅샷 저장됨 (박스 포함): {snapshot_path}")

                if now - last_yolo_time >= 2.0:
                    print(f"📊 [YOLO SPEED] {yolo_count / (now - last_yolo_time):.1f} FPS")
                    yolo_count, last_yolo_time = 0, now

                current_ids = set()
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes_small = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()

                    for box, track_id in zip(boxes_small, track_ids):
                        current_ids.add(track_id)
                        if track_id not in self.tracker_state:
                            self.tracker_state[track_id] = {"enter_time": None, "notified": False, "last_log_time": 0}
                        
                        # 320x320 좌표를 원래 해상도로 복원
                        x1 = int(box[0] * (w_orig / 320))
                        y1 = int(box[1] * (h_orig / 320))
                        x2 = int(box[2] * (w_orig / 320))
                        y2 = int(box[3] * (h_orig / 320))
                        
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        state = self.tracker_state[track_id]
                        
                        if cv2.pointPolygonTest(ROI_POLYGON, (cx, cy), False) >= 0:
                            if state["enter_time"] is None:
                                state["enter_time"] = time.time()
                                state["last_log_time"] = 0
                                print(f"⚠️ [ZONE] ID {track_id} 구역 내부 포착 (Timer 시작)")
                            
                            wait_time = time.time() - state["enter_time"]
                            if not state["notified"]:
                                if wait_time >= ENTER_THRESHOLD_SEC:
                                    p_depth = get_roi_depth(depth_map, x1, y1, x2, y2)
                                    rx, ry, rw, rh = cv2.boundingRect(ROI_POLYGON)
                                    z_depth = get_roi_depth(depth_map, rx, ry, rx+rw, ry+rh)
                                    depth_diff = abs(p_depth - z_depth)
                                    
                                    if depth_diff <= DEPTH_SIMILARITY_THRESHOLD:
                                        print(f"🚨 [DETECTION] ID {track_id} 최종 진입 확정! (거리차: {depth_diff:.2f}m)")
                                        state["notified"] = True
                                        # VLM에 보낼 큰 이미지는 필요한 시점에만 BGR 변환 수행
                                        frame_bgr = cv2.cvtColor(frame_raw, color_conv)
                                        ctx = frame_bgr.copy()
                                        cv2.rectangle(ctx, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                        cv2.polylines(ctx, [ROI_POLYGON], True, (255, 0, 0), 2)
                                        try: vlm_queue.put_nowait((ctx, track_id, p_depth, z_depth))
                                        except queue.Full: pass
                                    else:
                                        # 물리적 필터링 대기 로그 (1초 간격)
                                        if time.time() - state.get("last_log_time", 0) >= 1.0:
                                            print(f"⏳ [PHYSICAL] ID {track_id} 거리 불일치 (사람:{p_depth:.1f}m, 구역:{z_depth:.1f}m, 차이:{depth_diff:.2f}m)")
                                            state["last_log_time"] = time.time()
                                else:
                                    # 시간적 필터링 대기 로그 (1초 간격)
                                    if time.time() - state.get("last_log_time", 0) >= 1.0:
                                        print(f"⏳ [TEMPORAL] ID {track_id} 진입 대기 중... ({wait_time:.1f}/{ENTER_THRESHOLD_SEC}s)")
                                        state["last_log_time"] = time.time()
                        else:
                            state["enter_time"], state["notified"] = None, False

                for d_id in list(self.tracker_state.keys() - current_ids):
                    del self.tracker_state[d_id]
            except Exception as e:
                print(f"❌ [YOLO Error] {e}")
            finally:
                self.yolo_queue.task_done()

def app_callback(element, buffer, user_data):
    try:
        if buffer is None: return
        curr_time = time.time()
        
        # 최적화: YOLO 워커가 작업을 마쳐 프레임을 받을 준비가 된 경우에만 무거운 복사 연산을 수행
        if not user_data.yolo_ready:
            user_data.total_frames += 1
            user_data.status_frame_count += 1
            if curr_time - user_data.status_start_time >= 1.0:
                print(f"⏱️ [PIPELINE] FPS: {user_data.status_frame_count / (curr_time - user_data.status_start_time):.1f}")
                user_data.status_start_time, user_data.status_frame_count = curr_time, 0
            return

        user_data.yolo_ready = False

        if user_data.caps_info is None:
            user_data.caps_info = get_caps_from_pad_fixed(element.get_static_pad("sink"))
        fmt, w, h = user_data.caps_info

        frame_raw = get_numpy_from_buffer(buffer, fmt, w, h)
        if frame_raw is None: return

        roi = hailo.get_roi_from_buffer(buffer)
        depth_objs = roi.get_objects_typed(hailo.HAILO_DEPTH_MASK)
        depth_raw = depth_objs[0].get_data() if len(depth_objs) > 0 else None

        try:
            if user_data.yolo_queue.full():
                user_data.yolo_queue.get_nowait()
            user_data.yolo_queue.put_nowait((frame_raw, depth_raw, fmt))
        except queue.Empty: pass
        except queue.Full: pass
        
        user_data.total_frames += 1
        user_data.status_frame_count += 1
        if curr_time - user_data.status_start_time >= 1.0:
            print(f"⏱️ [PIPELINE] FPS: {user_data.status_frame_count / (curr_time - user_data.status_start_time):.1f}")
            user_data.status_start_time, user_data.status_frame_count = curr_time, 0
            
    except Exception as e:
        print(f"❌ [Callback Error] {e}")


def main():
    print("🚀 [HEADLESS PIPELINE OPTIMIZED] 시작")
    
    # 인자 설정
    if "--input" not in sys.argv: sys.argv.extend(["--input", "usb"])
    if "--width" not in sys.argv: sys.argv.extend(["--width", "640"])
    if "--height" not in sys.argv: sys.argv.extend(["--height", "480"])
    
    # 모델 경로
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolo26n_ncnn_model")
    model = YOLO(model_path, task="detect")
    
    # VLM 워커 시작
    vlm_thread = threading.Thread(target=vlm_worker_thread, daemon=True)
    vlm_thread.start()
    
    # 앱 초기화 및 실행
    user_data = HeadlessAppCallback(model)
    app = HeadlessDepthApp(app_callback, user_data)
    
    try: app.run()
    except KeyboardInterrupt: print("\n🛑 종료 중...")
    finally:
        stop_event.set()
        vlm_queue.put(None)
        user_data.yolo_thread.join(timeout=2.0)
        print("✅ 종료 완료")

if __name__ == "__main__":
    main()
