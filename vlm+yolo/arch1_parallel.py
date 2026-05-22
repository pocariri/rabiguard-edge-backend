import os
import sys
import time
import queue
import threading
from pathlib import Path

import cv2
import numpy as np

# GStreamer 하드웨어 디코더 플러그인 충돌 방지 및 Qt 로그 억제
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE,v4l2slh265dec:NONE,v4l2slh264dec:NONE,v4l2h265dec:NONE,v4l2h264dec:NONE"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.fonts=false"

# YOLO NCNN
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics 패키지가 필요합니다. 'pip install ultralytics ncnn' 명령어로 설치해주세요.")
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
    from hailo_apps.python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
    import hailo
    
    from hailo_apps.python.core.common.core import resolve_hef_path
    from hailo_apps.python.core.common.defines import VLM_CHAT_APP, SHARED_VDEVICE_GROUP_ID, HAILO10H_ARCH
    from hailo_platform import VDevice
    from hailo_platform.genai import VLM
except ImportError as e:
    print(f"❌ Hailo 라이브러리 임포트 실패: {e}")
    sys.exit(1)


# -----------------------------------------------------------------------
# 전역 설정
# -----------------------------------------------------------------------
ROI_POLYGON = np.array([[100, 100], [540, 100], [540, 380], [100, 380]], np.int32)
ENTER_THRESHOLD_SEC = 3.0
DEPTH_SIMILARITY_THRESHOLD = 0.5 # meter

vlm_queue = queue.LifoQueue(maxsize=1)
stop_event = threading.Event()
tracker_state = {}

# -----------------------------------------------------------------------
# VLM Worker Thread (NPU)
# -----------------------------------------------------------------------
def vlm_worker_thread():
    print("🤖 [VLM Worker] 초기화 시작...")
    hef_path = resolve_hef_path(None, app_name=VLM_CHAT_APP, arch=HAILO10H_ARCH)
    if not hef_path:
        print("❌ [VLM Worker] VLM 모델을 찾을 수 없습니다.")
        return
        
    vdevice = None
    vlm = None
    try:
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        vdevice = VDevice(params)
        vlm = VLM(vdevice, str(hef_path))
        print("🤖 [VLM Worker] VLM 초기화 완료 대기 중...")
        
        while not stop_event.is_set():
            try:
                item = vlm_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            context_img, track_id, p_depth, r_depth = item
            vlm_img = cv2.resize(context_img, (336, 336))
            if len(vlm_img.shape) == 3 and vlm_img.shape[2] == 3:
                vlm_img = cv2.cvtColor(vlm_img, cv2.COLOR_BGR2RGB)

            prompt = [
                {"role": "system", "content": [{"type": "text", "text": "You are an AI assistant that monitors CCTV and describes the situation."}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"A person (depth: {p_depth:.2f}m) is detected inside the zone (depth: {r_depth:.2f}m). Summarize this situation in less than 15 words."}]}
            ]
            
            print(f"\n🔍 [VLM Worker] 객체 ID {track_id} NPU 추론 시작...")
            try:
                response = vlm.generate_all(prompt=prompt, frames=[vlm_img], temperature=0.1, seed=42, max_generated_tokens=50)
                clean_text = response.split(". [{'type'")[0].split("<|im_end|>")[0].strip()
                print("\n" + "="*50)
                print("🚨 [VLM 상황 인식 알림] 🚨")
                print(f"내용: {clean_text}")
                print("="*50 + "\n")
            except Exception as e:
                print(f"⚠️ [VLM Worker] 추론 중 에러 발생: {e}")
            finally:
                vlm_queue.task_done()
    except Exception as e:
        print(f"❌ [VLM Worker] 에러: {e}")
    finally:
        if vlm: vlm.release()
        if vdevice: vdevice.release()


# -----------------------------------------------------------------------
# Callback & Main
# -----------------------------------------------------------------------
class ParallelAppCallback(app_callback_class):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.last_proc_time = 0
        self.fps_interval = 1.0 / 15.0  # Depth & Visualization은 15FPS로 제한

def get_roi_depth(depth_map, x1, y1, x2, y2):
    # depth_map은 256x320 크기일 수 있음
    h, w = depth_map.shape
    
    # 원본 좌표(640x480 기준 가정)를 depth_map 크기에 맞게 스케일링
    # main에서 640x480으로 강제하므로 비율 계산
    scale_y, scale_x = h / 480.0, w / 640.0
    
    tx1, ty1 = int(x1 * scale_x), int(y1 * scale_y)
    tx2, ty2 = int(x2 * scale_x), int(y2 * scale_y)

    ty1, ty2 = max(0, ty1), min(h, ty2)
    tx1, tx2 = max(0, tx1), min(w, tx2)
    
    if tx1 >= tx2 or ty1 >= ty2:
        return 0.0
        
    roi_depth_values = depth_map[ty1:ty2, tx1:tx2].flatten()
    roi_depth_values = roi_depth_values[roi_depth_values > 0.1]
    
    if len(roi_depth_values) == 0:
        return 0.0
        
    return float(np.median(roi_depth_values))

def app_callback(element, buffer, user_data):
    try:
        if buffer is None:
            return

        # 프레임 추출
        format, width, height = get_caps_from_pad(element.get_static_pad("sink"))
        frame_raw = get_numpy_from_buffer(buffer, format, width, height)
        if frame_raw is None: return
        
        # 1. 프레임 스킵 및 연산 제한
        curr_time = time.time()
        if curr_time - user_data.last_proc_time < user_data.fps_interval:
            return
        user_data.last_proc_time = curr_time

        frame = frame_raw.copy()
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        # 2. 깊이 정보 추출 (SCDepthV3: 320x256)
        roi = hailo.get_roi_from_buffer(buffer)
        depth_objs = roi.get_objects_typed(hailo.HAILO_DEPTH_MASK)
        
        depth_map = None
        if len(depth_objs) > 0:
            depth_data = depth_objs[0].get_data()
            tensor_w, tensor_h = 320, 256
            if len(depth_data) == tensor_w * tensor_h:
                depth_map = np.array(depth_data).reshape((tensor_h, tensor_w))
                
                # Depth 시각화 최적화
                depth_vis = np.clip(depth_map / 5.0 * 255, 0, 255).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                depth_colormap_resized = cv2.resize(depth_colormap, (width, height), interpolation=cv2.INTER_LINEAR)
                frame = cv2.addWeighted(frame, 0.4, depth_colormap_resized, 0.6, 0)
        
        # 3. YOLO 추론 (CPU)
        # NCNN YOLO는 BGR 이미지를 선호하므로 변환
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = user_data.model.track(frame_bgr, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)
        current_ids = set()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                current_ids.add(track_id)
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 구역 내 진입 여부
                inside_roi = cv2.pointPolygonTest(ROI_POLYGON, (cx, cy), False) >= 0
                if inside_roi:
                    if track_id not in tracker_state:
                        tracker_state[track_id] = {"enter_time": time.time(), "notified": False}
                    
                    state = tracker_state[track_id]
                    elapsed = time.time() - state["enter_time"]
                    
                    if elapsed >= ENTER_THRESHOLD_SEC and not state["notified"] and depth_map is not None:
                        person_depth = get_roi_depth(depth_map, x1, y1, x2, y2)
                        rx, ry, rw, rh = cv2.boundingRect(ROI_POLYGON)
                        zone_depth = get_roi_depth(depth_map, rx, ry, rx+rw, ry+rh)
                        
                        diff = abs(person_depth - zone_depth)
                        if diff <= DEPTH_SIMILARITY_THRESHOLD:
                            print(f"🚨 [Event] ID {track_id} Zone Entry Validated (Depth Diff: {diff:.2f}m)")
                            state["notified"] = True
                            
                            # VLM을 위한 스냅샷 저장
                            context_img = frame_bgr.copy()
                            cv2.rectangle(context_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            try:
                                vlm_queue.put_nowait((context_img, track_id, person_depth, zone_depth))
                            except queue.Full:
                                pass

        # Tracker State 정리
        disappeared_ids = list(tracker_state.keys() - current_ids)
        for d_id in disappeared_ids:
            del tracker_state[d_id]

        cv2.polylines(frame_bgr, [ROI_POLYGON], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # GStreamerApp의 큐에 프레임 전달 (OpenCV 별도 프로세스 출력)
        user_data.set_frame(frame_bgr)
            
    except Exception as e:
        print(f"❌ [app_callback Error] {e}")

def main():
    print("🚀 병렬 파이프라인 최적화 실행 (YOLO: CPU, Depth: NPU, VLM: NPU)")
    
    # 메인 인자 자동 설정 (지연 방지)
    if "--input" not in sys.argv:
        sys.argv.extend(["--input", "usb"])
    if "--use-frame" not in sys.argv:
        sys.argv.append("--use-frame")
    if "--width" not in sys.argv:
        sys.argv.extend(["--width", "640"])
    if "--height" not in sys.argv:
        sys.argv.extend(["--height", "480"])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, "yolo26n_ncnn_model")
    
    model = YOLO(model_path, task="detect")
    
    vlm_thread = threading.Thread(target=vlm_worker_thread, daemon=True)
    vlm_thread.start()
    
    user_data = ParallelAppCallback(model)
    app = GStreamerDepthApp(app_callback, user_data)
    app.video_sink = "fakesink"
    
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        vlm_thread.join()

if __name__ == "__main__":
    main()
