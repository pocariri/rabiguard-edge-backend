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

# 프로젝트 루트 및 hailo-apps 경로 설정
hailo_apps_dir = (Path.home() / "hailo-apps").resolve()

if str(hailo_apps_dir) not in sys.path:
    sys.path.insert(0, str(hailo_apps_dir))

try:
    from hailo_apps.python.pipeline_apps.depth.depth_pipeline import GStreamerDepthApp
    from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
    from hailo_apps.python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
    import hailo
except ImportError as e:
    print(f"❌ Hailo 라이브러리 임포트 실패: {e}")
    sys.exit(1)


# -----------------------------------------------------------------------
# 전역 설정
# -----------------------------------------------------------------------
ROI_POLYGON = np.array([[100, 100], [540, 100], [540, 380], [100, 380]], np.int32)
ENTER_THRESHOLD_SEC = 2.0  # 구역 진입 판단 시간
DEPTH_SIMILARITY_THRESHOLD = 0.5 # meter

# -----------------------------------------------------------------------
# Callback Class & Functions
# -----------------------------------------------------------------------
def get_roi_depth(depth_map, x1, y1, x2, y2):
    h, w = depth_map.shape
    # 640x480 -> 320x256 스케일 변환
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
        
    return float(np.mean(roi_depth_values)) # median 보다 빠른 mean 사용

class ParallelAppCallback(app_callback_class):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.last_proc_time = 0
        self.fps_interval = 1.0 / 30.0  # 메인 콜백: 30FPS로 부드럽게 유지
        
        # 비동기 YOLO 추론을 위한 스레드 설정 (지연 완전 해결)
        self.latest_boxes = []
        self.tracker_state = {}
        self.yolo_queue = queue.Queue(maxsize=1)
        
        self.yolo_thread = threading.Thread(target=self.yolo_worker, daemon=True)
        self.yolo_thread.start()
        
    def yolo_worker(self):
        print("🤖 [YOLO Worker] 백그라운드 추론 스레드 시작...")
        while True:
            data = self.yolo_queue.get()
            if data is None: 
                break
            frame_bgr, depth_map = data
            
            try:
                # imgsz 제한 제거 (ByteTrack 충돌 방지), CPU에서 낼 수 있는 최대 속도로 비동기 실행
                results = self.model.track(frame_bgr, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)
                
                current_ids = set()
                boxes_data = []

                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()

                    for box, track_id in zip(boxes, track_ids):
                        current_ids.add(track_id)
                        boxes_data.append((box, track_id))
                        
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        
                        # 구역 내 진입 여부 체크
                        inside_roi = cv2.pointPolygonTest(ROI_POLYGON, (cx, cy), False) >= 0
                        if inside_roi:
                            if track_id not in self.tracker_state:
                                self.tracker_state[track_id] = {"enter_time": time.time(), "notified": False}
                            
                            state = self.tracker_state[track_id]
                            elapsed = time.time() - state["enter_time"]
                            
                            if elapsed >= ENTER_THRESHOLD_SEC and not state["notified"] and depth_map is not None:
                                person_depth = get_roi_depth(depth_map, x1, y1, x2, y2)
                                rx, ry, rw, rh = cv2.boundingRect(ROI_POLYGON)
                                zone_depth = get_roi_depth(depth_map, rx, ry, rx+rw, ry+rh)
                                
                                diff = abs(person_depth - zone_depth)
                                if diff <= DEPTH_SIMILARITY_THRESHOLD:
                                    print(f"🚨 [Alert] ID {track_id} is in the ZONE! (Depth: {person_depth:.2f}m)")
                                    state["notified"] = True

                # Tracker State 정리
                disappeared_ids = list(self.tracker_state.keys() - current_ids)
                for d_id in disappeared_ids:
                    del self.tracker_state[d_id]

                # 메인 스레드 화면 출력을 위해 최신 박스 정보 업데이트
                self.latest_boxes = boxes_data

            except Exception as e:
                print(f"❌ [YOLO Worker Error] {e}")


def app_callback(element, buffer, user_data):
    try:
        if buffer is None:
            return

        # 메인 GStreamer 콜백은 프레임을 가공하고 렌더링하는 역할만 수행 (Non-blocking)
        curr_time = time.time()
        if curr_time - user_data.last_proc_time < user_data.fps_interval:
            return
        user_data.last_proc_time = curr_time

        format, width, height = get_caps_from_pad(element.get_static_pad("sink"))
        frame_raw = get_numpy_from_buffer(buffer, format, width, height)
        if frame_raw is None:
            return
            
        frame = frame_raw.copy()

        # RGB 변환
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        # 깊이(Depth) 추출 및 시각화 합성
        roi = hailo.get_roi_from_buffer(buffer)
        depth_objs = roi.get_objects_typed(hailo.HAILO_DEPTH_MASK)
        
        depth_map = None
        if len(depth_objs) > 0:
            depth_data = depth_objs[0].get_data()
            tensor_w, tensor_h = 320, 256
            depth_array = np.array(depth_data)
            
            if len(depth_array) == tensor_w * tensor_h:
                depth_map = depth_array.reshape((tensor_h, tensor_w))
                
                depth_vis = np.clip(depth_map / 5.0 * 255, 0, 255).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                depth_colormap_resized = cv2.resize(depth_colormap, (width, height), interpolation=cv2.INTER_LINEAR)
                
                frame = cv2.addWeighted(frame, 0.5, depth_colormap_resized, 0.5, 0)

        # OpenCV 출력을 위해 BGR로 변환
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # YOLO 백그라운드 스레드에 최신 프레임 비동기 전달
        try:
            user_data.yolo_queue.put_nowait((frame_bgr.copy(), depth_map))
        except queue.Full:
            pass # 큐가 찼다면(YOLO가 아직 이전 프레임 추론 중이라면) 스킵
        
        # 백그라운드에서 완료된 최신 YOLO 박스 정보를 화면에 표시
        for box, track_id in user_data.latest_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"ID: {track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ROI 구역 그리기
        cv2.polylines(frame_bgr, [ROI_POLYGON], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # 화면 출력
        user_data.set_frame(frame_bgr)
            
    except Exception as e:
        print(f"❌ [app_callback Error] {e}")

def main():
    print("🚀 YOLO(CPU Async) + Depth(NPU) 무지연 최적화 파이프라인 시작")
    print("💡 종료하려면 Ctrl+C를 한 번만 누르세요.")
    
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
    
    user_data = ParallelAppCallback(model)
    app = GStreamerDepthApp(app_callback, user_data)
    app.video_sink = "fakesink"
    
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        user_data.yolo_queue.put(None) # 워커 스레드 종료 시그널
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
