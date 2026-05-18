import os
import sys

# GStreamer 하드웨어 디코더 플러그인 중 일부 픽셀 포맷과 충돌하는 모듈들을 비활성화
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE,v4l2slh265dec:NONE,v4l2slh264dec:NONE,v4l2h265dec:NONE,v4l2h264dec:NONE"
# Headless 환경(디스플레이가 없는 서버 등)에서 OpenCV GUI 관련 에러를 방지
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import time
import queue
import threading
from pathlib import Path

import cv2
import numpy as np

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
        self.frame_count = 0
        self.video_writer = None
        self.video_writer = None

def get_roi_depth(depth_map, x1, y1, x2, y2):
    # 경계값 확인
    y1, y2 = max(0, y1), min(depth_map.shape[0], y2)
    x1, x2 = max(0, x1), min(depth_map.shape[1], x2)
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
        
    roi_depth_values = depth_map[y1:y2, x1:x2].flatten()
    if len(roi_depth_values) == 0:
        return 0.0
        
    # 아웃라이어 제거 후 중앙값 또는 평균 사용
    roi_depth_values = roi_depth_values[roi_depth_values > 0.1] # 0.1m 이하 노이즈 제거
    if len(roi_depth_values) == 0:
        return 0.0
        
    return float(np.median(roi_depth_values))

def app_callback(element, buffer, user_data):
    try:
        if buffer is None:
            return

        user_data.frame_count += 1
        if user_data.frame_count % 2 != 0:
            return

        # 프레임 추출
        format, width, height = get_caps_from_pad(element.get_static_pad("sink"))
        frame_raw = get_numpy_from_buffer(buffer, format, width, height)
        
        if frame_raw is None:
            return
            
        # 프레임이 읽기 전용 버퍼일 수 있으므로 복사본 생성
        frame = frame_raw.copy()

        # 4채널(RGBA/BGRA)일 경우 3채널(RGB/BGR)로 변환
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
        # 원본 프레임이 RGB인 경우, VideoWriter 저장을 위해 BGR로 변환
        # (YOLO NCNN 모델은 BGR 이미지를 입력으로 받아도 자체 처리하지만, OpenCV 저장은 BGR 포맷을 가정)
        # 만약 색상이 이상하게 저장된다면 이 부분을 제거하거나 cv2.COLOR_RGB2BGR를 사용하세요.
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Hailo outputs are often RGB
        
        # 깊이 정보 추출 (SCDepthV3 결과)
        roi = hailo.get_roi_from_buffer(buffer)
        depth_objs = roi.get_objects_typed(hailo.HAILO_DEPTH_MASK)
        
        depth_map = None
        if len(depth_objs) > 0:
            depth_data = depth_objs[0].get_data()
            
            # scdepthv3: 320x256
            depth_array = np.array(depth_data)
            tensor_w, tensor_h = 320, 256
            if len(depth_array) == tensor_w * tensor_h:
                depth_map = depth_array.reshape((tensor_h, tensor_w))
                depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)
            else:
                depth_map = None
        
        # YOLO 추론 (CPU)
        results = user_data.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)
        current_ids = set()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                current_ids.add(track_id)
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                
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
                        print(f"[Depth] ID {track_id} - Person: {person_depth:.2f}m, Zone: {zone_depth:.2f}m (Diff: {diff:.2f}m)", flush=True)
                        
                        if diff <= DEPTH_SIMILARITY_THRESHOLD:
                            print(f"👀 ID {track_id} 객체의 Depth가 구역과 일치합니다. VLM 분석 요청!", flush=True)
                            state["notified"] = True
                            
                            context_img = frame.copy()
                            cv2.rectangle(context_img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                            
                            try:
                                vlm_queue.put_nowait((context_img, track_id, person_depth, zone_depth))
                            except queue.Full:
                                try: vlm_queue.get_nowait()
                                except queue.Empty: pass
                                vlm_queue.put_nowait((context_img, track_id, person_depth, zone_depth))

        # Tracker State 정리
        disappeared_ids = list(tracker_state.keys() - current_ids)
        for d_id in disappeared_ids:
            del tracker_state[d_id]

        cv2.polylines(frame, [ROI_POLYGON], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # VideoWriter 초기화 (최초 1회)
        if user_data.video_writer is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "output_arch1_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            user_data.video_writer = cv2.VideoWriter(output_path, fourcc, 15.0, (width, height))
            print(f"🎬 비디오 저장 시작: {output_path} (Opened: {user_data.video_writer.isOpened()})", flush=True)

        # Depth 맵이 있으면 원본 프레임과 반투명하게 합성
        if depth_map is not None:
            depth_vis = np.clip(depth_map / 5.0 * 255, 0, 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 0.7, depth_colormap, 0.3, 0)

        # OpenCV 저장을 위해 RGB를 BGR로 변환
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if user_data.video_writer is not None and user_data.video_writer.isOpened():
            user_data.video_writer.write(frame_bgr)
            
    except Exception as e:
        import traceback
        print(f"❌ [app_callback Error] {e}", flush=True)
        traceback.print_exc()

def main():
    # -----------------------------------------------------------------------
    # GStreamer 하드웨어 디코더 플러그인 중 일부 픽셀 포맷과 충돌하는 모듈들을 비활성화
    # (depth_pipeline 내부에서 덮어씌울 수 있으므로 init 전에 강제로 다시 설정)
    # -----------------------------------------------------------------------
    os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE,v4l2slh265dec:NONE,v4l2slh264dec:NONE,v4l2h265dec:NONE,v4l2h264dec:NONE"

    print("🚀 실시간 병렬 실행 (YOLO: CPU, SCDepthV3: NPU, VLM: NPU) - Video Input")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 비디오 파일 입력 설정
    video_path = os.path.join(project_root, "_inputs", "test_video_1.MP4")
    if not os.path.exists(video_path):
        print(f"❌ 입력 비디오 파일이 존재하지 않습니다: {video_path}")
        sys.exit(1)
        
    if "--input" not in sys.argv:
        sys.argv.extend(["--input", video_path])
        print(f"🎬 비디오 파일 모드로 실행합니다: {video_path}")

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
        print("\n🛑 사용자에 의해 중단되었습니다.", flush=True)
        stop_event.set()
    finally:
        stop_event.set()
        print("💾 비디오 저장 객체를 해제합니다...", flush=True)
        if hasattr(user_data, 'video_writer') and user_data.video_writer is not None:
            user_data.video_writer.release()
            print("✅ 비디오 저장 완료: _outputs/output_arch1_video.mp4", flush=True)
        else:
            print("⚠️ 저장할 비디오 객체가 없습니다.", flush=True)
        # cv2.destroyAllWindows()
        vlm_thread.join(timeout=2.0)

if __name__ == "__main__":
    main()
