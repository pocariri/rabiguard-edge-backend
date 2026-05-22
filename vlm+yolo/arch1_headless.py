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
# 전역 설정 및 스레드 이벤트
# -----------------------------------------------------------------------
ROI_POLYGON = np.array([[100, 100], [540, 100], [540, 380], [100, 380]], np.int32)
ENTER_THRESHOLD_SEC = 2.0  # 구역 내 체류 시간
DEPTH_SIMILARITY_THRESHOLD = 0.5 # 거리 오차 허용 범위 (m)

stop_event = threading.Event()
vlm_queue = queue.LifoQueue(maxsize=1)

# -----------------------------------------------------------------------
# 유틸리티 함수
# -----------------------------------------------------------------------
def get_roi_depth(depth_map, x1, y1, x2, y2):
    if depth_map is None: return 0.0
    h, w = depth_map.shape
    # 원본 640x480 영상 좌표를 Depth 텐서 320x256 크기로 스케일링
    scale_y, scale_x = h / 480.0, w / 640.0
    
    tx1, ty1 = int(x1 * scale_x), int(y1 * scale_y)
    tx2, ty2 = int(x2 * scale_x), int(y2 * scale_y)

    ty1, ty2 = max(0, ty1), min(h, ty2)
    tx1, tx2 = max(0, tx1), min(w, tx2)
    
    if tx1 >= tx2 or ty1 >= ty2: return 0.0
        
    roi_depth_values = depth_map[ty1:ty2, tx1:tx2].flatten()
    roi_depth_values = roi_depth_values[roi_depth_values > 0.1] # 노이즈 제거
    
    if len(roi_depth_values) == 0: return 0.0
    return float(np.mean(roi_depth_values))

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
        print("✅ [VLM Worker] VLM 초기화 완료! 대기 중...")
        
        while not stop_event.is_set():
            try:
                item = vlm_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            if item is None: break # 종료 시그널
            context_img, track_id, p_depth, r_depth = item
            
            # VLM 입력 규격에 맞게 조정
            vlm_img = cv2.resize(context_img, (336, 336))
            if len(vlm_img.shape) == 3 and vlm_img.shape[2] == 3:
                vlm_img = cv2.cvtColor(vlm_img, cv2.COLOR_BGR2RGB)

            prompt = [
                {"role": "system", "content": [{"type": "text", "text": "You are an AI assistant that monitors CCTV and describes the situation."}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"A person (depth: {p_depth:.2f}m) is detected inside the restricted zone (depth: {r_depth:.2f}m). Please summarize what the person is doing in one short sentence."}]}
            ]
            
            print(f"\n🧠 [VLM] ID {track_id} 객체 상황 분석 시작...")
            try:
                response = vlm.generate_all(prompt=prompt, frames=[vlm_img], temperature=0.1, seed=42, max_generated_tokens=30)
                clean_text = response.split(". [{'type'")[0].split("<|im_end|>")[0].strip()
                print("="*60)
                print(f"🚨 [VLM 상황 요약 알림] 🚨")
                print(f"📍 객체 ID: {track_id}")
                print(f"📝 상황: {clean_text}")
                print("="*60 + "\n")
            except Exception as e:
                print(f"⚠️ [VLM Worker] 추론 중 에러 발생: {e}")
            finally:
                vlm_queue.task_done()
                
    except Exception as e:
        print(f"❌ [VLM Worker] 에러: {e}")
    finally:
        if vlm: vlm.release()
        if vdevice: vdevice.release()
        print("🛑 [VLM Worker] 종료됨.")


# -----------------------------------------------------------------------
# Callback Class & YOLO Worker
# -----------------------------------------------------------------------
class HeadlessAppCallback(app_callback_class):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.last_proc_time = 0
        self.fps_interval = 1.0 / 30.0  # 파이프라인 프레임 레이트 제어
        
        # 하트비트(FPS 모니터링) 용 변수
        self.total_frames = 0
        self.status_start_time = time.time()
        self.status_frame_count = 0
        
        # 비동기 YOLO 처리용
        self.tracker_state = {}
        self.yolo_queue = queue.Queue(maxsize=1)
        
        self.yolo_thread = threading.Thread(target=self.yolo_worker, daemon=True)
        self.yolo_thread.start()
        
    def yolo_worker(self):
        print("✅ [YOLO Worker] 백그라운드 추론 스레드 시작!")
        while not stop_event.is_set():
            try:
                data = self.yolo_queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            if data is None: break
            frame_bgr, depth_map = data
            
            try:
                # 화면 출력이 없으므로 최대한 빠르게 추론
                results = self.model.track(frame_bgr, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)
                current_ids = set()

                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()

                    for box, track_id in zip(boxes, track_ids):
                        current_ids.add(track_id)
                        
                        # 신규 객체 로깅
                        if track_id not in self.tracker_state:
                            print(f"👀 [TRACK] 신규 객체 탐지: ID {track_id}")
                            self.tracker_state[track_id] = {"enter_time": None, "notified": False}
                        
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        
                        state = self.tracker_state[track_id]
                        inside_roi = cv2.pointPolygonTest(ROI_POLYGON, (cx, cy), False) >= 0
                        
                        if inside_roi:
                            if state["enter_time"] is None:
                                state["enter_time"] = time.time()
                                print(f"⚠️ [ZONE] ID {track_id} ROI 구역 진입. 대기 중...")
                                
                            elapsed = time.time() - state["enter_time"]
                            if elapsed >= ENTER_THRESHOLD_SEC and not state["notified"]:
                                # --- 3D Depth 교차 검증 ---
                                person_depth = get_roi_depth(depth_map, x1, y1, x2, y2)
                                rx, ry, rw, rh = cv2.boundingRect(ROI_POLYGON)
                                zone_depth = get_roi_depth(depth_map, rx, ry, rx+rw, ry+rh)
                                
                                diff = abs(person_depth - zone_depth)
                                if diff <= DEPTH_SIMILARITY_THRESHOLD:
                                    print(f"🟢 [DEPTH] 검증 성공! ID {track_id} (사람: {person_depth:.2f}m, 구역: {zone_depth:.2f}m, 차이: {diff:.2f}m) -> VLM 전송")
                                    state["notified"] = True
                                    
                                    # VLM에 넘길 스냅샷 준비 (박스 및 ROI 그리기)
                                    context_img = frame_bgr.copy()
                                    cv2.rectangle(context_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                    cv2.polylines(context_img, [ROI_POLYGON], True, (255, 0, 0), 2)
                                    try:
                                        vlm_queue.put_nowait((context_img, track_id, person_depth, zone_depth))
                                    except queue.Full:
                                        pass
                                else:
                                    print(f"🔴 [DEPTH] 검증 실패. ID {track_id} (사람: {person_depth:.2f}m, 구역: {zone_depth:.2f}m, 차이: {diff:.2f}m)")
                                    # 재검증을 위해 시간 초기화 (원치 않으면 주석 처리)
                                    state["enter_time"] = time.time() 
                                    
                        else:
                            # 구역을 벗어나면 초기화
                            if state["enter_time"] is not None:
                                print(f"🏃 [ZONE] ID {track_id} ROI 구역 이탈.")
                                state["enter_time"] = None
                                state["notified"] = False

                # 화면 밖으로 사라진 객체 정리
                disappeared_ids = list(self.tracker_state.keys() - current_ids)
                for d_id in disappeared_ids:
                    print(f"👋 [TRACK] 객체 소실: ID {d_id}")
                    del self.tracker_state[d_id]

            except Exception as e:
                print(f"❌ [YOLO Worker Error] {e}")


def app_callback(element, buffer, user_data):
    try:
        if buffer is None: return

        # 프레임 간격 제어
        curr_time = time.time()
        if curr_time - user_data.last_proc_time < user_data.fps_interval: return
        user_data.last_proc_time = curr_time

        # 하트비트 로그 (1초마다 출력)
        user_data.total_frames += 1
        user_data.status_frame_count += 1
        elapsed_status = curr_time - user_data.status_start_time
        if elapsed_status >= 1.0:
            current_fps = user_data.status_frame_count / elapsed_status
            print(f"⏱️ [STATUS] 파이프라인 모니터링 중... | FPS: {current_fps:.1f} | 누적 프레임: {user_data.total_frames}")
            user_data.status_start_time = curr_time
            user_data.status_frame_count = 0

        format, width, height = get_caps_from_pad(element.get_static_pad("sink"))
        frame_raw = get_numpy_from_buffer(buffer, format, width, height)
        if frame_raw is None: return
            
        # 단 한 번의 색상 변환 (화면 출력이 없으므로 복사본 없이 BGR 직행)
        frame_bgr = cv2.cvtColor(frame_raw, cv2.COLOR_RGBA2BGR)
        
        # 깊이(Depth) 텐서 데이터 추출 (시각화 연산 없음)
        roi = hailo.get_roi_from_buffer(buffer)
        depth_objs = roi.get_objects_typed(hailo.HAILO_DEPTH_MASK)
        
        depth_map = None
        if len(depth_objs) > 0:
            depth_data = depth_objs[0].get_data()
            if len(depth_data) == 320 * 256:
                depth_map = np.array(depth_data).reshape((256, 320))
                
        # YOLO 스레드로 데이터 전송 (Non-blocking)
        try:
            user_data.yolo_queue.put_nowait((frame_bgr, depth_map))
        except queue.Full:
            pass
            
    except Exception as e:
        print(f"❌ [app_callback Error] {e}")


def main():
    print("="*60)
    print("🚀 [HEADLESS MODE] 무지연 백그라운드 파이프라인 시작")
    print("   (YOLO: CPU / Depth: NPU / VLM: NPU)")
    print("💡 화면 출력 없이 로그만 발생합니다. 종료 시 Ctrl+C 입력.")
    print("="*60)
    
    # 디스플레이 비활성화 인자
    if "--input" not in sys.argv: sys.argv.extend(["--input", "usb"])
    if "--width" not in sys.argv: sys.argv.extend(["--width", "640"])
    if "--height" not in sys.argv: sys.argv.extend(["--height", "480"])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, "yolo26n_ncnn_model")
    
    model = YOLO(model_path, task="detect")
    
    vlm_thread = threading.Thread(target=vlm_worker_thread, daemon=True)
    vlm_thread.start()
    
    user_data = HeadlessAppCallback(model)
    app = GStreamerDepthApp(app_callback, user_data)
    app.video_sink = "fakesink"  # 화면 출력 비활성화
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n🛑 프로그램 종료 요청 수신...")
    finally:
        # 안전한 스레드 종료
        stop_event.set()
        user_data.yolo_queue.put(None)
        vlm_queue.put(None)
        
        print("⏳ 스레드 종료 대기 중...")
        user_data.yolo_thread.join(timeout=2.0)
        vlm_thread.join(timeout=5.0)
        print("✅ 모든 자원이 안전하게 해제되었습니다.")

if __name__ == "__main__":
    main()
