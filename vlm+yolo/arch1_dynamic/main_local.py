import os
import sys
import time
import queue
import threading

import cv2
import numpy as np

# NCNN 환경 설정
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE,v4l2slh265dec:NONE,v4l2slh264dec:NONE,v4l2h265dec:NONE,v4l2h264dec:NONE"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.fonts=false"

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics 패키지가 필요합니다.")
    sys.exit(1)

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from pathlib import Path
hailo_apps_dir = (Path.home() / "hailo-apps").resolve()
if str(hailo_apps_dir) not in sys.path:
    sys.path.insert(0, str(hailo_apps_dir))

from hailo_apps.python.pipeline_apps.depth.depth_pipeline import GStreamerDepthApp
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.python.core.common.buffer_utils import get_numpy_from_buffer
import hailo
from hailo_apps.python.core.common.core import resolve_hef_path
from hailo_apps.python.core.common.defines import VLM_CHAT_APP, SHARED_VDEVICE_GROUP_ID, HAILO10H_ARCH
from hailo_platform import VDevice
from hailo_platform.genai import VLM

from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    INFERENCE_PIPELINE, INFERENCE_PIPELINE_WRAPPER, USER_CALLBACK_PIPELINE,
)

# 모듈 Import
from .config import zone_config_queue, vlm_queue, stop_event, MODEL_PATH, SAVE_DIR
from .zone_manager import ZoneManager
from .local_listener import start_local_listener

def get_caps_from_pad_fixed(pad):
    caps = pad.get_current_caps()
    if not caps: return None, None, None
    structure = caps.get_structure(0)
    if not structure: return None, None, None
    real_structure = getattr(structure, '_StructureWrapper__structure', structure)
    try: return real_structure.get_value("format"), real_structure.get_value("width"), real_structure.get_value("height")
    except AttributeError: return None, None, None

class HeadlessDepthApp(GStreamerDepthApp):
    def get_pipeline_string(self):
        source_pipeline = self.get_source_pipeline()
        depth_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path, post_process_so=self.post_process_so,
            post_function_name=self.post_function_name, name="depth_inference"
        )
        depth_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(depth_pipeline, bypass_max_size_buffers=2, name="inference_wrapper_depth")
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        return f"{source_pipeline} ! {depth_pipeline_wrapper} ! {user_callback_pipeline} ! fakesink sync=false"

def vlm_worker_thread():
    print("🤖 [VLM Worker] 초기화 시작...")
    hef_path = resolve_hef_path(None, app_name=VLM_CHAT_APP, arch=HAILO10H_ARCH)
    if not hef_path: return
    vdevice = vlm = None
    try:
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        vdevice = VDevice(params)
        vlm = VLM(vdevice, str(hef_path))
        print("✅ [VLM Worker] VLM 초기화 완료!")
        
        while not stop_event.is_set():
            try: item = vlm_queue.get(timeout=0.5)
            except queue.Empty: continue
            if item is None: break
            
            # Payload 딕셔너리 수신
            context_img = item["image"]
            track_id = item["track_id"]
            p_depth = item["p_depth"]
            r_depth = item["z_depth"]
            zone_id = item["zone_id"]
            
            vlm_img = cv2.resize(context_img, (336, 336))
            vlm_img = cv2.cvtColor(vlm_img, cv2.COLOR_BGR2RGB)
            prompt = [
                {"role": "system", "content": [{"type": "text", "text": "You are an AI assistant monitoring CCTV."}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"A person (depth: {p_depth:.2f}m) triggered zone '{zone_id}' (depth: {r_depth:.2f}m). Summarize the action."}]}
            ]
            print(f"\n🧠 [VLM] Zone '{zone_id}' -> ID {track_id} 분석 중...")
            try:
                response = vlm.generate_all(prompt=prompt, frames=[vlm_img], temperature=0.1, max_generated_tokens=30)
                clean_text = response.split(". [{'type'")[0].split("<|im_end|>")[0].strip()
                print("="*60 + f"\n🚨 [VLM 결과] Zone '{zone_id}' ID {track_id}: {clean_text}\n" + "="*60)
            except Exception as e: print(f"⚠️ [VLM Error] {e}")
            finally: vlm_queue.task_done()
    except Exception as e: print(f"❌ [VLM Worker] {e}")
    finally:
        if vlm: vlm.release()
        if vdevice: vdevice.release()

class DynamicAppCallback(app_callback_class):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.total_frames = 0
        self.status_start_time = time.time()
        self.status_frame_count = 0
        self.caps_info = None
        
        self.yolo_queue = queue.Queue(maxsize=1)
        self.yolo_ready = True

        # [SNAPSHOT] 정기적 스냅샷 설정 (10초 간격)
        self.last_snapshot_time = 0
        self.snapshot_interval = 10.0
        self.snapshot_dir = SAVE_DIR / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.yolo_thread = threading.Thread(target=self.yolo_worker, daemon=True)
        self.yolo_thread.start()

    def yolo_worker(self):
        print("✅ [YOLO Worker] 시작 (Dynamic Local Mode)!")
        zone_manager = ZoneManager()
        last_yolo_time = time.time()
        yolo_count = 0
        
        while not stop_event.is_set():
            self.yolo_ready = True
            
            # 구역 설정 업데이트 처리
            while not zone_config_queue.empty():
                try:
                    payload = zone_config_queue.get_nowait()
                    zone_manager.process_queue_events(payload)
                except queue.Empty: break

            try: data = self.yolo_queue.get(timeout=0.5)
            except queue.Empty: continue
            if data is None: break
            
            frame_raw, depth_raw, fmt = data
            try:
                h_orig, w_orig = frame_raw.shape[:2]
                color_conv = cv2.COLOR_RGB2BGR if fmt == "RGB" else cv2.COLOR_RGBA2BGR
                
                # YOLO 추론용 축소 이미지
                frame_small_rgb = cv2.resize(frame_raw, (320, 320), interpolation=cv2.INTER_LINEAR)
                frame_small_bgr = cv2.cvtColor(frame_small_rgb, color_conv)
                depth_map = np.array(depth_raw, dtype=np.float32).reshape((256, 320)) if depth_raw is not None else None
                
                results = self.model.track(frame_small_bgr, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)
                
                yolo_count += 1
                now = time.time()

                # [SNAPSHOT] 모든 활성 구역과 탐지 결과를 스냅샷으로 저장
                if now - self.last_snapshot_time >= self.snapshot_interval:
                    self.last_snapshot_time = now
                    snap = cv2.cvtColor(frame_raw, color_conv)
                    
                    # 모든 구역 그리기
                    for zid, z in zone_manager.zones.items():
                        if not z.is_active: continue
                        color = (255, 0, 0) # 파랑
                        if len(z.polygon) > 0:
                            cv2.polylines(snap, [z.polygon], True, color, 2)
                            cv2.putText(snap, zid, (z.polygon[0][0], z.polygon[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # 탐지된 사람 그리기
                    if results[0].boxes is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        ids = results[0].boxes.id.int().cpu().numpy() if results[0].boxes.id is not None else [None]*len(boxes)
                        for box, tid in zip(boxes, ids):
                            x1, y1, x2, y2 = map(int, [box[0]*(w_orig/320), box[1]*(h_orig/320), box[2]*(w_orig/320), box[3]*(h_orig/320)])
                            cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            if tid is not None: cv2.putText(snap, f"ID:{tid}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    ts = time.strftime("%Y%m%d-%H%M%S")
                    cv2.imwrite(str(self.snapshot_dir / f"dynamic_snap_{ts}.jpg"), snap)
                    print(f"📷 [SYSTEM] 로컬 다이나믹 스냅샷 저장됨: {ts}")

                if now - last_yolo_time >= 2.0:
                    print(f"📊 [YOLO SPEED] {yolo_count / (now - last_yolo_time):.1f} FPS")
                    yolo_count, last_yolo_time = 0, now

                zone_manager.check_zones(results, depth_map, frame_raw, color_conv, w_orig, h_orig)

            except Exception as e: print(f"❌ [YOLO Error] {e}")
            finally: self.yolo_queue.task_done()

def app_callback(element, buffer, user_data):
    try:
        if buffer is None: return
        curr_time = time.time()
        
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
            if user_data.yolo_queue.full(): user_data.yolo_queue.get_nowait()
            user_data.yolo_queue.put_nowait((frame_raw, depth_raw, fmt))
        except: pass
        
        user_data.total_frames += 1
        user_data.status_frame_count += 1
        if curr_time - user_data.status_start_time >= 1.0:
            print(f"⏱️ [PIPELINE] FPS: {user_data.status_frame_count / (curr_time - user_data.status_start_time):.1f}")
            user_data.status_start_time, user_data.status_frame_count = curr_time, 0
    except Exception as e: print(f"❌ [Callback Error] {e}")

def main():
    print("🚀 [DYNAMIC PIPELINE] 로컬 모드 시작")
    # 실행 시 인자 처리
    if "--input" not in sys.argv: sys.argv.extend(["--input", "usb"])
    if "--width" not in sys.argv: sys.argv.extend(["--width", "640"])
    if "--height" not in sys.argv: sys.argv.extend(["--height", "480"])
    
    model = YOLO(str(MODEL_PATH), task="detect")
    
    # 로컬 리스너 시작 (프로젝트 루트의 zones_config.json 감시)
    config_file = "zones_config.json"
    local_thread = threading.Thread(target=start_local_listener, args=(config_file,), daemon=True)
    local_thread.start()

    vlm_thread = threading.Thread(target=vlm_worker_thread, daemon=True)
    vlm_thread.start()
    
    user_data = DynamicAppCallback(model)
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
