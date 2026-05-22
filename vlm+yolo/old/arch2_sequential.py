import os
import sys
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

# Hailo Imports
hailo_apps_dir = (Path.home() / "hailo-apps").resolve()
if str(hailo_apps_dir) not in sys.path:
    sys.path.insert(0, str(hailo_apps_dir))

try:
    from hailo_apps.python.core.common.core import resolve_hef_path
    from hailo_apps.python.core.common.defines import VLM_CHAT_APP, DEPTH_PIPELINE, SHARED_VDEVICE_GROUP_ID, HAILO10H_ARCH
    from hailo_platform import VDevice, HEF, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType, HailoStreamInterface
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
depth_queue = queue.LifoQueue(maxsize=1)
stop_event = threading.Event()
tracker_state = {}

# 공유 VDevice 관리
global_vdevice = None

# -----------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------
def get_roi_depth(depth_map, x1, y1, x2, y2):
    y1, y2 = max(0, y1), min(depth_map.shape[0], y2)
    x1, x2 = max(0, x1), min(depth_map.shape[1], x2)
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
        
    roi_depth_values = depth_map[y1:y2, x1:x2].flatten()
    roi_depth_values = roi_depth_values[roi_depth_values > 0.1]
    if len(roi_depth_values) == 0:
        return 0.0
    return float(np.median(roi_depth_values))

# -----------------------------------------------------------------------
# VLM Worker Thread (NPU - 항시 실행)
# -----------------------------------------------------------------------
def vlm_worker_thread():
    global global_vdevice
    print("🤖 [VLM Worker] 초기화 시작...")
    hef_path = resolve_hef_path(None, app_name=VLM_CHAT_APP, arch=HAILO10H_ARCH)
    if not hef_path:
        print("❌ [VLM Worker] VLM 모델을 찾을 수 없습니다.")
        return
        
    vlm = None
    try:
        # VDevice 생성 대기 (메인 스레드에서 설정)
        while global_vdevice is None and not stop_event.is_set():
            time.sleep(0.1)
            
        vlm = VLM(global_vdevice, str(hef_path))
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

# -----------------------------------------------------------------------
# Depth Worker Thread (NPU - 필요시 트리거됨)
# -----------------------------------------------------------------------
def depth_worker_thread():
    global global_vdevice
    print("📏 [Depth Worker] 대기 중... (트리거 시 NPU 실행)")
    hef_path = resolve_hef_path(None, app_name=DEPTH_PIPELINE, arch=HAILO10H_ARCH)
    if not hef_path:
        print("❌ [Depth Worker] Depth HEF 모델을 찾을 수 없습니다.")
        return
        
    try:
        while global_vdevice is None and not stop_event.is_set():
            time.sleep(0.1)
            
        hef = HEF(str(hef_path))
        # 스트림 인터페이스는 환경에 따라 다를 수 있습니다.
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_groups = global_vdevice.configure(hef, configure_params)
        network_group = network_groups[0]
        
        input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        
        input_name = list(input_vstreams_params.keys())[0]
        output_name = list(output_vstreams_params.keys())[0]
        
        input_info = hef.get_input_vstream_infos()[0]
        input_shape = input_info.shape
        hef_h, hef_w = input_shape[1], input_shape[2]

        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            while not stop_event.is_set():
                try:
                    item = depth_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                frame, track_id, (x1, y1, x2, y2) = item
                orig_h, orig_w = frame.shape[:2]
                
                print(f"📏 [Depth Worker] 프레임 수신! NPU Depth 추론 시작... (ID: {track_id})")
                
                # 전처리
                resized_frame = cv2.resize(frame, (hef_w, hef_h))
                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                
                input_data = {input_name: np.expand_dims(resized_frame, axis=0)}
                
                # NPU 추론
                infer_results = infer_pipeline.infer(input_data)
                tensor_data = infer_results[output_name][0]
                
                # 후처리 (NumPy) - C++ postprocess 대체
                output = np.exp(-tensor_data)
                output = 1 / (1 + output)
                depth_map = 1 / (output * 10 + 0.009)
                
                # 원본 해상도로 복원
                depth_map = cv2.resize(depth_map, (orig_w, orig_h))
                
                # Depth 검증
                person_depth = get_roi_depth(depth_map, x1, y1, x2, y2)
                rx, ry, rw, rh = cv2.boundingRect(ROI_POLYGON)
                zone_depth = get_roi_depth(depth_map, rx, ry, rx+rw, ry+rh)
                
                diff = abs(person_depth - zone_depth)
                print(f"[Depth] ID {track_id} - Person: {person_depth:.2f}m, Zone: {zone_depth:.2f}m (Diff: {diff:.2f}m)")
                
                if diff <= DEPTH_SIMILARITY_THRESHOLD:
                    print(f"👀 ID {track_id} 객체의 Depth가 구역과 일치합니다. VLM 분석 요청!")
                    
                    context_img = frame.copy()
                    cv2.rectangle(context_img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    
                    try:
                        vlm_queue.put_nowait((context_img, track_id, person_depth, zone_depth))
                    except queue.Full:
                        try: vlm_queue.get_nowait()
                        except queue.Empty: pass
                        vlm_queue.put_nowait((context_img, track_id, person_depth, zone_depth))
                        
                depth_queue.task_done()
    except Exception as e:
        print(f"❌ [Depth Worker] 에러: {e}")


# -----------------------------------------------------------------------
# Main Thread (YOLO CPU)
# -----------------------------------------------------------------------
def main():
    global global_vdevice
    print("🚀 순차 트리거 실행 (YOLO: CPU -> SCDepthV3: NPU -> VLM: NPU)")
    
    # 1. NCNN 모델 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, "yolo26n_ncnn_model")
    model = YOLO(model_path, task="detect")
    
    # 2. VDevice 초기화 (공유)
    try:
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        global_vdevice = VDevice(params)
    except Exception as e:
        print(f"❌ VDevice 초기화 실패: {e}")
        sys.exit(1)
    
    # 3. 워커 스레드 시작
    vlm_thread = threading.Thread(target=vlm_worker_thread, daemon=True)
    vlm_thread.start()
    
    depth_thread = threading.Thread(target=depth_worker_thread, daemon=True)
    depth_thread.start()
    
    # 4. 카메라 캡처 (OpenCV)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        stop_event.set()
        return

    frame_skip = 2
    frame_count = 0
    
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            # YOLO 추론 (CPU)
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)
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
                        
                        if elapsed >= ENTER_THRESHOLD_SEC and not state["notified"]:
                            print(f"👀 ID {track_id} 객체가 3초 이상 ROI에 머물렀습니다. Depth 검증 요청!")
                            state["notified"] = True
                            
                            # Depth Worker로 프레임 전달 (비동기)
                            try:
                                depth_queue.put_nowait((frame.copy(), track_id, (x1, y1, x2, y2)))
                            except queue.Full:
                                pass # Queue가 꽉 찼으면 무시

            disappeared_ids = list(tracker_state.keys() - current_ids)
            for d_id in disappeared_ids:
                del tracker_state[d_id]

            cv2.polylines(frame, [ROI_POLYGON], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.imshow("Arch2: Sequential Trigger", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ 에러: {e}")
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        if global_vdevice: global_vdevice.release()
        print("✅ 프로그램 종료")

if __name__ == "__main__":
    main()
