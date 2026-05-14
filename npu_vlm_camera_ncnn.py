# NCNN 적용의 핵심 변경 사항
#    - 이전에 모델을 호출하던 부분을 model = YOLO("yolov8n.pt")에서 model = YOLO("yolo26n_ncnn_model", task="detect")로
#      변경했습니다.
#    - ultralytics 라이브러리는 내보낸 모델 디렉토리(여기서는 yolo26n_ncnn_model 폴더)를 불러오면 내부적으로 C++ 최적화된
#      NCNN 백엔드를 자동으로 구동하여 추론합니다. 이 덕분에 복잡한 전후처리 코드를 수동으로 작성할 필요 없이, 가장
#      최적화된 형태로 NCNN을 활용하고 ByteTrack 객체 추적 기능도 기존과 동일하게 사용할 수 있습니다.

#   필요 패키지 설치
#   NCNN을 파이썬 환경에서 사용하려면 아래 패키지가 설치되어 있어야 합니다.
#    1 pip install ultralytics ncnn

#   실행 방법
#    1 python3 npu_vlm_camera_ncnn.py

import sys
import time
import queue
import threading
from pathlib import Path

import cv2
import numpy as np

# YOLO 및 객체 추적을 위한 ultralytics 라이브러리 사용
# NCNN 백엔드를 사용하려면 ncnn 패키지가 추가로 필요합니다.
# 설치: pip install ultralytics ncnn
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics 패키지가 필요합니다. 'pip install ultralytics ncnn' 명령어로 설치해주세요.")
    sys.exit(1)

# -----------------------------------------------------------------------
# Hailo NPU / VLM 초기화
# -----------------------------------------------------------------------
hailo_apps_dir = (Path(__file__).parent / "hailo-apps").resolve()
if str(hailo_apps_dir) not in sys.path:
    sys.path.insert(0, str(hailo_apps_dir))

try:
    from hailo_apps.python.core.common.core import resolve_hef_path
    from hailo_apps.python.core.common.defines import VLM_CHAT_APP, SHARED_VDEVICE_GROUP_ID, HAILO10H_ARCH
    from hailo_platform import VDevice
    from hailo_platform.genai import VLM
except ImportError as e:
    print(f"❌ Hailo 라이브러리 임포트 실패: {e}")
    sys.exit(1)


# -----------------------------------------------------------------------
# 전역 설정 및 상태 변수
# -----------------------------------------------------------------------
ROI_POLYGON = np.array([[100, 100], [540, 100], [540, 380], [100, 380]], np.int32)
ENTER_THRESHOLD_SEC = 3.0 # ROI 진입 후 알림을 발생시키기 위한 대기 시간 (초)

# LIFO Queue (최대 크기 1)
vlm_queue = queue.LifoQueue(maxsize=1)
stop_event = threading.Event()
tracker_state = {}


# -----------------------------------------------------------------------
# VLM 알림 함수
# -----------------------------------------------------------------------
def send_alert(image, text):
    print("\n" + "="*50)
    print("🚨 [VLM 상황 인식 알림] 🚨")
    print(f"내용: {text}")
    print("="*50 + "\n")


# -----------------------------------------------------------------------
# VLM Worker Thread (소비자) - NPU 담당
# -----------------------------------------------------------------------
def vlm_worker_thread():
    print("🤖 [VLM Worker] 초기화 시작...")
    
    hef_path = resolve_hef_path(None, app_name=VLM_CHAT_APP, arch=HAILO10H_ARCH)
    if not hef_path:
        print("❌ [VLM Worker] VLM 모델(HEF)을 찾을 수 없습니다.")
        stop_event.set()
        return
        
    print(f"🤖 [VLM Worker] VLM 모델 로드 중: {hef_path}")
    
    vdevice = None
    vlm = None
    try:
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        vdevice = VDevice(params)
        vlm = VLM(vdevice, str(hef_path))
        print("🤖 [VLM Worker] VLM 초기화 완료 및 대기 중...")
        
        while not stop_event.is_set():
            try:
                item = vlm_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            crop_img, track_id = item
            
            vlm_img = cv2.resize(crop_img, (336, 336))
            if len(vlm_img.shape) == 3 and vlm_img.shape[2] == 3:
                vlm_img = cv2.cvtColor(vlm_img, cv2.COLOR_BGR2RGB)
                
            prompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an AI assistant that monitors CCTV and describes the situation."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Summarize this situation in less than 15 words."}
                    ]
                }
            ]
            
            print(f"\n🔍 [VLM Worker] 객체 ID {track_id} NPU 추론 시작...")
            try:
                response = vlm.generate_all(
                    prompt=prompt, 
                    frames=[vlm_img], 
                    temperature=0.1, 
                    seed=42, 
                    max_generated_tokens=50
                )
                clean_text = response.split(". [{'type'")[0].split("<|im_end|>")[0].strip()
                send_alert(crop_img, clean_text)
                
            except Exception as e:
                print(f"⚠️ [VLM Worker] 추론 중 에러 발생: {e}")
                
            finally:
                vlm_queue.task_done()
                
    except Exception as e:
        print(f"❌ [VLM Worker] 초기화/실행 에러: {e}")
    finally:
        print("🤖 [VLM Worker] 리소스 해제 중...")
        if vlm:
            try: vlm.clear_context(); vlm.release()
            except: pass
        if vdevice:
            try: vdevice.release()
            except: pass


# -----------------------------------------------------------------------
# Main Thread (생산자) - CPU/NCNN 담당
# -----------------------------------------------------------------------
def main():
    print("🚀 실시간 객체 감지 및 VLM 상황 인식 시스템 시작")
    
    # 1. NCNN 형식의 YOLO 모델 로드 (ultralytics가 자동으로 ncnn 모듈을 백엔드로 사용함)
    # yolo26n_ncnn_model 폴더 내의 파라미터와 bin 파일을 읽어 추론 수행
    model_path = "yolo26n_ncnn_model"
    print(f"📦 YOLO 모델(NCNN 백엔드) 로드 중: {model_path}")
    
    try:
        model = YOLO(model_path, task="detect")
    except Exception as e:
        print(f"❌ NCNN 모델 로드 실패: {e}")
        print("💡 팁: 'pip install ncnn' 명령어로 NCNN 패키지가 설치되어 있는지 확인해주세요.")
        sys.exit(1)
    
    # 2. VLM Worker Thread 시작
    vlm_thread = threading.Thread(target=vlm_worker_thread, daemon=True)
    vlm_thread.start()
    
    # 3. 카메라 캡처 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        stop_event.set()
        vlm_thread.join()
        return

    frame_skip = 2
    frame_count = 0
    
    print("🎥 카메라 스트리밍 시작... (종료: 'q' 키)")
    
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("⚠️ 카메라 프레임을 읽을 수 없습니다.")
                break
                
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            # NCNN 백엔드를 통한 감지 및 ByteTrack 추적 (클래스 0: 사람)
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)
            
            current_ids = set()

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()

                for box, track_id in zip(boxes, track_ids):
                    current_ids.add(track_id)
                    x1, y1, x2, y2 = map(int, box)
                    
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    
                    inside_roi = cv2.pointPolygonTest(ROI_POLYGON, (cx, cy), False) >= 0
                    
                    if inside_roi:
                        if track_id not in tracker_state:
                            tracker_state[track_id] = {"enter_time": time.time(), "notified": False}
                        
                        state = tracker_state[track_id]
                        elapsed = time.time() - state["enter_time"]
                        
                        if elapsed >= ENTER_THRESHOLD_SEC and not state["notified"]:
                            print(f"👀 ID {track_id} 객체가 3초 이상 ROI에 머물렀습니다. VLM 분석 요청!")
                            state["notified"] = True
                            
                            crop_x1, crop_y1 = max(0, x1), max(0, y1)
                            crop_x2, crop_y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                            crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                            
                            if crop_img.size > 0:
                                try:
                                    vlm_queue.put_nowait((crop_img, track_id))
                                except queue.Full:
                                    try: vlm_queue.get_nowait()
                                    except queue.Empty: pass
                                    vlm_queue.put_nowait((crop_img, track_id))
                                    print(f"🔄 VLM Queue 갱신 완료 (ID: {track_id} 최신 프레임 반영)")

            # Tracker State 정리
            disappeared_ids = list(tracker_state.keys() - current_ids)
            for d_id in disappeared_ids:
                del tracker_state[d_id]
                print(f"🗑️ 화면에서 사라진 ID {d_id} 제거 (Tracker State 관리)")

            cv2.polylines(frame, [ROI_POLYGON], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(frame, "ROI Area", (ROI_POLYGON[0][0], ROI_POLYGON[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            cv2.imshow("System", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 종료 신호 수신 ('q' 입력)")
                break

    except KeyboardInterrupt:
        print("\n🛑 사용자 인터럽트(Ctrl+C) 수신")
    except Exception as e:
        print(f"❌ 메인 루프 에러: {e}")
    finally:
        print("🧹 리소스 정리 중...")
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        vlm_thread.join()
        print("✅ 프로그램이 안전하게 종료되었습니다.")

if __name__ == "__main__":
    main()
