#   적용된 주요 아키텍처 및 로직

#    1. 하드웨어 분배 및 스레딩 (생산자-소비자 패턴)
#       - Main Thread (CPU): cv2.VideoCapture에서 프레임을 읽어와서 Ultralytics의 YOLO 모델을 사용하여 객체(사람)를
#         감지합니다. 이 라이브러리는 내장된 ByteTrack을 사용하여 빠르고 정확하게 객체를 추적합니다. 프레임
#         스킵(frame_skip = 2)이 적용되어 CPU 부하를 완화합니다.
#       - VLM Thread (Hailo NPU): Qwen 모델 기반의 Vision Language Model을 호출하여, vlm_queue에 들어온 크롭 이미지를
#         분석합니다.
#       - LIFO Queue (크기 1): Main Thread는 VLM이 작업 중일 경우 queue.Full 예외를 잡아서 가장 오래된 이미지를 버리고
#         최신 이미지로 교체(get_nowait() -> put_nowait())하여 실시간성을 보장합니다.
#    2. ROI 감지 및 상태 관리
#       - 지정된 ROI_POLYGON 안에 객체의 중심점이 있는지를 cv2.pointPolygonTest로 실시간 판단합니다.
#       - tracker_state 전역 딕셔너리로 진입 시간 및 알림 발송 여부를 관리합니다. 진입 후 3초가 지나면 VLM 큐에 객체를
#         전송합니다.
#       - 카메라 화면 밖으로 벗어나거나 가려져 ID가 손실된 경우 tracker_state에서 삭제하여 메모리 누수를 원천
#         차단했습니다.
#    3. VLM 제약 조건 및 알림 처리
#       - VLM 프롬프트를 "Summarize this situation in less than 15 words." 와 같이 영어로 설정하여, Hailo의 Qwen 모델이
#         원활하게 응답할 수 있도록 구성했습니다.
#       - 추론된 텍스트와 원본 크롭 이미지는 터미널 출력용 send_alert 함수로 처리됩니다. 객체당 단 1회 알림만 전송됩니다.
#    4. Graceful Shutdown
#       - threading.Event()를 사용하여 메인 스레드와 VLM 작업 스레드 간의 안전한 종료(Resource Release, Context Clear)를
#         지원합니다.

#   실행 방법

#    1 python3 npu_vlm_camera.py
#   > 참고: 코드에 사용된 ultralytics 패키지(YOLO 감지+ByteTrack 통합 지원 모듈)가 라즈베리파이에 없다면, pip install
#   ultralytics onnxruntime 명령어로 설치가 필요합니다. Ultralytics 패키지를 통해 제공된 ONNX 모델(.onnx)을 사용할 경우
#   ONNX Runtime을 자동으로 활용합니다. 직접 보유하신 yolo26n.onnx 파일이 있다면 코드의 model_path를 변경하여 실행하시면
#   됩니다.

import sys
import time
import queue
import threading
from pathlib import Path

import cv2
import numpy as np

# YOLO 및 객체 추적을 위한 ultralytics 라이브러리 사용 (ONNX Runtime 자동 활용)
# 패키지가 없다면 터미널에서 설치: pip install ultralytics onnxruntime
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics 패키지가 필요합니다. 'pip install ultralytics onnxruntime' 명령어로 설치해주세요.")
    sys.exit(1)

# -----------------------------------------------------------------------
# Hailo NPU / VLM 초기화
# -----------------------------------------------------------------------
# hailo-apps 모듈 시스템 경로 추가
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

# 스레드 종료 플래그
stop_event = threading.Event()

# Tracker State 관리 Dictionary
# 구조: { track_id: {"enter_time": float, "notified": bool} }
tracker_state = {}


# -----------------------------------------------------------------------
# VLM 알림 함수
# -----------------------------------------------------------------------
def send_alert(image, text):
    """
    VLM 응답을 터미널에 출력하는 알림 함수
    """
    print("\n" + "="*50)
    print("🚨 [VLM 상황 인식 알림] 🚨")
    print(f"내용: {text}")
    print("="*50 + "\n")
    # 필요한 경우 여기서 image를 저장하거나 서버로 전송할 수 있습니다.
    # cv2.imwrite(f"./outputs/alert_{time.time()}.jpg", image)


# -----------------------------------------------------------------------
# VLM Worker Thread (소비자)
# -----------------------------------------------------------------------
def vlm_worker_thread():
    """
    NPU (Hailo-10H) 기반 VLM 추론을 전담하는 스레드
    """
    print("🤖 [VLM Worker] 초기화 시작...")
    
    # 1. HEF 파일 경로 자동 탐색 (VLM 모델)
    hef_path = resolve_hef_path(None, app_name=VLM_CHAT_APP, arch=HAILO10H_ARCH)
    if not hef_path:
        print("❌ [VLM Worker] VLM 모델(HEF)을 찾을 수 없습니다.")
        stop_event.set()
        return
        
    print(f"🤖 [VLM Worker] VLM 모델 로드 중: {hef_path}")
    
    vdevice = None
    vlm = None
    try:
        # 2. VDevice 생성 및 VLM 인스턴스 초기화
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        vdevice = VDevice(params)
        vlm = VLM(vdevice, str(hef_path))
        print("🤖 [VLM Worker] VLM 초기화 완료 및 대기 중...")
        
        while not stop_event.is_set():
            try:
                # 큐에서 데이터를 기다림 (timeout을 주어 주기적으로 stop_event 확인)
                item = vlm_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            crop_img, track_id = item
            
            # VLM 모델 입력 크기에 맞춰 전처리 (보통 336x336 사용)
            vlm_img = cv2.resize(crop_img, (336, 336))
            if len(vlm_img.shape) == 3 and vlm_img.shape[2] == 3:
                vlm_img = cv2.cvtColor(vlm_img, cv2.COLOR_BGR2RGB)
                
            # Qwen 기반 VLM을 위한 영어 프롬프트 구성
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
                # NPU 추론 실행
                response = vlm.generate_all(
                    prompt=prompt, 
                    frames=[vlm_img], 
                    temperature=0.1, 
                    seed=42, 
                    max_generated_tokens=50
                )
                # 불필요한 태그 파싱 방지
                clean_text = response.split(". [{'type'")[0].split("<|im_end|>")[0].strip()
                
                # 결과 알림
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
            try:
                vlm.clear_context()
                vlm.release()
            except: pass
        if vdevice:
            try:
                vdevice.release()
            except: pass


# -----------------------------------------------------------------------
# Main Thread (생산자)
# -----------------------------------------------------------------------
def main():
    print("🚀 실시간 객체 감지 및 VLM 상황 인식 시스템 시작")
    
    # 1. YOLO 모델 로드 (ONNX 모델이 없다면 PyTorch(.pt) 모델이 다운로드되며 내부적으로 ONNX Runtime 처럼 활용됨)
    # 실제 ONNX 모델 파일이 있다면 'yolov8n.onnx' 처럼 명시 가능
    model_path = "yolov8n.pt"  # 혹은 "yolov8n.onnx"
    print(f"📦 YOLO 모델 로드 중: {model_path}")
    model = YOLO(model_path)
    
    # 2. VLM Worker Thread 시작
    vlm_thread = threading.Thread(target=vlm_worker_thread, daemon=True)
    vlm_thread.start()
    
    # 3. 카메라 캡처 초기화
    cap = cv2.VideoCapture(0) # 혹은 /dev/video0
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        stop_event.set()
        vlm_thread.join()
        return

    # 프레임 스킵 설정 (CPU 부하 감소)
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

            # YOLO 기반 객체 감지 및 ByteTrack 추적 (클래스 0: 사람만)
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)
            
            # 현재 프레임에서 감지된 객체 ID 목록 수집 (사라진 객체 확인용)
            current_ids = set()

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()

                for box, track_id in zip(boxes, track_ids):
                    current_ids.add(track_id)
                    x1, y1, x2, y2 = map(int, box)
                    
                    # 중심점 계산
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    # 화면에 BBox 및 ID 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    
                    # 중심점이 ROI 내부에 있는지 검사
                    inside_roi = cv2.pointPolygonTest(ROI_POLYGON, (cx, cy), False) >= 0
                    
                    if inside_roi:
                        # ROI 진입 처리
                        if track_id not in tracker_state:
                            tracker_state[track_id] = {
                                "enter_time": time.time(),
                                "notified": False
                            }
                        
                        state = tracker_state[track_id]
                        elapsed = time.time() - state["enter_time"]
                        
                        # 3초 경과 및 아직 알림을 보내지 않았다면
                        if elapsed >= ENTER_THRESHOLD_SEC and not state["notified"]:
                            print(f"👀 ID {track_id} 객체가 3초 이상 ROI에 머물렀습니다. VLM 분석 요청!")
                            
                            # 중복 알림 방지 상태 업데이트
                            state["notified"] = True
                            
                            # 객체 크롭 (경계값 안전성 확보)
                            crop_x1, crop_y1 = max(0, x1), max(0, y1)
                            crop_x2, crop_y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                            crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                            
                            # VLM Queue 삽입 (LifoQueue, Maxsize 1 제약 처리)
                            if crop_img.size > 0:
                                try:
                                    vlm_queue.put_nowait((crop_img, track_id))
                                except queue.Full:
                                    # 큐가 꽉 찼다면 (VLM이 바쁘다면) 이전 이미지를 버리고 최신 이미지로 덮어쓰기
                                    try:
                                        vlm_queue.get_nowait()
                                    except queue.Empty:
                                        pass
                                    vlm_queue.put_nowait((crop_img, track_id))
                                    print(f"🔄 VLM Queue 갱신 완료 (ID: {track_id} 최신 프레임 반영)")

                    else:
                        # 객체가 ROI 밖으로 나갔다면 (필요시 진입 시간 초기화 가능)
                        pass
            
            # Tracker State 정리 (메모리 누수 방지: 화면에서 사라진 ID 삭제)
            disappeared_ids = list(tracker_state.keys() - current_ids)
            for d_id in disappeared_ids:
                del tracker_state[d_id]
                print(f"🗑️ 화면에서 사라진 ID {d_id} 제거 (Tracker State 관리)")

            # 화면에 ROI 다각형 그리기
            cv2.polylines(frame, [ROI_POLYGON], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(frame, "ROI Area", (ROI_POLYGON[0][0], ROI_POLYGON[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 결과 화면 출력
            cv2.imshow("System", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 종료 신호 수신 ('q' 입력)")
                break

    except KeyboardInterrupt:
        print("\n🛑 사용자 인터럽트(Ctrl+C) 수신")
    except Exception as e:
        print(f"❌ 메인 루프 에러: {e}")
    finally:
        # Graceful Shutdown
        print("🧹 리소스 정리 중...")
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        vlm_thread.join()
        print("✅ 프로그램이 안전하게 종료되었습니다.")

if __name__ == "__main__":
    main()
