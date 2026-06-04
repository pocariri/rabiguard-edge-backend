import asyncio
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
from firebase_admin import credentials, db, initialize_app
import numpy as np

## 1. 초기화
cred = credentials.Certificate("/home/rafour/workspace/seungmin/rafour-app/serviceAccountKey.json")
initialize_app(cred, {'databaseURL': 'https://rafour-7f37f-default-rtdb.asia-southeast1.firebasedatabase.app/'})

# 2. 카메라 영상 트랙 정의 클래스
class CameraStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0) # 0번 카메라 연결

        # 해상도를 명시적으로 지정하면 스트림이 더 안정적
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    async def recv(self):
        pts, time_base = await self.next_timestamp()
        
        # 카메라에서 프레임 읽기 시도
        ret, frame = self.cap.read()

        # 프레임 획득 실패 시 검은 화면이라도 생성 (None 에러 방지 핵심)
        if not ret or frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 중요: OpenCV(BGR)를 PyAV가 선호하는 RGB로 확실히 변환
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 프레임 생성 및 속성 부여 (이 부분이 정확해야 에러가 안 남)
        new_frame = VideoFrame.from_ndarray(img, format="rgb24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        
        return new_frame
	
    # 종료 시 카메라 해제 (선택 사항)
    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

pc = RTCPeerConnection()
relay = MediaRelay()

# 3. Answer 생성 함수 (여기에 트랙 추가 로직 포함)
async def create_answer(offer_sdp):
    try:
        print("=== Offer SDP ===")
        print(offer_sdp)
        print("=================")
        # offer를 먼저 세팅한 다음에 트랙 추가
        offer = RTCSessionDescription(sdp=offer_sdp, type='offer')
        await pc.setRemoteDescription(offer)        
        
        # 카메라 트랙을 연결에 추가 (비디오 전송 시작)
        camera_track = CameraStreamTrack()
        pc.addTrack(camera_track)
        print("비디오 트랙이 연결에 추가되었습니다.")

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        db.reference('signaling/smart_cctv/answer').set({
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        })
        print("영상 정보를 담은 Answer 전송 완료!")
    except Exception as e:
        import traceback
        traceback.print_exc()  # 이걸로 전체 스택 트레이스 출력
        print(f"오류 발생: {e}")

def on_offer_received(event):
    print(f"[WebRTC] 이벤트 수신: {event.data}")
    if not event.data:
        print("[WebRTC] 데이터 없음 (삭제 이벤트), 무시")
        return
    if not isinstance(event.data, dict):
        print("[WebRTC] 딕셔너리 아님, 무시")
        return
    offer_sdp = event.data.get('sdp')
    if not offer_sdp:
        print("[WebRTC] SDP 없음, 무시")
        return
    print("\n[감지] 새로운 Offer 수신")
    asyncio.run_coroutine_threadsafe(create_answer(offer_sdp), main_loop)

# 메인 실행부
print("WebRTC 시그널링 대기 중... (Ctrl+C로 종료)")

# 1. 메인 루프를 미리 생성
main_loop = asyncio.new_event_loop()
asyncio.set_event_loop(main_loop)

# 2. signaling 초기화
db.reference('signaling/smart_cctv/offer').delete()
db.reference('signaling/smart_cctv/answer').delete()
print("[WebRTC] 기존 signaling 데이터 초기화 완료")

# 잠깐 대기 (삭제 이벤트 전파 완료 대기)
import time
time.sleep(0.5)

# 3. 리스너 등록
db.reference('signaling/smart_cctv/offer').listen(on_offer_received)

# 4. ready 신호 전송 (리스너 등록 후)
db.reference('signaling/smart_cctv/stream_status').set('ready')
print("[WebRTC] Ready 신호 전송 완료")

# 5. 루프 실행
try:
    main_loop.run_forever()
except KeyboardInterrupt:
    print("\n정지합니다...")
finally:
    main_loop.run_until_complete(pc.close())
