# webrtc_video.py
import asyncio
import threading
import queue
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from firebase_admin import db

# ------------------------------------------------------------
# 공유 데이터 (main.py와 공유)
# ------------------------------------------------------------

# GStreamer 프레임 공유
shared_frame = None
shared_frame_lock = threading.Lock()

# 이미지 전송 큐 (main.py에서 이미지 경로 넣으면 Data Channel로 전송)
image_send_queue = queue.Queue()

def update_shared_frame(frame):
    global shared_frame
    with shared_frame_lock:
        shared_frame = frame.copy()

# ------------------------------------------------------------
# 공유 프레임 트랙
# ------------------------------------------------------------

class SharedFrameTrack(VideoStreamTrack):
    async def recv(self):
        pts, time_base = await self.next_timestamp()
        with shared_frame_lock:
            frame = shared_frame.copy() if shared_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        new_frame = VideoFrame.from_ndarray(img, format="rgb24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        return new_frame

# ------------------------------------------------------------
# WebRTC
# ------------------------------------------------------------

pc = RTCPeerConnection()
data_channel = None
main_loop = None
webrtc_app = None

async def create_answer(offer_sdp):
    global data_channel
    try:
        offer = RTCSessionDescription(sdp=offer_sdp, type='offer')
        await pc.setRemoteDescription(offer)

        data_channel = pc.createDataChannel("image")
        print("[WebRTC] Data Channel 생성 완료")

        # Data Channel 열렸을 때 큐에 있는 이미지 전송
        def on_data_channel_open():
            print("[WebRTC] Data Channel 열림, 이미지 전송 시작")
            asyncio.ensure_future(image_sender_loop())

        data_channel.on("open", on_data_channel_open)

        pc.addTrack(SharedFrameTrack())
        print("[WebRTC] 비디오 트랙 추가 완료")

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        from firebase_admin import db as webrtc_db
        webrtc_db.reference('signaling/smart_cctv/answer', app=webrtc_app).set({
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        })
        print("[WebRTC] Answer 전송 완료")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[WebRTC] 오류: {e}")

async def image_sender_loop():
    """이미지 전송 큐를 감시하며 Data Channel로 전송"""
    print("[Data Channel] 이미지 전송 루프 시작")
    while True:
        try:
            image_path = await asyncio.get_event_loop().run_in_executor(
                None, lambda: image_send_queue.get(timeout=1.0)
            )
            await send_image_via_data_channel(image_path)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Data Channel] 전송 오류: {e}")

async def send_image_via_data_channel(image_path: str):
    """이미지 파일을 읽어서 Data Channel로 전송"""
    global data_channel
    if data_channel is None or data_channel.readyState != "open":
        print("[Data Channel] 채널이 열려있지 않음")
        return

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()

        # 청크 단위로 전송 (64KB)
        chunk_size = 64 * 1024
        total_chunks = (len(image_data) + chunk_size - 1) // chunk_size

        # 시작 신호 전송
        data_channel.send(f"START:{total_chunks}")

        for i in range(total_chunks):
            chunk = image_data[i * chunk_size:(i + 1) * chunk_size]
            data_channel.send(chunk)

        # 완료 신호 전송
        data_channel.send("END")
        print(f"[Data Channel] 이미지 전송 완료: {image_path}")

    except Exception as e:
        print(f"[Data Channel] 이미지 전송 실패: {e}")

def on_offer_received(event):
    if event.data and isinstance(event.data, dict):
        offer_sdp = event.data.get('sdp')
        if offer_sdp:
            print("\n[감지] 새로운 Offer 수신")
            asyncio.run_coroutine_threadsafe(create_answer(offer_sdp), main_loop)

def start_webrtc_server():
    global main_loop, webrtc_app

    import firebase_admin
    from firebase_admin import credentials

    try:
        webrtc_app = firebase_admin.get_app('webrtc')
    except ValueError:
        cred = credentials.Certificate("/home/rafour/workspace/seungmin/rafour-app/serviceAccountKey.json")
        webrtc_app = firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://rafour-7f37f-default-rtdb.asia-southeast1.firebasedatabase.app/'
        }, name='webrtc')

    from firebase_admin import db as webrtc_db
    
    # 시작할 때 기존 signaling 데이터 초기화
    webrtc_db.reference('signaling/smart_cctv', app=webrtc_app).delete()
    print("[WebRTC] 기존 signaling 데이터 초기화 완료")

    main_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(main_loop)
    webrtc_db.reference('signaling/smart_cctv/offer', app=webrtc_app).listen(on_offer_received)
    print("[WebRTC] 시그널링 대기 중...")
    main_loop.run_forever()
