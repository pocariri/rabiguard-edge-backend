import asyncio
import json
import os
import sys
from pathlib import Path
from aiortc import RTCPeerConnection, RTCSessionDescription
from firebase_admin import credentials, db, initialize_app

# ------------------------------------------------------------
# Path settings for cross-module import
# ------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
RABIGUARD_DIR = ROOT_DIR / "rabiguard"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# config에서 통합 설정 임포트
try:
    from rabiguard.config import SNAPSHOT_DIR, FIREBASE_KEY_PATH
except ImportError:
    # 실행 위치에 따라 rabiguard가 패키지가 아닐 경우 처리
    if str(RABIGUARD_DIR) not in sys.path:
        sys.path.insert(0, str(RABIGUARD_DIR))
    from config import SNAPSHOT_DIR, FIREBASE_KEY_PATH

# ------------------------------------------------------------
# Firebase Initialization
# ------------------------------------------------------------
try:
    if not FIREBASE_KEY_PATH.exists():
        # 기본 경로에 없으면 백업 경로 시도 (라즈베리 파이 환경 대응)
        FIREBASE_KEY_PATH = Path("/home/rafour/workspace/seungmin/rafour-app/serviceAccountKey.json")

    cred = credentials.Certificate(str(FIREBASE_KEY_PATH))
    initialize_app(cred, {
        'databaseURL': 'https://rafour-7f37f-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })
    print(f"✅ [DataTransfer] Firebase 초기화 완료 (Key: {FIREBASE_KEY_PATH.name})")
except Exception as e:
    print(f"❌ [DataTransfer] Firebase 초기화 실패: {e}")
    sys.exit(1)

# ------------------------------------------------------------
# WebRTC & Transfer Logic
# ------------------------------------------------------------
pc = RTCPeerConnection()
data_channel = None

async def send_file(channel, file_path):
    """파일 하나를 조각내어 데이터 채널로 전송합니다."""
    filename = file_path.name
    file_bytes = file_path.read_bytes()
    filesize = len(file_bytes)
    
    # 1. 파일 시작 알림 (Metadata)
    channel.send(json.dumps({
        "type": "file_start",
        "filename": filename,
        "size": filesize
    }))
    
    # 2. 실제 데이터 전송 (16KB 단위 Chunking)
    chunk_size = 16384
    for i in range(0, filesize, chunk_size):
        chunk = file_bytes[i:i + chunk_size]
        channel.send(chunk)
        if channel.bufferedAmount > channel.bufferedAmountLowThreshold:
            await asyncio.sleep(0.01)
            
    # 3. 파일 종료 알림
    channel.send(json.dumps({"type": "file_end", "filename": filename}))
    print(f"📤 [DataTransfer] 전송 완료: {filename} ({filesize} bytes)")

async def start_transfer(event_id):
    """이벤트 폴더의 모든 이미지를 전송합니다."""
    global data_channel
    
    event_path = SNAPSHOT_DIR / event_id
    if not event_path.exists():
        print(f"❌ [DataTransfer] 이벤트 경로를 찾을 수 없음: {event_path}")
        return

    image_files = sorted(list(event_path.glob("*.jpg")))
    print(f"📦 [DataTransfer] {len(image_files)}개의 파일을 전송 준비 중 (Event: {event_id})")

    while data_channel is None or data_channel.readyState != "open":
        await asyncio.sleep(0.1)

    data_channel.send(json.dumps({
        "type": "transfer_start",
        "event_id": event_id,
        "total_files": len(image_files)
    }))

    for img_file in image_files:
        await send_file(data_channel, img_file)

    data_channel.send(json.dumps({"type": "transfer_end"}))
    print("✅ [DataTransfer] 모든 파일 전송 완료.")
    
    await asyncio.sleep(1)
    await pc.close()
    asyncio.get_event_loop().stop()

async def create_answer(offer_sdp, event_id):
    try:
        offer = RTCSessionDescription(sdp=offer_sdp, type='offer')
        await pc.setRemoteDescription(offer)
        
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        db.reference('signaling/smart_cctv/data_answer').set({
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        })
        print("📡 [DataTransfer] Answer 전송 완료. 클라이언트 연결 대기 중...")
        
        @pc.on("datachannel")
        def on_datachannel(channel):
            global data_channel
            data_channel = channel
            print(f"🔗 [DataTransfer] 데이터 채널 연결됨: {channel.label}")
            asyncio.create_task(start_transfer(event_id))

    except Exception as e:
        print(f"❌ [DataTransfer] Answer 생성 중 오류: {e}")

def on_offer_received(event, event_id):
    if event.data and isinstance(event.data, dict):
        offer_sdp = event.data.get('sdp')
        if offer_sdp:
            print("\n📩 [DataTransfer] 새로운 데이터 연결 요청(Offer) 수신")
            asyncio.run_coroutine_threadsafe(create_answer(offer_sdp, event_id), main_loop)

# ------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python webrtc_data_transfer.py <event_id>")
        sys.exit(1)
    
    target_event_id = sys.argv[1]
    print(f"🚀 [DataTransfer] 시작 - 대상 이벤트 ID: {target_event_id}")

    main_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(main_loop)

    db.reference('signaling/smart_cctv/data_offer').listen(
        lambda e: on_offer_received(e, target_event_id)
    )

    try:
        main_loop.run_forever()
    except KeyboardInterrupt:
        print("\n🛑 [DataTransfer] 중단됨")
    finally:
        main_loop.run_until_complete(pc.close())
