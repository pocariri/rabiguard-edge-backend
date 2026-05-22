import asyncio
import json
from aiortc import RTCPeerConnection, RTCSessionDescription
from firebase_admin import credentials, db, initialize_app

# 1. Firebase 초기화 (이미 설정된 경로 사용)
cred = credentials.Certificate("/home/rafour/workspace/seungmin/serviceAccountKey.json")
initialize_app(cred, {'databaseURL': 'https://rafour-7f37f-default-rtdb.asia-southeast1.firebasedatabase.app/'})

pc = RTCPeerConnection()

# 앱으로부터 Offer를 받았을 때 실행되는 함수
async def create_answer(offer_sdp):
    try:
        # 1. 받은 Offer 설정
        offer = RTCSessionDescription(sdp=offer_sdp, type='offer')
        await pc.setRemoteDescription(offer)
        print("Remote Description (Offer) 설정 완료")

        # 2. Answer 생성
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        print("Local Description (Answer) 생성 완료")

        # 3. Firebase에 Answer 업로드
        db.reference('signaling/smart_cctv/answer').set({
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        })
        print("Firebase에 Answer 전송 완료! 이제 앱과 연결을 시도합니다.")

    except Exception as e:
        print(f"Answer 생성 중 오류 발생: {e}")

def on_offer_received(event):
    if event.data and isinstance(event.data, dict):
        offer_sdp = event.data.get('sdp')
        if offer_sdp:
            print("\n[감지] 새로운 연결 요청(Offer)이 왔습니다.")
	    # 현재 실행 중인 루프를 안전하게 가져와서 작업을 넘깁니다.
            asyncio.run_coroutine_threadsafe(create_answer(offer_sdp), main_loop)

# 메인 실행부
print("WebRTC 시그널링 대기 중... (Ctrl+C로 종료)")

# 1. 메인 루프를 미리 생성합니다.
main_loop = asyncio.new_event_loop()
asyncio.set_event_loop(main_loop)

# 2. 리스너 등록
db.reference('signaling/smart_cctv/offer').listen(on_offer_received)

# 3. 루프 실행
try:
    main_loop.run_forever()
except KeyboardInterrupt:
    print("\n정지합니다...")
finally:
    main_loop.run_until_complete(pc.close())
