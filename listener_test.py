import firebase_admin
from firebase_admin import credentials, db

# 1. 초기화 (이미 되어있다면 중복 실행 주의)
cred = credentials.Certificate("/home/rafour/workspace/seungmin/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {'databaseURL': 'https://rafour-7f37f-default-rtdb.asia-southeast1.firebasedatabase.app/'})

# 2. 감시할 경로 설정 (시그널링 오퍼 경로)
offer_ref = db.reference('signaling/smart_cctv/offer')

# 3. 데이터 변화가 감지되었을 때 실행할 함수 (콜백)
def on_offer_received(event):
    # event.data에 Firebase에 새로 써진 내용이 담겨 옴
    if event.data is not None:
        print(f"\n[감지] 앱으로부터 메시지가 왔습니다: {event.data}")
        # 이제 여기에 WebRTC 응답(Answer) 로직을 연결하면 됨 
    else:
        print("\n[알림] 데이터가 삭제되었습니다.")

# 4. 리스너 시작
print("Firebase를 감시하며 대기 중입니다... (취소하려면 Ctrl+C)")
offer_ref.listen(on_offer_received)
