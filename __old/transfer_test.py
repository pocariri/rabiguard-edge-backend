import firebase_admin
from firebase_admin import credentials, db

# 1. 초기화 (싱가포르 서버 주소 적용)
cred = credentials.Certificate("/home/rafour/workspace/seungmin/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://rafour-7f37f-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# 2. 경로 참조 (Ref) 설정
ref_status = db.reference('status/smart_cctv')
ref_offer = db.reference('signaling/smart_cctv/offer')

# 3. 데이터 전송 테스트 함수
def update_cctv_status(count, gender, age):
    ref_status.update({
        'is_online': True,
        'total_count': count,
        'last_detected': {
            'gender': gender,
            'age_group': age
        }
    })
    print("성공적으로 smart_cctv 데이터를 업데이트했습니다.")

# 실행 테스트
update_cctv_status(1, "unknown", "child")
