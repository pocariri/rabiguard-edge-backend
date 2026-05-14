import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# 1. 카메라 설정
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# 2. NCNN 모델 로드
model = YOLO("./yolo26n_ncnn_model")

# 3. 동영상 저장 설정 (VideoWriter)
# 초당 10프레임(10.0), 640x480 해상도의 mp4 파일로 저장 준비
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./outputs/output_video.mp4', fourcc, 10.0, (640, 480))

print("녹화를 시작합니다! 카메라 앞을 지나가 보세요.")
print("종료하고 영상을 저장하려면 터미널에서 'Ctrl + C'를 누르세요.")

try:
    while True:
        # 프레임 가져오기
        frame = picam2.capture_array()

        # NCNN 모델 추론 (사람 등 객체 찾기)
        results = model(frame)

        # 바운딩 박스가 그려진 이미지 생성
        annotated_frame = results[0].plot()

        # 그려진 이미지를 동영상 파일에 한 장씩 기록
        out.write(annotated_frame)

except KeyboardInterrupt:
    # 사용자가 Ctrl+C를 눌렀을 때
    print("\n녹화를 중지합니다...")

finally:
    # 4. 자원 해제 및 파일 저장 완료
    picam2.stop()
    out.release() # 이게 실행되어야 동영상 파일이 정상적으로 완성됩니다.
    print("✅ 'output_video.mp4' 파일이 성공적으로 저장되었습니다!")
