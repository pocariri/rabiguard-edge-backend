import cv2
from ultralytics import YOLO

def main():
    # 1. NCNN 모델 로드
    # 사용 중이던 yolo26n 모델 디렉토리를 지정합니다.
    print("모델 로드 중...")
    model = YOLO("./yolo26n_ncnn_model")

    # 2. USB 카메라 설정
    # 일반적으로 인덱스 0이 첫 번째 USB 카메라(/dev/video0)를 의미합니다.
    cap = cv2.VideoCapture(0)

    # 카메라 해상도 설정 (최적화를 위해 640x480 등 사용)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ 에러: USB 카메라를 찾거나 열 수 없습니다.")
        return

    # HDMI 디스플레이 출력을 위한 창 설정
    window_name = "HDMI Display - YOLO 실시간 객체 인식"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 전체 화면으로 띄우고 싶다면 아래 주석을 해제하세요.
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("🚀 실시간 카메라 추론을 시작합니다.")
    print("화면이 띄워진 상태에서 'q' 키를 누르면 종료됩니다.")

    try:
        while True:
            # USB 카메라에서 프레임 읽어오기
            ret, frame = cap.read()
            if not ret:
                print("❌ 프레임을 읽어올 수 없습니다.")
                break

            # 모델 추론 (YOLO NCNN)
            results = model(frame)

            # 결과(바운딩 박스)가 그려진 이미지 가져오기
            annotated_frame = results[0].plot()

            # HDMI 디스플레이에 화면 출력
            cv2.imshow(window_name, annotated_frame)

            # 1ms 대기하며 'q' 키가 입력되었는지 확인
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("종료 요청('q')을 받았습니다.")
                break

    except KeyboardInterrupt:
        print("\n사용자에 의해 강제로 종료되었습니다.")
    
    finally:
        # 3. 모든 자원 해제
        cap.release()
        cv2.destroyAllWindows()
        print("✅ 프로그램이 안전하게 종료되었습니다.")

if __name__ == "__main__":
    main()
