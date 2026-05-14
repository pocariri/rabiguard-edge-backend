from ultralytics import YOLO

# (PC에서 가져온) NCNN 모델 폴더의 경로를 지정하여 바로 로드
ncnn_model = YOLO("./yolo26n_ncnn_model")

# 곧바로 추론(Inference) 실행
results = ncnn_model("./bus.jpg")  # 이미지 파일 경로를 지정하여 추론 실행

# 결과 확인
results[0].save("./outputs/result_bus.jpg")

print("결과 이미지가 성공적으로 저장되었습니다!")