# Capstone Design (2026-1) - Rafour

## 참고 링크

### Hailo AI
- [Hailo Application Development Guide](https://github.com/hailo-ai/hailo-apps/blob/main/doc/developer_guide/app_development.md)
- [YOLO26](https://github.com/hailo-ai/hailo-apps/tree/main/hailo_apps/python/standalone_apps/yolo26/object_detection)

### Ultralytics
- [Quick Start Guide: Raspberry Pi with Ultralytics YOLO26](https://docs.ultralytics.com/guides/raspberry-pi/)

# 실행 방법
```bash
// 1. 가상환경 활성화
source .venv/bin/activate

// 2. 가상환경 활성화 후,
// YOLO와 VLM 사용(YOLO로 사람 탐지 -> VLM): 연결된 디스플레이에 출력
// /rafour-app 에서 실행
DISPLAY=:0 python npu_vlm_camera_ncnn.py

// YOLOE (사전 없이 방대한 객체 탐지): 연결된 디스플레이에 출력
// /rafour-app/yoloe_tests 에서 실행
DISPLAY=:0 python run.py

// 3. VLM 실행 (test image 읽어서 실행)
// /rafour-app/vlm 에서 실행
// /rafour-app/images 에 입력 이미지 존재함
python qwen_runner.py

```
## 가상환경 활성화
`rafour-app`에서
```bash
source .venv/bin/activate
```

## hailo-apps
### python/standalone_apps
#### /yolo26
`object_detection_onnx_postproc.py` 실행 시 `-n` 옵션 붙이지 않으면 자동으로 YOLO26n 모델 다운로드 한다. *(resources_config.yaml 참조)*