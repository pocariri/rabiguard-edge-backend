# Capstone Design (2026-1) - Rafour

## 실행 방법
### 1.가상환경 활성화

애플리케이션을 실행하기 전, 프로젝트 디렉토리(`rafour-app`)에서 가상환경을 먼저 활성화해야 합니다.
```bash
source .venv/bin/activate
```

### 2.모델 실행

가상환경이 활성화된 상태에서, 실행하고자 하는 기능에 맞춰 해당 경로로 이동한 뒤 스크립트를 실행합니다. 

**2-1. YOLO + VLM**

경로: `/rafour-app`

YOLO26n으로 사람 탐지 후 VLM에 연계합니다. Raspberry Pi와 연결된 디스플레이에 출력합니다.
```bash
DISPLAY=:0 python npu_vlm_camera_ncnn.py
```

**2-2. YOLOE(테스트)**

경로: `/rafour-app/yoloe_tests`

YOLOE (사전 없이 방대한 객체 탐지): 연결된 디스플레이에 출력합니다.
```bash
DISPLAY=:0 python run.py
```

**2-3. VLM(테스트)**

실행 경로: `/rafour-app/vlm`

이미지 경로: `/rafour-app/images`

test image를 읽어서 실행합니다.

```bash
python qwen_runner.py
```

## hailo-apps 관련 참고 사항

### python/standalone_apps

#### /yolo26
`object_detection_onnx_postproc.py` 실행 시 `-n` 옵션 붙이지 않으면 자동으로 YOLO26n 모델 다운로드 한다. *(resources_config.yaml 참조)*


## 참고 링크

### Hailo AI
- [Hailo Application Development Guide](https://github.com/hailo-ai/hailo-apps/blob/main/doc/developer_guide/app_development.md)
- [YOLO26](https://github.com/hailo-ai/hailo-apps/tree/main/hailo_apps/python/standalone_apps/yolo26/object_detection)

### Ultralytics
- [Quick Start Guide: Raspberry Pi with Ultralytics YOLO26](https://docs.ultralytics.com/guides/raspberry-pi/)