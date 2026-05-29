# Capstone Design (2026-1) - Rafour

## 실행 방법

### 1. 가상환경 활성화

애플리케이션을 실행하기 전, 프로젝트 디렉토리(`rafour-app`)에서 가상환경을 먼저 활성화해야 합니다.
```bash
source .venv/bin/activate
```

### 2. 메인 애플리케이션 실행 (RabiGuard)

Firestore와 연동되어 실시간 구역 감지 및 VLM 상황 분석을 수행하는 통합 엔진입니다.

**준비 사항:**
- `firebase_key.json`: Firebase 서비스 계정 키 파일이 프로젝트 루트에 위치해야 합니다.

**실행:**
```bash
# 디스플레이 출력이 필요한 경우 (GStreamer 윈도우 표시)
DISPLAY=:0 python -m rabiguard.main

# 터미널에서만 실행 (Headless)
python -m rabiguard.main
```

### 3. 자동 구역 설정 (ROI Extractor)

카메라를 통해 실내 환경의 주요 객체(침대, 의자 등)를 탐지하여 추천 감지 구역을 Firestore에 자동으로 등록합니다.

```bash
python -m rabiguard.dynamic_roi_extractor
```

### 4. WebRTC 실시간 스트리밍 (모니터링)

카메라 영상을 WebRTC를 통해 원격으로 스트리밍합니다.

```bash
python webRTC/webrtc_video.py
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