### Forked Repository Notice
본 저장소는 팀 프로젝트 당시 **[backend]** 파트 팀원의 작업물이며 전체 프로젝트 코드 참고 및 보존을 위해 Fork해 온 저장소입니다.

**저의 주 작업물은 아래 저장소에서 확인하실 수 있습니다.**

[capstone-rafour/frontend](https://github.com/Capstone-Rafour/rabiguard-mobile-app)

# Capstone Design (2026-1) - Rafour

라즈베리파이 5와 Hailo-10H NPU를 활용한 지능형 홈 케어 시스템입니다.

## 실행 방법

### 1. 가상환경 활성화

애플리케이션을 실행하기 전, 프로젝트 디렉토리(`rafour-app`)에서 가상환경을 먼저 활성화해야 합니다.
```bash
source .venv/bin/activate
```

### 2. 마스터 오케스트레이터 실행 (추천)

라즈베리파이의 모든 기능을 통합 관리하고 원격 제어할 수 있는 마스터 프로세스입니다. Firestore 명령 대기열을 통해 침입 탐지, 영상 스트리밍 등을 전 세계 어디서든 제어할 수 있습니다.

**준비 사항:**
- `firebase_key.json`: Firebase 서비스 계정 키 파일이 프로젝트 루트에 위치해야 합니다.

**실행:**
```bash
python3 rabiguard/orchestrator.py
```

### 3. 클라이언트 명령 테스트

아직 모바일 앱이 준비되지 않은 경우, 테스트 스크립트를 통해 오케스트레이터에 명령을 내릴 수 있습니다.
```bash
python3 rabiguard/test_orchestrator.py
```

---

## 주요 기능별 직접 실행 (개별 테스트용)

오케스트레이터를 거치지 않고 각 모듈을 독립적으로 테스트하고 싶을 때 사용합니다.

### 1. 메인 탐지 엔진 (RabiGuard)
Firestore와 연동되어 실시간 구역 감지 및 VLM 상황 분석을 수행합니다.
```bash
python -m rabiguard.main
```

### 2. 자동 구역 설정 (ROI Extractor)
카메라를 통해 실내 가구를 탐지하여 추천 감지 구역을 Firestore에 자동 등록합니다.
```bash
python -m rabiguard.dynamic_roi_extractor
```

### 3. WebRTC 실시간 스트리밍
카메라 영상을 WebRTC를 통해 원격으로 스트리밍합니다.
```bash
python webRTC/webrtc_video.py
```

## hailo-apps 관련 참고 사항

### python/standalone_apps
#### /yolo26
`object_detection_onnx_postproc.py` 실행 시 `-n` 옵션 붙이지 않으면 자동으로 YOLO26n 모델 다운로드 한다. *(resources_config.yaml 참조)*

## 참고 링크
- [Hailo AI Application Development Guide](https://github.com/hailo-ai/hailo-apps/blob/main/doc/developer_guide/app_development.md)
- [Ultralytics Raspberry Pi Guide](https://docs.ultralytics.com/guides/raspberry-pi/)
