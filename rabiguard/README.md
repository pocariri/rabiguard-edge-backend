# RabiGuard (Modular & Firestore-driven)

이 폴더는 실시간 객체 탐지 및 다각형 기반 구역 감시를 수행하는 통합 엔진입니다. 
클라이언트(Expo 앱 등)에서 Firebase Firestore를 통해 모든 설정을 실시간으로 제어할 수 있도록 설계되었습니다.

## 🏗️ 핵심 아키텍처: Orchestrator

시스템의 모든 프로세스는 `orchestrator.py`에 의해 관리됩니다. 
카메라 자원의 상호 배제(Mutual Exclusivity)를 보장하며, 다음과 같은 전환 로직을 수행합니다.

- **감시 모드(`main.py`)**: YOLO + Depth + VLM을 활용한 상시 침입 탐지.
- **스트리밍 모드(`webrtc_video.py`)**: 실시간 영상 확인 시 감시 모드를 일시 중지하고 스트림 개시.
- **자동 설정 모드(`dynamic_roi_extractor.py`)**: 가구 위치를 분석하여 구역을 자동 생성.

## 📁 파일 구조 및 역할

- **`orchestrator.py` (Entry Point)**
  - Firestore `commands` 컬렉션을 감시하여 각 프로세스를 실행/종료 및 복구.
- **`main.py`**
  - GStreamer 파이프라인(카메라 + Depth) 및 YOLO/VLM 워커 구동.
- **`zone_manager.py`**
  - **다각형(Polygon)** 기반 구역 관리 및 마스크 기반 정밀 Depth 검증 수행.
- **`firestore_listener.py`**
  - Firestore의 `zones` 컬렉션을 구독하여 실시간으로 감시 구역 동기화.
- **`dynamic_roi_extractor.py`**
  - YOLOE-Seg 모델을 활용하여 실내 객체 외곽선을 따서 구역 자동 추천.
- **`config.py`**
  - 전역 임계값(Threshold), 경로, 스레드 안전 큐 관리.

## 🔄 데이터 흐름

1. **[Orchestrator]** 앱의 명령을 받아 `main.py` 실행.
2. **[Firestore Listener]** 사용자가 설정한 다각형 구역 정보를 실시간 로드.
3. **[YOLO Worker]** 사람을 탐지하고 `ZoneManager`에 판단 위임.
4. **[Zone Manager]** 다각형 내 체류 시간 및 바닥면 대비 Depth 차이 검증.
5. **[VLM Worker]** 조건 충족 시 깨끗한 크롭 이미지를 NPU로 분석하여 한국어 요약 생성.
6. **[Firebase Writer]** 분석 결과와 이미지를 Firestore에 저장 -> 앱 알림 전송.

## 🚀 실행 방법

가상환경 활성화 후 오케스트레이터를 실행하는 것이 권장됩니다.

```bash
source .venv/bin/activate
python3 rabiguard/orchestrator.py
```

> **주의:** Firebase 연동을 위해 프로젝트 루트에 `firebase_key.json` 파일이 반드시 존재해야 합니다.
