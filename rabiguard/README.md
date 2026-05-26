# Arch1 Dynamic (Modular & Firestore-driven)

이 폴더는 기존의 단일 파일 기반 `arch1_headless.py`를 확장성 높은 모듈형 아키텍처로 리팩토링한 버전입니다. 
클라이언트(iOS 앱 등)에서 Firebase Firestore를 통해 구역(Zone)을 실시간으로 추가/수정/삭제하면, 디바이스를 재시작하지 않고도 즉시 감지 로직에 반영되도록 설계되었습니다.

## 📁 파일 구조 및 역할

- **`config.py`**
  - 스레드 안전 큐(`zone_config_queue`, `vlm_queue`) 생성 (Phase 1)
  - 글로벌 임계값 및 경로(NCNN 모델, VLM 캡처 이미지 저장 경로 등) 관리

- **`firestore_listener.py`**
  - Firestore의 `Zones` 컬렉션을 비동기적으로 구독(Subscribe)하는 백그라운드 스레드 (Phase 2)
  - 클라이언트의 변경 사항을 감지하여 표준화된 페이로드(Payload) 딕셔너리로 변환 후 큐에 푸시

- **`zone_manager.py`**
  - 개별 구역의 상태를 관리하는 `Zone` 클래스 정의 (Phase 3)
  - 다중 구역의 논리 연산을 담당하는 `ZoneManager` 클래스 정의
  - 매 프레임마다 YOLO 탐지 결과와 Depth 맵을 받아 일괄 진입(Entry) 검사 및 타이머 계산 수행
  - 이벤트 조건(명수, 시간) 충족 시 VLM 분석 큐로 전송 및 증적 이미지 저장

- **`main.py`**
  - GStreamer 파이프라인 (카메라 + Hailo NPU Depth) 제어
  - YOLO 워커 스레드 구동 및 NCNN 추론 (Phase 4)
  - VLM 워커 스레드 구동 (Hailo NPU 텍스트 생성)
  - 모든 모듈의 진입점(Entry Point)이자 하드웨어 자원 관리 총괄

## 🔄 데이터 흐름도 (Data Flow)

1. **[Firestore Listener]** `zone_config_queue`에 구역 변경 이벤트 푸시 (Non-blocking)
2. **[GStreamer Callback]** 카메라 영상 + 깊이 맵 추출 -> `yolo_queue` 푸시
3. **[YOLO Worker]** 
   - 큐에서 구역 설정 업데이트 (`zone_manager.process_queue_events`)
   - NCNN 기반 객체 탐지 실행
   - `zone_manager.check_zones` 호출
4. **[Zone Manager]** 구역 조건 검사 완료 시 -> `vlm_queue` 푸시 + 캡처 이미지 저장
5. **[VLM Worker]** Hailo NPU를 활용하여 상황 요약 생성 및 출력

## 🚀 실행 방법

가상환경을 활성화한 후, 프로젝트 루트에서 모듈 형태로 실행합니다.

```bash
# 가상환경 활성화
source .venv/bin/activate

# 모듈 형태로 실행
python -m vlm+yolo.arch1_dynamic.main
```

> **주의:** Firestore 실시간 구독을 위해서는 Firebase 서비스 계정 자격 증명이 필요합니다. 
> (`GOOGLE_APPLICATION_CREDENTIALS` 환경 변수를 설정하거나 `firestore_listener.py` 내의 초기화 코드를 수정하세요.)
