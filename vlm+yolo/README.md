# VLM + YOLO (NPU & CPU Hybrid Architecture)

이 디렉토리는 Hailo NPU(SCDepthV3, VLM)와 CPU(YOLO26n NCNN)를 혼합하여 실시간으로 객체를 감지하고, 깊이(Depth)를 검증한 뒤, 이상 상황을 VLM으로 분석하는 아키텍처 구현체들을 포함하고 있습니다.

## 📂 파일 구성 및 설명

### 1. `arch1_headless.py` (최종 최적화 - Headless 운영용)
*   **방식**: 디스플레이 출력 없이(Fakesink) 백그라운드에서 실행되며, 최적화된 비동기 파이프라인을 통해 YOLO(CPU), SCDepthV3(NPU), VLM(NPU)을 동시에 구동합니다.
*   **특징**: 
    *   **지능적 프레임 드랍**: YOLO 연산이 바쁠 때 불필요한 데이터 복사를 건너뛰어 파이프라인 지연을 원천 차단합니다.
    *   **비동기 Worker 스레드**: YOLO와 VLM을 독립적인 스레드에서 실행하여 전체 시스템 반응성을 확보합니다.
    *   **최적의 성능**: 라즈베리파이 5 + Hailo-10H 환경에서 가장 안정적이고 빠른 처리가 가능합니다.
*   **실행**: `python3 arch1_headless.py`

### 2. `arch1_yolo_depth.py` (실시간 시각화 버전)
*   **방식**: 카메라 프레임에 깊이 맵(Depth Map)을 오버레이하여 보여주면서 실시간으로 YOLO 객체 탐지를 수행합니다.
*   **특징**: OpenCV를 통해 현재 분석 상황을 시각적으로 확인할 수 있어 개발 및 디버깅에 유리합니다.
*   **실행**: `python3 arch1_yolo_depth.py`

### 3. `npu_vlm_camera_ncnn.py` (VLM 기본 연동 버전)
*   **방식**: YOLO 객체 감지(CPU)와 VLM 모델(NPU)만을 연동한 기본적인 상황 인식 스크립트입니다.
*   **특징**: Depth 판별 로직 없이 객체 감지 시 즉시 VLM 분석을 수행하는 기초 파이프라인입니다.
*   **실행**: `python3 npu_vlm_camera_ncnn.py`

### 4. `depth_only_camera.py` (깊이 측정 테스트용)
*   **방식**: 오직 SCDepthV3 모델만을 사용하여 실시간으로 깊이 맵을 시각화합니다.
*   **특징**: NPU의 깊이 측정 성능과 정확도를 독립적으로 테스트할 때 사용합니다.
*   **실행**: `python3 depth_only_camera.py`

---

## 🛠️ 요구 사항 (Requirements)

1.  **Hailo-apps 및 Hailo 플랫폼**: 시스템 내에 Hailo NPU 환경(`hailo_platform`) 및 `hailo-apps` 패키지가 정상적으로 구성되어 있어야 합니다. (특히 `hailo-apps`가 홈 디렉토리에 위치해야 함)
2.  **YOLO NCNN 모델**: 상위 디렉토리에 `yolo26n_ncnn_model` 폴더가 존재해야 합니다.
3.  **Python 패키지**: `ultralytics`, `ncnn`, `opencv-python`, `numpy` 등이 설치되어 있어야 합니다.

## ⚙️ 로직 흐름 (arch1_headless 기준)

1.  **YOLO 탐지 (CPU)**: 사람(Class 0)을 실시간으로 감지하고 ROI(관심 구역) 내 진입 여부를 확인합니다.
2.  **시간 검증**: 해당 객체가 ROI 구역 내에 `2초` 이상 머무르는지(Tracking) 확인합니다.
3.  **Depth 검증 (NPU)**: SCDepthV3로 추출된 객체의 Depth와 ROI 구역의 평균 Depth를 비교하여, 오차가 `0.5m` 이내인지 확인하여 실제 구역 내 존재 여부를 입체적으로 판별합니다.
4.  **VLM 분석 (NPU)**: 모든 조건을 통과하면, 해당 프레임을 VLM(Vision Language Model)으로 넘겨 현재 상황을 요약 분석하여 알림을 생성합니다.

## 📝 참고 사항
*   `history_arch1_headless.md`: `arch1_headless.py`의 최적화 과정과 주요 버그 수정 기록이 담겨 있습니다.
*   GStreamer 1.26.2 버전의 버그(StructureWrapper)에 대한 패치가 코드 내에 포함되어 있습니다.
