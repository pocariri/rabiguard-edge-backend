# YOLOE Auto ROI Tests

[YOLOE: Real-Time Seeing Anything](https://docs.ultralytics.com/ko/models/yoloe#textvisual-prompt-models)

이 폴더는 **YOLOE Prompt-Free (PF) 분할 모델**을 사용하여 사용자의 환경에 맞는 **관심 영역(ROI, Region of Interest)**을 지능적으로 감지하고 자동 제안(Proposal)하기 위한 테스트 및 스크립트들을 포함하고 있습니다.

단순히 화면에서 가장 큰 객체를 잡는 방식이 아니라, 카메라가 설치된 장소(예: 스마트 홈, 매장, 공장 등)의 특성에 맞춰 감시하기 적합한 '고정형 사물'만을 필터링하여 사용자에게 제안하도록 고도화되었습니다.

## 📂 주요 폴더 및 파일 설명

### 🧠 모델 및 테스트 스크립트
* **`yoloe-26n-seg-pf.pt`**: 실시간 세그멘테이션 및 객체 인식을 수행하는 YOLOe Prompt-Free 경량 모델입니다.
* **`run.py`**: 모델이 실시간 웹캠(또는 이미지)에서 정상 작동하는지 확인하는 기본 테스트 스크립트입니다.
* **`auto_roi.py`**: 실시간 카메라 화면에서 가장 눈에 띄는(Salient) 하나의 객체를 찾아 초록색 다각형으로 표시하고, 단축키(`s`)를 통해 해당 다각형의 좌표를 `roi_config.json`으로 저장하는 스크립트입니다. (클래스 필터링 없음)
* **`extract_roi_candidates.py`**: ⭐️ **핵심 스크립트**. 특정 이미지와 환경 번호(`--env 1~5`)를 입력받아 해당 환경에 적합한 객체들만 찾아냅니다. 감지된 각 객체의 크롭 이미지와 정밀한 폴리곤 좌표, 바운딩 박스 정보를 `proposals/` 폴더 내에 이미지들과 `proposals.json` 파일로 저장하여 사용자에게 제안할 후보군을 생성합니다.

### 📋 환경별 ROI 객체 목록 (`roi_obj_lists/`)
감시 구역으로 삼기 부적합한 작거나 이동성이 큰 객체(컵, 장난감, 동물 등)는 제외하고, 고정적이고 확실한 구역 역할을 할 수 있는 사물(문, 창문, 소파, 테이블, 계산대 등)들의 이름이 담긴 텍스트 파일들입니다. 
* `retail_and_convenience_store.txt` (무인 매장/편의점)
* `food_and_beverage.txt` (카페/식당)
* `smart_home_and_elderly_care.txt` (스마트 홈/실버 케어)
* `industrial_and_logistics.txt` (산업 현장/물류 창고)
* `education_and_daycare.txt` (학교/유치원)
* `ram_tag_list.txt`: 모델이 인식할 수 있는 전체 RAM 태그 원본 목록입니다.

### 🛠️ 유틸리티 스크립트
* **`categorize_tags.py`**: 원본 RAM 태그 목록에서 환경별로 객체를 1차 분류한 스크립트입니다.
* **`temp/` 폴더**: 태그 목록을 정제(작은 객체 제외, 공통 객체 추가 등)하기 위해 사용된 일회성 스크립트들(`add_common_tags.py`, `remove_small_tags.py`, `update_lists.py`)이 백업되어 있습니다.

## 🚀 사용 방법 (ROI 후보 추출)

터미널에서 분석할 이미지 경로와 환경 번호(`--env`)를 입력하여 실행합니다.

```bash
# 예시: 스마트 홈 환경(3번) 기준으로 room.jpg 이미지를 분석
python extract_roi_candidates.py path/to/your/image.jpg --env 3
```

**환경 번호 안내:**
* `1`: 🏪 무인 매장 및 스마트 편의점
* `2`: ☕ 카페 및 식당
* `3`: 🏠 스마트 홈 및 실버 케어
* `4`: 🏭 산업 현장 및 물류 창고
* `5`: 🏫 학교 및 유치원/어린이집

실행 완료 후, 이 스크립트와 같은 경로의 `proposals/` 폴더에 제안할 객체들의 이미지 크롭본과 상세 좌표가 담긴 `proposals.json`이 생성됩니다.
