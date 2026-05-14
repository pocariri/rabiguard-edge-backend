# Capstone Design (2026-1) - Rafour

## 참고 링크

### Hailo AI
- [Hailo Application Development Guide](https://github.com/hailo-ai/hailo-apps/blob/main/doc/developer_guide/app_development.md)
- [YOLO26](https://github.com/hailo-ai/hailo-apps/tree/main/hailo_apps/python/standalone_apps/yolo26/object_detection)

### Ultralytics
- [Quick Start Guide: Raspberry Pi with Ultralytics YOLO26](https://docs.ultralytics.com/guides/raspberry-pi/)


## 가상환경 활성화
`rafour-app`에서
```bash
source .venv/bin/activate
```

## hailo-apps
### python/standalone_apps
#### /yolo26
`object_detection_onnx_postproc.py` 실행 시 `-n` 옵션 붙이지 않으면 자동으로 YOLO26n 모델 다운로드 한다. *(resources_config.yaml 참조)*