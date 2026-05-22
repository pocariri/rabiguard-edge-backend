import os
import sys
import time
from pathlib import Path

# GStreamer 하드웨어 디코더 플러그인 중 일부 픽셀 포맷과 충돌하는 모듈들을 비활성화
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE,v4l2slh265dec:NONE,v4l2slh264dec:NONE,v4l2h265dec:NONE,v4l2h264dec:NONE"

import cv2
import numpy as np

# Hailo GStreamer Imports
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

# 프로젝트 루트 및 hailo-apps 경로 설정
hailo_apps_dir = (Path.home() / "hailo-apps").resolve()

if str(hailo_apps_dir) not in sys.path:
    sys.path.insert(0, str(hailo_apps_dir))

try:
    from hailo_apps.python.pipeline_apps.depth.depth_pipeline import GStreamerDepthApp
    from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
    from hailo_apps.python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
    import hailo
except ImportError as e:
    print(f"❌ Hailo 라이브러리 임포트 실패: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------
# Callback Class & Function
# -----------------------------------------------------------------------
class DepthAppCallback(app_callback_class):
    def __init__(self):
        super().__init__()
        self.last_proc_time = 0
        self.fps_interval = 1.0 / 20.0  # 최대 20FPS로 처리 제한 (지연 방지)

def app_callback(element, buffer, user_data):
    try:
        if buffer is None:
            return

        # 1. 프레임 스킵 로직: 처리 속도가 너무 빠르면 건너뛰어 큐 병목 방지
        curr_time = time.time()
        if curr_time - user_data.last_proc_time < user_data.fps_interval:
            return
        user_data.last_proc_time = curr_time

        # GStreamer 버퍼에서 프레임 정보 추출
        format, width, height = get_caps_from_pad(element.get_static_pad("sink"))
        frame_raw = get_numpy_from_buffer(buffer, format, width, height)
        
        if frame_raw is None:
            return
            
        # 프레임 복사 (GStreamer 버퍼는 읽기 전용일 수 있음)
        frame = frame_raw.copy()

        # RGB 포맷으로 변환
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        # 깊이(Depth) 텐서 데이터 추출
        roi = hailo.get_roi_from_buffer(buffer)
        depth_objs = roi.get_objects_typed(hailo.HAILO_DEPTH_MASK)
        
        if len(depth_objs) > 0:
            depth_data = depth_objs[0].get_data()
            
            # SCDepthV3 모델 해상도 (320x256)
            tensor_w, tensor_h = 320, 256
            depth_array = np.array(depth_data)
            
            if len(depth_array) == tensor_w * tensor_h:
                depth_map = depth_array.reshape((tensor_h, tensor_w))
                
                # 시각화 연산 최적화: 결과물을 작은 크기에서 먼저 처리 후 확대
                depth_vis = np.clip(depth_map / 5.0 * 255, 0, 255).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                
                # 연산 부하를 줄이기 위해 원본 크기에 맞춰 리사이즈
                depth_colormap_resized = cv2.resize(depth_colormap, (width, height), interpolation=cv2.INTER_LINEAR)
                
                # 합성 (5:5 비율)
                frame = cv2.addWeighted(frame, 0.5, depth_colormap_resized, 0.5, 0)

        # OpenCV 출력을 위해 BGR로 변환
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # GStreamerApp의 큐에 프레임 전달 (use-frame 옵션)
        user_data.set_frame(frame_bgr)
            
    except Exception as e:
        print(f"❌ [app_callback Error] {e}")

def main():
    print("🚀 Hailo 실시간 Depth 인식 최적화 모드 (Camera)")
    print("💡 종료하려면 Ctrl+C를 한 번만 누르세요.")

    # 2. 성능 최적화를 위한 인자 설정
    # --width, --height: 입력 해상도를 낮추어 처리량 감소 (640x480 권장)
    if "--input" not in sys.argv:
        sys.argv.extend(["--input", "usb"])
    if "--use-frame" not in sys.argv:
        sys.argv.append("--use-frame")
    if "--width" not in sys.argv:
        sys.argv.extend(["--width", "640"])
    if "--height" not in sys.argv:
        sys.argv.extend(["--height", "480"])
    
    user_data = DepthAppCallback()
    app = GStreamerDepthApp(app_callback, user_data)
    
    # GStreamer 자체 싱크 대신 OpenCV 윈도우 사용
    app.video_sink = "fakesink"
    
    # 3. 중복 핸들러 제거: GStreamerApp 내부 핸들러가 안전 종료를 처리하도록 함

    try:
        app.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
