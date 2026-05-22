import os
import sys
import threading
import signal
from pathlib import Path

# GStreamer 하드웨어 디코더 플러그인 중 일부 픽셀 포맷과 충돌하는 모듈들을 비활성화
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE,v4l2slh265dec:NONE,v4l2slh264dec:NONE,v4l2h265dec:NONE,v4l2h264dec:NONE"
# Headless 환경(디스플레이가 없는 서버 등)에서 OpenCV GUI 관련 에러를 방지
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import numpy as np

# Hailo GStreamer Imports
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

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

# 전역 이벤트
stop_event = threading.Event()

# -----------------------------------------------------------------------
# Callback & Main
# -----------------------------------------------------------------------
class DepthAppCallback(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame_count = 0
        self.video_writer = None

def app_callback(element, buffer, user_data):
    try:
        if buffer is None:
            return

        user_data.frame_count += 1
        
        # 속도 조절이 필요하다면 프레임 스킵 (예: 2프레임당 1프레임 처리)
        # if user_data.frame_count % 2 != 0:
        #     return

        # 프레임 추출
        format, width, height = get_caps_from_pad(element.get_static_pad("sink"))
        frame_raw = get_numpy_from_buffer(buffer, format, width, height)
        
        if frame_raw is None:
            return
            
        # 프레임이 읽기 전용 버퍼일 수 있으므로 복사본 생성
        frame = frame_raw.copy()

        # 4채널(RGBA/BGRA)일 경우 3채널(RGB/BGR)로 변환
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        # 깊이 정보 추출 (SCDepthV3 결과)
        roi = hailo.get_roi_from_buffer(buffer)
        depth_objs = roi.get_objects_typed(hailo.HAILO_DEPTH_MASK)
        
        depth_map = None
        if len(depth_objs) > 0:
            depth_data = depth_objs[0].get_data()
            
            # scdepthv3 해상도는 기본적으로 320x256
            depth_array = np.array(depth_data)
            tensor_w, tensor_h = 320, 256
            if len(depth_array) == tensor_w * tensor_h:
                depth_map = depth_array.reshape((tensor_h, tensor_w))
                depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)
            else:
                depth_map = None

        # VideoWriter 초기화 (최초 1회)
        if user_data.video_writer is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "output_depth_only.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 입력 영상의 대략적인 FPS에 맞춤 (예: 30)
            user_data.video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
            print(f"🎬 비디오 저장 시작: {output_path}", flush=True)

        # Depth 맵이 있으면 원본 프레임과 반투명하게 합성
        if depth_map is not None:
            # Depth 값을 0~255 범위로 스케일링 (5.0m 기준 임의 설정)
            depth_vis = np.clip(depth_map / 5.0 * 255, 0, 255).astype(np.uint8)
            # 컬러맵 적용 (Jet: 빨간색이 가깝고, 파란색이 멂)
            depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            
            # 원본 이미지(frame)와 Depth 맵(depth_colormap)을 5:5 비율로 블렌딩
            frame = cv2.addWeighted(frame, 0.5, depth_colormap, 0.5, 0)
        else:
            # Depth 맵이 없을 때는 경고 메시지만 표시 (최초 프레임에서 발생 가능)
            if user_data.frame_count % 30 == 0:
                pass # 로그가 너무 많아지므로 주석 처리

        # OpenCV VideoWriter는 BGR 포맷을 요구하므로 변환
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if user_data.video_writer is not None and user_data.video_writer.isOpened():
            user_data.video_writer.write(frame_bgr)
            
    except Exception as e:
        import traceback
        print(f"❌ [app_callback Error] {e}", flush=True)
        traceback.print_exc()


def main():
    # GStreamer 하드웨어 디코더 플러그인 충돌 방지
    os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE,v4l2slh265dec:NONE,v4l2slh264dec:NONE,v4l2h265dec:NONE,v4l2h264dec:NONE"

    print("🚀 단일 실행 (SCDepthV3: NPU) - 비디오 기반 Depth 맵 시각화")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 비디오 파일 입력 설정
    video_path = os.path.join(project_root, "_inputs", "test_video_1.MP4")
    if not os.path.exists(video_path):
        print(f"❌ 입력 비디오 파일이 존재하지 않습니다: {video_path}")
        sys.exit(1)
        
    if "--input" not in sys.argv:
        sys.argv.extend(["--input", video_path])
        print(f"🎬 비디오 파일 모드로 실행합니다: {video_path}")
    
    user_data = DepthAppCallback()
    app = GStreamerDepthApp(app_callback, user_data)
    app.video_sink = "fakesink"
    
    # ----------------------------------------------------
    # 영상 반복 재생(루프) 방지 및 종료 처리
    # ----------------------------------------------------
    def custom_on_eos():
        print("✅ 영상 처리가 완료되었습니다 (End of Stream). 종료합니다.", flush=True)
        app.shutdown()
    
    app.on_eos = custom_on_eos
    
    # ----------------------------------------------------
    # GStreamer 파이프라인 종료 지연 전에 비디오 저장을 강제 완료하기 위한 훅
    # ----------------------------------------------------
    original_shutdown = app.shutdown
    def custom_shutdown(signum=None, frame=None):
        print("\n🛑 파이프라인 종료 중... 비디오 저장 객체를 해제합니다.", flush=True)
        if hasattr(user_data, 'video_writer') and user_data.video_writer is not None:
            user_data.video_writer.release()
            print("✅ 비디오 저장 완료: _outputs/output_depth_only.mp4", flush=True)
            user_data.video_writer = None
        else:
            print("⚠️ 저장할 비디오 객체가 없습니다.", flush=True)
        stop_event.set()
        original_shutdown(signum, frame)

    app.shutdown = custom_shutdown
    signal.signal(signal.SIGINT, custom_shutdown)
    
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(user_data, 'video_writer') and user_data.video_writer is not None:
            user_data.video_writer.release()
            print("✅ 비디오 저장 완료: _outputs/output_depth_only.mp4", flush=True)

if __name__ == "__main__":
    main()
