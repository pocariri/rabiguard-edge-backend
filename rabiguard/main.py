# main.py

import os
import sys
import time
import queue
import threading
import uuid
from pathlib import Path
from collections import deque

import cv2
import numpy as np

# ------------------------------------------------------------
# Environment settings
# ------------------------------------------------------------

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["GST_PLUGIN_FEATURE_RANK"] = (
    "vaapidecodebin:NONE,"
    "v4l2slh265dec:NONE,"
    "v4l2slh264dec:NONE,"
    "v4l2h265dec:NONE,"
    "v4l2h264dec:NONE"
)
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.fonts=false"

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# ------------------------------------------------------------
# Local modules
# ------------------------------------------------------------

try:
    from .config import (
        zone_config_queue, vlm_queue, stop_event, MODEL_PATH,
        SNAPSHOT_DIR, SNAPSHOT_FPS, SNAPSHOT_BEFORE_SEC, SNAPSHOT_AFTER_SEC
    )
    from .zone_manager import ZoneManager
    from .firestore_listener import start_firestore_listener
    from .prompts import SYSTEM_PROMPT, USER_PROMPT
    from .translator import translate_to_korean
    from .firebase_writer import save_vlm_result_to_firestore

except ImportError:
    from config import (
        zone_config_queue, vlm_queue, stop_event, MODEL_PATH,
        SNAPSHOT_DIR, SNAPSHOT_FPS, SNAPSHOT_BEFORE_SEC, SNAPSHOT_AFTER_SEC
    )
    from zone_manager import ZoneManager
    from firestore_listener import start_firestore_listener
    from prompts import SYSTEM_PROMPT, USER_PROMPT
    from translator import translate_to_korean
    from firebase_writer import save_vlm_result_to_firestore

# ------------------------------------------------------------
# Snapshot Buffer
# ------------------------------------------------------------

snapshot_ring_buffer = deque(maxlen=SNAPSHOT_BEFORE_SEC * SNAPSHOT_FPS)
snapshot_lock = threading.Lock()

def snapshot_recorder_worker():
    pending_snapshots = queue.Queue()

    def saver_thread():
        while not stop_event.is_set():
            try:
                event_data = pending_snapshots.get(timeout=1.0)
            except queue.Empty:
                continue

            event_id = event_data["event_id"]
            before_frames = event_data["before_frames"]

            after_frames = []
            start_wait = time.time()
            frame_interval = 1.0 / SNAPSHOT_FPS

            print(f"📸 [Snapshot] Event {event_id} - 'After' 수집 시작...")

            while time.time() - start_wait < SNAPSHOT_AFTER_SEC:
                if stop_event.is_set():
                    break
                with snapshot_lock:
                    if snapshot_ring_buffer:
                        after_frames.append(snapshot_ring_buffer[-1])
                time.sleep(frame_interval)

            event_dir = SNAPSHOT_DIR / event_id
            event_dir.mkdir(parents=True, exist_ok=True)

            total_frames = list(before_frames) + after_frames
            print(f"💾 [Snapshot] Event {event_id} - 총 {len(total_frames)}장 저장 중...")

            for i, frame in enumerate(total_frames):
                filename = event_dir / f"frame_{i:03d}.jpg"
                cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

            print(f"✅ [Snapshot] Event {event_id} 저장 완료: {event_dir}")
            pending_snapshots.task_done()

    t = threading.Thread(target=saver_thread, daemon=True)
    t.start()
    return pending_snapshots

snapshot_event_queue = None

# ------------------------------------------------------------
# YOLO
# ------------------------------------------------------------

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics 패키지가 필요합니다.")
    sys.exit(1)

# ------------------------------------------------------------
# GStreamer / Hailo
# ------------------------------------------------------------

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    import hailo

    hailo_apps_dir = (Path.home() / "hailo-apps").resolve()
    if str(hailo_apps_dir) not in sys.path:
        sys.path.insert(0, str(hailo_apps_dir))

    from hailo_apps.python.pipeline_apps.depth.depth_pipeline import GStreamerDepthApp
    from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
    from hailo_apps.python.core.common.buffer_utils import get_numpy_from_buffer
    from hailo_apps.python.core.common.core import resolve_hef_path
    from hailo_apps.python.core.common.defines import (
        VLM_CHAT_APP,
        SHARED_VDEVICE_GROUP_ID,
        HAILO10H_ARCH,
    )
    from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
        INFERENCE_PIPELINE,
        INFERENCE_PIPELINE_WRAPPER,
        USER_CALLBACK_PIPELINE,
    )

    from hailo_platform import VDevice
    from hailo_platform.genai import VLM

except ImportError as e:
    print(f"❌ Hailo/GStreamer 라이브러리 임포트 실패: {e}")
    sys.exit(1)

# ------------------------------------------------------------
# Local modules
# ------------------------------------------------------------

try:
    from .config import zone_config_queue, vlm_queue, stop_event, MODEL_PATH
    from .zone_manager import ZoneManager
    from .firestore_listener import start_firestore_listener
    from .prompts import SYSTEM_PROMPT, USER_PROMPT
    from .translator import translate_to_korean
    from .firebase_writer import save_vlm_result_to_firestore
    from .postprocess import clean_vlm_caption

except ImportError:
    from config import zone_config_queue, vlm_queue, stop_event, MODEL_PATH
    from zone_manager import ZoneManager
    from firestore_listener import start_firestore_listener
    from prompts import SYSTEM_PROMPT, USER_PROMPT
    from translator import translate_to_korean
    from firebase_writer import save_vlm_result_to_firestore
    from postprocess import clean_vlm_caption


# ------------------------------------------------------------
# GStreamer caps compatibility fix
# ------------------------------------------------------------

def get_caps_from_pad_fixed(pad):
    if pad is None:
        return None, None, None
    caps = pad.get_current_caps()
    if not caps:
        return None, None, None
    structure = caps.get_structure(0)
    if not structure:
        return None, None, None
    real_structure = getattr(structure, "_StructureWrapper__structure", structure)
    try:
        fmt = real_structure.get_value("format")
        width = real_structure.get_value("width")
        height = real_structure.get_value("height")
        return fmt, width, height
    except AttributeError:
        return None, None, None

# ------------------------------------------------------------
# Headless Depth App
# ------------------------------------------------------------

class HeadlessDepthApp(GStreamerDepthApp):
    def get_pipeline_string(self):
        source_pipeline = self.get_source_pipeline()
        depth_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            name="depth_inference",
        )
        depth_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(
            depth_pipeline,
            bypass_max_size_buffers=2,
            name="inference_wrapper_depth",
        )
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        return (
            f"{source_pipeline} ! "
            f"{depth_pipeline_wrapper} ! "
            f"{user_callback_pipeline} ! "
            f"fakesink sync=false"
        )

# ------------------------------------------------------------
# VLM utility
# ------------------------------------------------------------

def clean_response(response: str) -> str:
    if not response:
        return ""
    text = str(response)
    if ".\n[{'type'" in text:
        text = text.split(".\n[{'type'")[0]
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0]
    return text.strip()

# ------------------------------------------------------------
# VLM Worker
# ------------------------------------------------------------

def vlm_worker_thread(collection_name="vlm_events"):
<<<<<<< HEAD
=======
    """
    ZoneManager에서 전달한 이벤트 이미지를 받아
    VLM 분석 → 영어 결과 정리 → 한국어 번역 → Firestore 저장까지 수행합니다.

    VLM 생성 실패 시에도 fallback 문장을 사용하여 Firestore에 이벤트를 저장합니다.
    """
>>>>>>> 0c13580ee990fd29dc20b197b779bdde9c0dcee0
    print("🔵 [VLM Worker] 초기화 시작...")

    hef_path = resolve_hef_path(
        None,
        app_name=VLM_CHAT_APP,
        arch=HAILO10H_ARCH,
    )

    if not hef_path:
        print("❌ [VLM Worker] VLM HEF 모델을 찾을 수 없습니다.")
        return

    print(f"✅ [VLM Worker] HEF 사용: {hef_path}")

    vdevice = None
    vlm = None

    try:
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        vdevice = VDevice(params)
        vlm = VLM(vdevice, str(hef_path))
        print("✅ [VLM Worker] VLM 초기화 완료!")

        while not stop_event.is_set():
            try:
                item = vlm_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break

            try:
                context_img = item["image"]
                track_id = item["track_id"]
                p_depth = item["p_depth"]
                z_depth = item["z_depth"]
                zone_id = item["zone_id"]
<<<<<<< HEAD
                image_path = item.get("image_path", f"zone_{zone_id}_track_{track_id}")
                people_count = item.get("people_count", 1)
                enter_threshold_sec = item.get("enter_threshold_sec", 0)
                event_id = item.get("event_id", "")

                vlm_img = cv2.resize(context_img, (336, 336), interpolation=cv2.INTER_LINEAR)
=======

                image_path = item.get(
                    "image_path",
                    f"zone_{zone_id}_track_{track_id}",
                )

                # VLM 입력 이미지 전처리
                vlm_img = cv2.resize(
                    context_img,
                    (336, 336),
                    interpolation=cv2.INTER_LINEAR,
                )

>>>>>>> 0c13580ee990fd29dc20b197b779bdde9c0dcee0
                if len(vlm_img.shape) == 3 and vlm_img.shape[2] == 3:
                    vlm_img = cv2.cvtColor(vlm_img, cv2.COLOR_BGR2RGB)
                vlm_img = vlm_img.astype(np.uint8)

                # prompts.py의 SYSTEM_PROMPT, USER_PROMPT만 사용
                # zone_id, depth, 체류시간 같은 메타정보는 VLM 프롬프트에 넣지 않음
                prompt = [
<<<<<<< HEAD
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": event_prompt}]},
=======
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": SYSTEM_PROMPT,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": USER_PROMPT,
                            },
                        ],
                    },
>>>>>>> 0c13580ee990fd29dc20b197b779bdde9c0dcee0
                ]

                print(f"\n🔎 [VLM] Zone '{zone_id}' -> ID {track_id} 분석 중...")

                status = "completed"

                try:
                    response = vlm.generate_all(
                        prompt=prompt,
                        frames=[vlm_img],
                        temperature=0.1,
                        seed=42,
                        max_generated_tokens=30,
                    )

                    english_result = clean_response(response)
                    english_result = clean_vlm_caption(english_result)

                    if not english_result:
                        raise ValueError("Empty VLM output")

                    korean_result = translate_to_korean(english_result)

                    if not korean_result:
                        korean_result = "감시 구역에서 사람이 감지되었습니다."

                except Exception as e:
                    print(f"⚠️ [VLM Generate Error] {e}")

                    english_result = "A person was detected in the monitored zone."
                    korean_result = "감시 구역에서 사람이 감지되었습니다."
                    status = "vlm_failed"

                print("=" * 70)
                print("🚨 [VLM 상황 요약 알림]")
                print(f"Zone: {zone_id}")
                print(f"객체 ID: {track_id}")
                print(f"Status: {status}")
                print(f"Depth: person={p_depth:.2f}m, zone={z_depth:.2f}m")
                print(f"Image: {image_path}")
                print(f"EN: {english_result}")
                print(f"KO: {korean_result}")
                print("=" * 70)

                try:
                    doc_id = save_vlm_result_to_firestore(
                        english_text=english_result,
                        korean_text=korean_result,
                        image_path=str(image_path),
                        collection_name=collection_name,
                        zone_id=str(zone_id),
                        track_id=track_id,
                        person_depth=float(p_depth),
                        zone_depth=float(z_depth),
<<<<<<< HEAD
                        event_id=event_id,
=======
                        status=status,
>>>>>>> 0c13580ee990fd29dc20b197b779bdde9c0dcee0
                    )
                    print(f"✅ [Firestore] 저장 완료. Document ID: {doc_id}")

                except Exception as e:
                    print(f"⚠️ [Firestore Save Error] {e}")

            except Exception as e:
                print(f"⚠️ [VLM Error] {e}")

            finally:
                vlm_queue.task_done()

    except Exception as e:
        print(f"❌ [VLM Worker] {e}")

    finally:
        if vlm:
            try:
                vlm.clear_context()
            except Exception:
                pass
            try:
                vlm.release()
            except Exception as e:
                print(f"⚠️ [VLM release warning] {e}")
        if vdevice:
            try:
                vdevice.release()
            except Exception as e:
                print(f"⚠️ [VDevice release warning] {e}")
        print("🔴 [VLM Worker] 종료됨.")

# ------------------------------------------------------------
# Dynamic App Callback
# ------------------------------------------------------------

class DynamicAppCallback(app_callback_class):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.total_frames = 0
        self.status_start_time = time.time()
        self.status_frame_count = 0
        self.caps_info = None
        self.yolo_queue = queue.Queue(maxsize=1)
        self.yolo_ready = True
        self.snapshot_event_queue = None
        self.yolo_thread = threading.Thread(target=self.yolo_worker, daemon=True)
        self.yolo_thread.start()

    def yolo_worker(self):
        print("✅ [YOLO Worker] 시작 (Dynamic Mode)!")
        zone_manager = ZoneManager()

        # snapshot_event_queue가 설정될 때까지 대기
        while self.snapshot_event_queue is None:
            time.sleep(0.1)
        
        zone_manager.snapshot_event_queue = self.snapshot_event_queue
        last_yolo_time = time.time()
        yolo_count = 0

<<<<<<< HEAD
        dummy_payload = {
            "action": "update",
            "zone_id": "Zone_A1",
            "data": {
                "polygon": [[100, 100], [540, 100], [540, 380], [100, 380]],
                "enter_threshold_sec": 2.0,
                "min_people": 1,
                "is_active": True,
            },
        }
        zone_config_queue.put(dummy_payload)
=======
        # 개발/테스트용 기본 구역 주석 처리 (Firestore 데이터만 사용)
        # dummy_payload = {
        #     "action": "update",
        #     "zone_id": "Zone_A1",
        #     "data": {
        #         "polygon": [
        #             [100, 100],
        #             [540, 100],
        #             [540, 380],
        #             [100, 380],
        #         ],
        #         "enter_threshold_sec": 2.0,
        #         "min_people": 1,
        #         "is_active": True,
        #     },
        # }
        # zone_config_queue.put(dummy_payload)
>>>>>>> 0c13580ee990fd29dc20b197b779bdde9c0dcee0

        while not stop_event.is_set():
            self.yolo_ready = True

            while not zone_config_queue.empty():
                try:
                    payload = zone_config_queue.get_nowait()
                    zone_manager.process_queue_events(payload)
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"⚠️ [Zone Config Error] {e}")

            try:
                data = self.yolo_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if data is None:
                break

            frame_raw, depth_raw, fmt = data

            try:
                h_orig, w_orig = frame_raw.shape[:2]
                frame_small = cv2.resize(frame_raw, (320, 320), interpolation=cv2.INTER_LINEAR)

                if fmt == "RGB":
                    color_conv = cv2.COLOR_RGB2BGR
                elif fmt == "RGBA":
                    color_conv = cv2.COLOR_RGBA2BGR
                else:
                    color_conv = cv2.COLOR_RGB2BGR

                frame_small_bgr = cv2.cvtColor(frame_small, color_conv)

                depth_map = None
                if depth_raw is not None:
                    depth_arr = np.array(depth_raw, dtype=np.float32).flatten()
                    if depth_arr.size == 256 * 320:
                        depth_map = depth_arr.reshape((256, 320))
                    else:
                        print(f"⚠️ [DEPTH] 예상과 다른 depth size: {depth_arr.size}")

                results = self.model.track(
                    frame_small_bgr,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    verbose=False,
                )

                yolo_count += 1
                now = time.time()
                if now - last_yolo_time >= 2.0:
                    print(f"📊 [YOLO SPEED] {yolo_count / (now - last_yolo_time):.1f} FPS")
                    yolo_count = 0
                    last_yolo_time = now

                zone_manager.check_zones(
                    results=results,
                    depth_map=depth_map,
                    frame_raw=frame_raw,
                    color_conv=color_conv,
                    w_orig=w_orig,
                    h_orig=h_orig,
                )

            except Exception as e:
                print(f"❌ [YOLO Error] {e}")

            finally:
                self.yolo_queue.task_done()

# ------------------------------------------------------------
# GStreamer callback
# ------------------------------------------------------------

def app_callback(element, buffer, user_data):
    try:
        if buffer is None:
            return

        curr_time = time.time()

        # Snapshot Ring Buffer 업데이트
        frame_interval = 1.0 / SNAPSHOT_FPS
        if not hasattr(app_callback, "last_snapshot_time"):
            app_callback.last_snapshot_time = 0

        if curr_time - app_callback.last_snapshot_time >= frame_interval:
            if user_data.caps_info is None:
                user_data.caps_info = get_caps_from_pad_fixed(element.get_static_pad("sink"))
            fmt, w, h = user_data.caps_info
            if fmt and w and h:
                frame_raw = get_numpy_from_buffer(buffer, fmt, w, h)
                if frame_raw is not None:
                    frame_bgr = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
                    with snapshot_lock:
                        snapshot_ring_buffer.append(frame_bgr)
                    app_callback.last_snapshot_time = curr_time

        if not user_data.yolo_ready:
            user_data.total_frames += 1
            user_data.status_frame_count += 1
            if curr_time - user_data.status_start_time >= 1.0:
                fps = user_data.status_frame_count / (curr_time - user_data.status_start_time)
                print(f"⏱️ [PIPELINE] FPS: {fps:.1f}")
                user_data.status_start_time = curr_time
                user_data.status_frame_count = 0
            return

        user_data.yolo_ready = False

        if user_data.caps_info is None:
            user_data.caps_info = get_caps_from_pad_fixed(element.get_static_pad("sink"))

        fmt, w, h = user_data.caps_info
        if fmt is None or w is None or h is None:
            return

        frame_raw = get_numpy_from_buffer(buffer, fmt, w, h)
        if frame_raw is None:
            return

        roi = hailo.get_roi_from_buffer(buffer)
        depth_objs = roi.get_objects_typed(hailo.HAILO_DEPTH_MASK)
        depth_raw = None
        if len(depth_objs) > 0:
            depth_raw = depth_objs[0].get_data()

        try:
            if user_data.yolo_queue.full():
                try:
                    user_data.yolo_queue.get_nowait()
                    user_data.yolo_queue.task_done()
                except Exception:
                    pass
            user_data.yolo_queue.put_nowait((frame_raw, depth_raw, fmt))
        except queue.Full:
            pass

        user_data.total_frames += 1
        user_data.status_frame_count += 1
        if curr_time - user_data.status_start_time >= 1.0:
            fps = user_data.status_frame_count / (curr_time - user_data.status_start_time)
            print(f"⏱️ [PIPELINE] FPS: {fps:.1f}")
            user_data.status_start_time = curr_time
            user_data.status_frame_count = 0

    except Exception as e:
        print(f"❌ [Callback Error] {type(e).__name__}: {e}")

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    print("=" * 70)
    print("[DYNAMIC PIPELINE INTEGRATED]")
    print("YOLO(CPU) + Dynamic ROI + Depth(NPU) + VLM(NPU) + 번역 + Firestore")
    print("종료: Ctrl+C")
    print("=" * 70)

    if "--input" not in sys.argv:
        sys.argv.extend(["--input", "usb"])
    if "--width" not in sys.argv:
        sys.argv.extend(["--width", "640"])
    if "--height" not in sys.argv:
        sys.argv.extend(["--height", "480"])

    if not MODEL_PATH.exists():
        print(f"❌ YOLO 모델 폴더를 찾을 수 없습니다: {MODEL_PATH}")
        sys.exit(1)

    print(f"✅ YOLO model path: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH), task="detect")

    fs_thread = threading.Thread(target=start_firestore_listener, daemon=True)
    fs_thread.start()

    vlm_thread = threading.Thread(
        target=vlm_worker_thread,
        kwargs={"collection_name": "vlm_events"},
        daemon=True,
    )
    vlm_thread.start()

    global snapshot_event_queue
    snapshot_event_queue = snapshot_recorder_worker()

    user_data = DynamicAppCallback(model)
    user_data.snapshot_event_queue = snapshot_event_queue

    app = HeadlessDepthApp(app_callback, user_data)

    try:
        app.run()
    except KeyboardInterrupt:
        print("\n🛑 종료 요청 수신")
    finally:
        stop_event.set()
        try:
            vlm_queue.put(None)
        except Exception:
            pass
        try:
            user_data.yolo_queue.put(None)
        except Exception:
            pass
        print("⏳ 스레드 종료 대기 중...")
        try:
            user_data.yolo_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            vlm_thread.join(timeout=5.0)
        except Exception:
            pass
        print("✅ 종료 완료")


if __name__ == "__main__":
    main()
