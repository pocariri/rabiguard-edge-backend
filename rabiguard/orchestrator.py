# orchestrator.py

import os
import sys
import time
import subprocess
import signal
from pathlib import Path

# ------------------------------------------------------------
# Path settings
# ------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from firebase_writer import init_firestore
    from config import stop_event
except ImportError:
    from .firebase_writer import init_firestore
    from .config import stop_event

# ------------------------------------------------------------
# State Management
# ------------------------------------------------------------
# 현재 감시(main.py)가 켜져 있어야 하는 상태인지 추적
should_guard = False

processes = {
    "guard": None,    # main.py
    "stream": None,   # webrtc_video.py
    "roi": None,      # dynamic_roi_extractor.py
    "data": None,     # webrtc_data_transfer.py
}

def kill_process(name):
    """실행 중인 서브프로세스를 안전하게 종료합니다."""
    proc = processes.get(name)
    if proc and proc.poll() is None:
        print(f"🛑 [Orchestrator] {name} 프로세스 종료 중...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print(f"✅ [Orchestrator] {name} 종료 완료")
    processes[name] = None

def start_process(name, script_path, args=None):
    """새로운 서브프로세스를 실행합니다."""
    if processes.get(name) and processes[name].poll() is None:
        print(f"⚠️ [Orchestrator] {name} 프로세스가 이미 실행 중입니다.")
        return processes[name]

    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    print(f"🚀 [Orchestrator] {name} 프로세스 시작: {script_path} (args: {args})")
    proc = subprocess.Popen(cmd)
    processes[name] = proc
    return proc

# ------------------------------------------------------------
# Firestore Command Callback
# ------------------------------------------------------------
def on_command_snapshot(col_snapshot, changes, read_time):
    global should_guard
    for change in changes:
        try:
            if change.type.name == "ADDED":
                doc = change.document
                cmd_data = doc.to_dict()
                cmd_type = cmd_data.get("type")

                print(f"📩 [Orchestrator] 명령 수신: {cmd_type}")

                # 1. 침입 탐지 (Guard) 제어
                if cmd_type == "start_guard":
                    should_guard = True
                    # 카메라 충돌 방지를 위해 실행 중인 다른 작업 종료
                    if processes["stream"]:
                        kill_process("stream")
                    if processes["roi"]:
                        kill_process("roi")
                    
                    start_process("guard", CURRENT_DIR / "main.py")

                elif cmd_type == "stop_guard":
                    should_guard = False
                    kill_process("guard")

                # 2. 영상 스트리밍 (Stream) 제어
                elif cmd_type == "start_stream":
                    if processes["guard"]:
                        kill_process("guard")
                    start_process("stream", ROOT_DIR / "webRTC" / "webrtc_video.py")

                elif cmd_type == "stop_stream":
                    kill_process("stream")
                    if should_guard:
                        start_process("guard", CURRENT_DIR / "main.py")

                # 3. ROI 자동 추출
                elif cmd_type == "trigger_roi":
                    if processes["guard"]:
                        kill_process("guard")
                    start_process("roi", CURRENT_DIR / "dynamic_roi_extractor.py")

                # 4. WebRTC 데이터 전송 (스냅샷 다운로드)
                elif cmd_type == "download_event":
                    event_id = cmd_data.get("event_id")
                    if event_id:
                        # 데이터 전송은 파일을 읽는 작업이므로 감시와 병행 가능하지만,
                        # 시그널링 채널 혼선을 막기 위해 스트리밍이 켜져있다면 종료 고려
                        if processes["stream"]:
                            print("🔄 [Orchestrator] 데이터 전송을 위해 스트리밍을 종료합니다.")
                            kill_process("stream")

                        kill_process("data")
                        start_process("data", ROOT_DIR / "webRTC" / "webrtc_data_transfer.py", [event_id])
                    else:
                        print("⚠️ [Orchestrator] download_event 명령에 event_id가 없습니다.")

                # 명령 처리 후 문서 삭제
                doc.reference.delete()

        except Exception as e:
            print(f"⚠️ [Orchestrator Command Error] {e}")

# ------------------------------------------------------------
# Main Loop
# ------------------------------------------------------------
def main():
    print("=" * 70)
    print("[MASTER ORCHESTRATOR STARTED (Auto-Switching Mode)]")
    print("감시 대상: main.py, webrtc_video.py, dynamic_roi_extractor.py, webrtc_data_transfer.py")
    print("종료: Ctrl+C")
    print("=" * 70)

    try:
        db = init_firestore()
        query_watch = db.collection("commands").on_snapshot(on_command_snapshot)
        print("✅ [Orchestrator] Firestore 'commands' 컬렉션 감시 시작")

        while not stop_event.is_set():
            # ROI 추출 프로세스 완료 체크
            if processes["roi"] and processes["roi"].poll() is not None:
                print("✅ [Orchestrator] ROI 추출 완료")
                processes["roi"] = None
                if should_guard and not processes["stream"]:
                    start_process("guard", CURRENT_DIR / "main.py")
            
            # 데이터 전송 프로세스 완료 체크
            if processes["data"] and processes["data"].poll() is not None:
                print("✅ [Orchestrator] 데이터 전송 완료")
                processes["data"] = None
                # 전송 완료 후 원래 상태 복구 필요 시 로직 추가 가능
            
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 [Orchestrator] 종료 요청 수신")
    finally:
        stop_event.set()
        kill_process("guard")
        kill_process("stream")
        kill_process("roi")
        kill_process("data")
        print("✅ [Orchestrator] 모든 자원 정리 완료")

if __name__ == "__main__":
    main()
