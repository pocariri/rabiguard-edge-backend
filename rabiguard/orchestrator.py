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

def start_process(name, script_path):
    """새로운 서브프로세스를 실행합니다."""
    if processes.get(name) and processes[name].poll() is None:
        print(f"⚠️ [Orchestrator] {name} 프로세스가 이미 실행 중입니다.")
        return processes[name]

    print(f"🚀 [Orchestrator] {name} 프로세스 시작: {script_path}")
    proc = subprocess.Popen([sys.executable, str(script_path)])
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
                        print("🔄 [Orchestrator] Guard 시작을 위해 스트리밍을 종료합니다.")
                        kill_process("stream")
                    if processes["roi"]:
                        print("🔄 [Orchestrator] Guard 시작을 위해 ROI 추출을 종료합니다.")
                        kill_process("roi")
                    
                    start_process("guard", CURRENT_DIR / "main.py")

                elif cmd_type == "stop_guard":
                    should_guard = False
                    kill_process("guard")

                # 2. 영상 스트리밍 (Stream) 제어
                elif cmd_type == "start_stream":
                    # 카메라 충돌 방지를 위해 Guard 종료
                    if processes["guard"]:
                        print("🔄 [Orchestrator] 스트리밍을 위해 Guard를 일시 중지합니다.")
                        kill_process("guard")
                    start_process("stream", ROOT_DIR / "webRTC" / "webrtc_video.py")

                elif cmd_type == "stop_stream":
                    kill_process("stream")
                    # 스트리밍 종료 후 Guard 상태 복구
                    if should_guard:
                        print("🔄 [Orchestrator] 스트리밍이 종료되어 Guard를 재개합니다.")
                        start_process("guard", CURRENT_DIR / "main.py")

                # 3. ROI 자동 추출 (일회성)
                elif cmd_type == "trigger_roi":
                    # 카메라 충돌 방지를 위해 Guard 종료
                    if processes["guard"]:
                        print("🔄 [Orchestrator] ROI 추출을 위해 Guard를 일시 중지합니다.")
                        kill_process("guard")
                    start_process("roi", CURRENT_DIR / "dynamic_roi_extractor.py")

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
    print("감시 대상: main.py, webrtc_video.py, dynamic_roi_extractor.py")
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
                # ROI 추출 완료 후 Guard 상태 복구
                if should_guard and not processes["stream"]:
                    print("🔄 [Orchestrator] ROI 추출이 완료되어 Guard를 재개합니다.")
                    start_process("guard", CURRENT_DIR / "main.py")
            
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 [Orchestrator] 종료 요청 수신")
    finally:
        stop_event.set()
        kill_process("guard")
        kill_process("stream")
        kill_process("roi")
        print("✅ [Orchestrator] 모든 자원 정리 완료")

if __name__ == "__main__":
    main()
