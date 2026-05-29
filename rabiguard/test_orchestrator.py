# rabiguard/test_orchestrator.py

import firebase_admin
from firebase_admin import credentials, firestore
from pathlib import Path
import sys

# ------------------------------------------------------------
# Path settings
# ------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
FIREBASE_KEY_PATH = ROOT_DIR / "firebase_key.json"

# ------------------------------------------------------------
# Firebase Initialization
# ------------------------------------------------------------
def init_db():
    if not firebase_admin._apps:
        if not FIREBASE_KEY_PATH.exists():
            print(f"❌ Firebase key file not found: {FIREBASE_KEY_PATH}")
            sys.exit(1)
        
        cred = credentials.Certificate(str(FIREBASE_KEY_PATH))
        firebase_admin.initialize_app(cred)
    
    return firestore.client()

# ------------------------------------------------------------
# Command Sender
# ------------------------------------------------------------
def send_command(db, cmd_type):
    print(f"\n📤 Sending command: '{cmd_type}' to Firestore...")
    try:
        doc_ref = db.collection("commands").add({
            "type": cmd_type,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        # Firestore returns (timestamp, doc_ref)
        print(f"✅ Command sent successfully! (Doc ID: {doc_ref[1].id})")
        print("💡 Wait for the Orchestrator to process and delete this document.")
    except Exception as e:
        print(f"❌ Failed to send command: {e}")

# ------------------------------------------------------------
# Main Menu
# ------------------------------------------------------------
def main():
    db = init_db()
    
    while True:
        print("\n" + "="*45)
        print("   Rafour Orchestrator Test Menu")
        print("="*45)
        print(" [Guard Controls]")
        print("  1. Start Guard (main.py)")
        print("  2. Stop Guard")
        print("\n [Streaming Controls]")
        print("  3. Start Stream (webrtc_video.py)")
        print("  4. Stop Stream")
        print("\n [ROI Controls]")
        print("  5. Trigger ROI Extraction")
        print("\n [System]")
        print("  0. Exit")
        print("="*45)
        
        choice = input("\nSelect an option (0-5): ").strip()
        
        if choice == '1':
            send_command(db, "start_guard")
        elif choice == '2':
            send_command(db, "stop_guard")
        elif choice == '3':
            send_command(db, "start_stream")
        elif choice == '4':
            send_command(db, "stop_stream")
        elif choice == '5':
            send_command(db, "trigger_roi")
        elif choice == '0':
            print("👋 Exiting test script.")
            break
        else:
            print("⚠️ Invalid option, please try again.")

if __name__ == "__main__":
    main()
