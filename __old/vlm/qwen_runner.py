# qwen_runner.py

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from hailo_platform import VDevice
from hailo_platform.genai import VLM

from prompts import SYSTEM_PROMPT, USER_PROMPT
from translator import translate_to_korean
from firebase_writer import save_vlm_result_to_firestore

repo_root = None
for p in Path(__file__).resolve().parents:
    if (p / "hailo_apps" / "config" / "config_manager.py").exists():
        repo_root = p
        break

if repo_root is not None:
    sys.path.insert(0, str(repo_root))

from hailo_apps.python.core.common.core import handle_list_models_flag, resolve_hef_path
from hailo_apps.python.core.common.defines import (
    VLM_CHAT_APP,
    SHARED_VDEVICE_GROUP_ID,
    HAILO10H_ARCH,
)
from hailo_apps.python.core.common.hailo_logger import get_logger

logger = get_logger(__name__)


def clean_response(response: str) -> str:
    """VLM 응답 문자열에서 불필요한 부분 제거"""
    if not response:
        return ""

    text = response

    if ". [{'type'" in text:
        text = text.split(". [{'type'")[0]

    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0]

    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Qwen VLM Runner")

    parser.add_argument(
        "--hef-path",
        type=str,
        default=None,
        help="Path to HEF model file",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models",
    )

    parser.add_argument(
        "--image-path",
        type=str,
        default="/home/rafour/workspace/jongmin/rafour-app/images/test3.jpg",
        help="Path to input image file",
    )

    parser.add_argument(
        "--collection-name",
        type=str,
        default="vlm_results",
        help="Firestore collection name",
    )

    handle_list_models_flag(parser, VLM_CHAT_APP)
    args = parser.parse_args()

    hef_path = resolve_hef_path(
        args.hef_path,
        app_name=VLM_CHAT_APP,
        arch=HAILO10H_ARCH,
    )

    if hef_path is None:
        logger.error("Failed to resolve HEF path for VLM model.")
        sys.exit(1)

    logger.info(f"Using HEF: {hef_path}")
    print(f"✓ Model file found: {hef_path}")

    vdevice = None
    vlm = None

    try:
        print("\n[1/5] Initializing Hailo device...")
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        vdevice = VDevice(params)
        print("✓ Hailo device initialized")

        print("[2/5] Loading VLM model...")
        vlm = VLM(vdevice, str(hef_path))
        print("✓ Model loaded successfully")

        prompt = [
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
        ]

        image_path = Path(args.image_path)

        print(f"[3/5] Loading image from: {image_path}")
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Could not load image file: {image_path}")

        print(f"✓ Image loaded (size: {image.shape[1]}x{image.shape[0]})")

        print("[4/5] Preprocessing image...")

        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (336, 336),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        print("✓ Image preprocessed (resized to 336x336, converted to RGB)")

        print("[5/5] Sending prompt with image to VLM...")
        print(f"   System prompt: '{SYSTEM_PROMPT}'")
        print(f"   User prompt: '{USER_PROMPT}'")

        response = vlm.generate_all(
            prompt=prompt,
            frames=[image],
            temperature=0.1,
            seed=42,
            max_generated_tokens=80,
        )

        result = clean_response(response)
        ko_result = translate_to_korean(result)

        print("\nResponse received:")
        print("-" * 60)
        print(result)
        print("-" * 60)

        print("\nKorean Translation:")
        print("-" * 60)
        print(ko_result)
        print("-" * 60)

        print("\nSaving result to Firestore...")

        try:
            doc_id = save_vlm_result_to_firestore(
                english_text=result,
                korean_text=ko_result,
                image_path=str(image_path),
                collection_name=args.collection_name,
            )

            print(f"✓ Firestore saved successfully")
            print(f"✓ Document ID: {doc_id}")

        except Exception as e:
            print(f"[Firestore Save Error] {e}")

        print("\n✓ Example completed successfully")

    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        sys.exit(1)

    finally:
        if vlm:
            try:
                vlm.clear_context()
                vlm.release()
            except Exception as e:
                logger.warning(f"Error releasing VLM: {e}")

        if vdevice:
            try:
                vdevice.release()
            except Exception as e:
                logger.warning(f"Error releasing VDevice: {e}")


if __name__ == "__main__":
    main()