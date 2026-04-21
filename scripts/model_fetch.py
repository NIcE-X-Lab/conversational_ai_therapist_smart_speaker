#!/usr/bin/env python3
"""Download the Gemma 4 E2B LiteRT model from HuggingFace.

Usage:
    python scripts/model_fetch.py

The script is idempotent — it skips the download when the model file
already exists at the expected path.
"""

import os
import sys
import time

REPO_ID = "litert-community/gemma-4-E2B-it-litert-lm"
LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "litert",
)
# Only download the model file needed for Jetson (skip web/Qualcomm variants)
_ALLOW_PATTERNS = ["gemma-4-E2B-it.litertlm", "README.md", ".gitattributes"]
_MODEL_FILENAME = "gemma-4-E2B-it.litertlm"


def _marker_path() -> str:
    return os.path.join(LOCAL_DIR, ".download_complete")


def _model_path() -> str:
    return os.path.join(LOCAL_DIR, _MODEL_FILENAME)


def _mark_complete() -> None:
    with open(_marker_path(), "w") as file_handle:
        file_handle.write("ok\n")


def _model_present() -> bool:
    return os.path.isfile(_model_path()) and os.path.getsize(_model_path()) > 0


def fetch_model():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] huggingface-hub is not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    marker = _marker_path()
    if os.path.isfile(marker) and _model_present():
        print(f"[OK] Model already downloaded at {LOCAL_DIR}")
        return

    if _model_present():
        print(f"[OK] Model file already present at {_model_path()} -- marking download complete.")
        _mark_complete()
        return

    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"[INFO] Downloading {REPO_ID} -> {LOCAL_DIR} ...")
    print("[INFO] This is ~2.6 GB and may take several minutes.")

    last_error = None
    for attempt in range(1, 6):
        try:
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=LOCAL_DIR,
                allow_patterns=_ALLOW_PATTERNS,
                resume_download=True,
            )
            break
        except TypeError:
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=LOCAL_DIR,
                allow_patterns=_ALLOW_PATTERNS,
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt == 5:
                raise
            wait_seconds = attempt * 10
            print(f"[WARN] Download attempt {attempt}/5 failed: {exc}")
            print(f"[WARN] Retrying in {wait_seconds}s and resuming the partial download...")
            time.sleep(wait_seconds)

    # Write marker so subsequent runs are no-ops
    if not _model_present():
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Model download completed but {_MODEL_FILENAME} is still missing")

    _mark_complete()

    print(f"[OK] Model downloaded to {LOCAL_DIR}")


if __name__ == "__main__":
    fetch_model()
