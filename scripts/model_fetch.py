#!/usr/bin/env python3
"""Download the Gemma 4 E2B LiteRT model from HuggingFace.

Usage:
    python scripts/model_fetch.py

The script is idempotent — it skips the download when the model file
already exists at the expected path.
"""

import os
import sys

REPO_ID = "litert-community/gemma-4-E2B-it-litert-lm"
LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "litert",
)
# Only download the model file needed for Jetson (skip web/Qualcomm variants)
_ALLOW_PATTERNS = ["gemma-4-E2B-it.litertlm", "README.md", ".gitattributes"]


def fetch_model():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] huggingface-hub is not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    marker = os.path.join(LOCAL_DIR, ".download_complete")
    if os.path.isfile(marker):
        print(f"[OK] Model already downloaded at {LOCAL_DIR}")
        return

    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"[INFO] Downloading {REPO_ID} -> {LOCAL_DIR} ...")
    print("[INFO] This is ~2.6 GB and may take several minutes.")

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        allow_patterns=_ALLOW_PATTERNS,
    )

    # Write marker so subsequent runs are no-ops
    with open(marker, "w") as f:
        f.write("ok\n")

    print(f"[OK] Model downloaded to {LOCAL_DIR}")


if __name__ == "__main__":
    fetch_model()
