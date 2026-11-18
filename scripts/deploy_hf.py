"""
deploy_hf.py: upload the app to a Hugging Face Space (free Docker hosting, 16 GB RAM).

One-time setup:
  1. Create a free account at https://huggingface.co
  2. Settings > Access Tokens > Create new token, type "Write". Copy it.
  3. pip install huggingface_hub

Deploy (creates the Space the first time, updates it after that):
    python scripts/deploy_hf.py --space YOUR_HF_USERNAME/cassava-leaf-doctor --token hf_xxx

The Space builds the Dockerfile and runs it. Building takes about 5 to 10 minutes.
The app is then live at https://YOUR_HF_USERNAME-cassava-leaf-doctor.hf.space
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPACE_HEADER = """---
title: Cassava Leaf Doctor
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="username/space-name")
    ap.add_argument("--token", required=True, help="Hugging Face write token")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)

    # Spaces read their settings from the top of README.md
    readme = SPACE_HEADER + (ROOT / "README.md").read_text(encoding="utf-8")
    api.upload_file(path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md",
                    repo_id=args.space, repo_type="space")
    api.upload_folder(
        folder_path=str(ROOT), repo_id=args.space, repo_type="space",
        allow_patterns=["Dockerfile", "requirements.txt", "src/**", "api/**", "ui/**",
                        "models/cassava_mobilenetv2.keras", "models/model_metadata.json",
                        "data/train/**", "data/test/**"],
        commit_message="Deploy cassava classifier",
    )
    user, name = args.space.split("/")
    print(f"Uploaded. Build logs: https://huggingface.co/spaces/{args.space}")
    print(f"Live app (after the build): https://{user}-{name}.hf.space".lower())


if __name__ == "__main__":
    main()
