#!/usr/bin/env python3
"""
一键下载 PickScore / CLIPScore / Aesthetic Score 权重到本地 models 目录
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

# 本地保存根目录
LOCAL_ROOT = Path("models").resolve()
LOCAL_ROOT.mkdir(exist_ok=True)

# 需要下载的模型列表
MODELS = {
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "yuvalkirstain/PickScore_v1",
    "openai/clip-vit-large-patch14",   # 同时用于 CLIPScore 与 Aesthetic Score
}

for repo_id in MODELS:
    repo_name = repo_id.split("/")[-1]          # 取最后一段作为子目录名
    target_dir = LOCAL_ROOT / repo_name
    print(f"Downloading {repo_id} -> {target_dir}")
    snapshot_download(
        repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,  # 不使用软链接，方便迁移
    )

print("✅ All models downloaded to", LOCAL_ROOT)
