# scripts/download_llama3.py

from huggingface_hub import snapshot_download
from pathlib import Path

TARGET_DIR = Path("models/llama3.1-8b").resolve()

if TARGET_DIR.exists() and any(TARGET_DIR.iterdir()):
    print(f"✅ Model đã tồn tại ở: {TARGET_DIR}, không cần tải lại.")
else:
    print("⬇️ Đang tải mô hình LLaMA 3.1 8B lần đầu...")
    snapshot_download(
        repo_id="meta-llama/Llama-3.1-8B",
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False,
        token=True  # Tự dùng token đã login bằng `huggingface-cli login`
    )
    print(f"✅ Đã tải mô hình vào: {TARGET_DIR}")
