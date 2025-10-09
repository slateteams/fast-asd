"""Modal configuration for TalkNet ASD"""

import modal

# Modal image configuration
image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install_from_requirements("requirements.txt")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "libglib2.0-0"])
    .add_local_dir("talknet", remote_path="/root/talknet")
    .add_local_dir("utils", remote_path="/root/utils")
    .add_local_file("config.py", remote_path="/root/config.py")
)

# Modal volume for model storage
model_volume = modal.Volume.from_name("talknet-models", create_if_missing=True)

# GPU configuration
GPU_CONFIG = ["L4", "A10", "L40S"]
MEMORY_MB = 16384
TIMEOUT_SECONDS = 3600
