#!/bin/bash
# Install mvbrief with NVIDIA CUDA (tested target: RTX 4090, WSL2 Ubuntu 22.04/24.04 or native Linux).
# Prereqs: NVIDIA driver on the host (Windows driver for WSL2; do NOT install a Linux driver inside WSL), `nvidia-smi` works.
set -e
cd "$(dirname "$0")"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "WARNING: nvidia-smi not found; GPU may be unavailable"
sudo apt-get update && sudo apt-get install -y python3-venv python3-dev ffmpeg build-essential git
python3 -m venv .venv && . .venv/bin/activate && pip install -U pip wheel
# PyTorch with CUDA 12.4 wheels (bundle their own CUDA runtime; works on 4090 / Ada)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
# cuDNN 9 + cuBLAS for faster-whisper (CTranslate2) on GPU
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
echo "export LD_LIBRARY_PATH=$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:\$LD_LIBRARY_PATH" >> .venv/bin/activate
. .venv/bin/activate
pip install -e ".[essentia,ui]"
# optional: allin1 structure analysis (needs python3-dev for madmom; natten must match torch/CUDA — skip if it fails)
( pip install cython && pip install --no-build-isolation git+https://github.com/CPJKU/madmom && pip install allin1 ) || echo "allin1 skipped (librosa fallback is used)"
python fetch_models.py
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -c "import ctranslate2; print('ctranslate2 cuda devices:', ctranslate2.get_cuda_device_count())"
echo "Done. Use:  . .venv/bin/activate && mvbrief song.mp3 -o out/song --whisper-model large-v3   |  UI: mvbrief-ui"
