#!/bin/bash
# Recreate the venv and fetch models. Run from /workspace/music-analyzer.
set -e
python3 -m venv .venv && . .venv/bin/activate && pip -q install -U pip
pip -q install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip -q install -r requirements.txt && pip -q install -e ".[ui]"
python fetch_models.py
