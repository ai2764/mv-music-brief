# mv-music-brief

把一首歌变成一份按时间对齐的音乐分析，供 LLM 规划 MV 分镜。产出内容：
- 逐句歌词的分镜卡片
- 能量、情绪、节拍信息
- ComfyUI / Wan 可用的音频驱动曲线和 4n+1 帧切段

Turn a song into a time-aligned, LLM-readable analysis for music-video planning. It produces:
- per-lyric-line shot cards
- energy, mood and beat information
- ComfyUI/Wan audio-reactive curves and 4n+1-frame clip splits

## Install
```bash
./setup.sh          # CPU (Linux)
./install_cuda.sh   # WSL2 / Linux + NVIDIA GPU (e.g. RTX 4090)
```

## Usage
```bash
. .venv/bin/activate
mvbrief song.mp3 -o out/song [--lyrics lyrics.txt] [--fps 16] [--whisper-model large-v3]
mvbrief-ui          # Gradio drag-and-drop UI
```

Outputs, all in `out/song/`:
- `analysis.json`, `analysis.md`, `analysis.png`
- `comfy/`: `curves.csv`, `schedules/*.txt`, `segments/<id>/`, `shots.json`, `shots.md`
- `agent_brief.md` and `agent_brief.json`
- `report.html`: a self-contained interactive report (Chinese UI). Add `--copy-audio` for playback.

Agent skill: `skill/mv-music-brief/SKILL.md`. Copy that folder into your agent's skills directory (e.g. `~/.codex/skills/`).

The pipeline uses librosa, essentia-tensorflow, demucs (htdemucs), faster-whisper and CLAP. Model weights are downloaded by `fetch_models.py` and are not included in this repo.
