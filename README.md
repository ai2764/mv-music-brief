# mv-music-brief

把一首歌变成一份按时间对齐的音乐分析，供 LLM 规划 MV 分镜。产出内容：
- 逐句歌词的分镜卡片
- 能量、情绪、节拍信息
- ComfyUI / Wan 可用的音频驱动曲线和 4n+1 帧切段
- 一个交互式 HTML 报告

Turn a song into a time-aligned, LLM-readable analysis for music-video planning. It produces:
- per-lyric-line shot cards
- energy, mood and beat information
- ComfyUI/Wan audio-reactive curves and 4n+1-frame clip splits
- an interactive HTML report

## Windows 11（原生 native，RTX 4090 等 NVIDIA 显卡）

需要 Python 3.10–3.12（推荐 3.11）和 NVIDIA 驱动。ffmpeg 可选。
Requires Python 3.10–3.12 (3.11 recommended) and the NVIDIA driver. ffmpeg is optional.

```powershell
winget install Python.Python.3.11
winget install Gyan.FFmpeg        # optional
git clone https://github.com/ai2764/mv-music-brief; cd mv-music-brief
powershell -ExecutionPolicy Bypass -File .\install.ps1          # add -Cpu for no GPU
```

使用 / Usage:
- **拖放 / drag-and-drop:** drop an `.mp3` / `.wav` / `.flac` onto `mvbrief.bat`.
  - If `<song>.txt` sits next to the audio, it is used as the lyrics.
  - Results go to `<song folder>\mvbrief_out\<song>\`, and `report.html` opens automatically.
- **命令行 / command line:** `.\mvbrief.bat song.mp3 --lyrics 歌词.txt --fps 16 --whisper-model large-v3`
- **界面 / UI:** `.\.venv\Scripts\mvbrief-ui.exe`, then open http://127.0.0.1:7860

Notes for Windows:
- essentia (Linux-only) is skipped. Key comes from librosa (Krumhansl), BPM from librosa (plus beat_this if installed via `pip install -e .[beat]`), and mood/valence/arousal from CLAP.
- Paths and lyrics may contain Chinese characters; everything is UTF-8.

## Linux / WSL2

```bash
./install_cuda.sh   # NVIDIA GPU (e.g. RTX 4090, WSL2 or native)
./setup.sh          # CPU
. .venv/bin/activate
mvbrief song.mp3 -o out/song [--lyrics lyrics.txt] [--fps 16] [--whisper-model large-v3] [--copy-audio]
mvbrief-ui
```

## Outputs / 输出

All in the output folder:
- `analysis.json`, `analysis.md`, `analysis.png`: full analysis (sections, per-lyric-line segments, stems, lyrics, mood, beats)
- `comfy/`: `curves.csv`, `schedules/*.txt` (`0:(0.12), 1:(0.35), …`), `segments/<id>/`, `shots.json`, `shots.md`. Wan clips are 4n+1 frames and ≤81 frames.
- `agent_brief.md` / `agent_brief.json`: compact Chinese brief to paste into an LLM agent
- `report.html`: self-contained interactive report (use `--copy-audio` for playback)

## Agent skill

`skill/mv-music-brief/SKILL.md`. Copy the folder into your agent's skills directory, e.g. `~/.codex/skills/mv-music-brief` or `%USERPROFILE%\.codex\skills\mv-music-brief`.

The pipeline uses librosa, demucs (htdemucs), faster-whisper, CLAP and, optionally, essentia-tensorflow (Linux) and beat_this. Model weights are downloaded by `fetch_models.py` and are not included in this repo.
