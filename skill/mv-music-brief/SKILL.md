---
name: mv-music-brief
description: Analyze a song (audio file, optional real lyrics) into a time-aligned music brief and plan a music-video shot list per lyric line for ComfyUI + Wan video generation. Use when the user wants an MV storyboard, shot list, Wan prompts, beat-synced cuts, or audio-reactive curves for a song.
---

# MV Music Brief

Turn a song into a per-lyric-line shot list for AI video (ComfyUI + Wan). Music analysis is automatic; YOU translate it into shots.

**Priority order (highest first):**
1. User's creative intent (story, characters, style, "我的创意" section). Never override it.
2. Lyric meaning.
3. Music features (energy, pace, trend, arrangement, beats).
4. Auto mood tags (least reliable; ignore when they contradict lyrics or user intent).

## Step 1 — Run the pipeline

Project: `/workspace/music-analyzer` (Linux/WSL2; GPU used automatically if CUDA works).

```bash
cd /workspace/music-analyzer && . .venv/bin/activate
mvbrief song.mp3 -o out/mysong                         # analyze + export + brief
mvbrief song.mp3 -o out/mysong --lyrics lyrics.txt     # better: real lyrics, one sung line per line
mvbrief song.mp3 -o out/mysong --fps 24                # Wan default is 16 fps
```
Equivalent manual steps: `python analyze.py <audio> -o <out> [--lyrics txt]`, then `python export_comfy.py <out> --fps 16`, then `python make_brief.py <out> --fps 16`.
First-time setup: `./setup.sh` (CPU) or `./install_cuda.sh` (NVIDIA GPU).

Do not force `--language`; it is auto-detected. Run time on CPU is about 2–3 min for a 4-min song.

## Step 2 — Read the outputs

| File | Use |
|---|---|
| `<out>/report.html` | Human-readable interactive report (open in a browser; `--copy-audio` enables playback). Not needed by the agent. |
| `<out>/agent_brief.md` | **Main input.** Song overview + one card per segment. Read this first. |
| `<out>/agent_brief.json` | Same content, machine-readable, with full beat frame lists. |
| `<out>/comfy/shots.json` | Per segment: frames, Wan clip split, wan_prompt, motion/denoise, beat/downbeat/accent frames. |
| `<out>/comfy/segments/<id>/*.txt` | Per-clip audio-reactive schedules `0:(0.12), 1:(0.35), …` for ComfyUI nodes. |
| `<out>/comfy/curves.csv` | Whole-song per-frame curves (sub_low, low_mid, mid, high, rms, kick, snare_hat, vocal, bass_stem, *_trig). |

## Field glossary

- **segment**: one lyric line (`type=lyric`) or one instrumental gap/phrase (`type=instrumental`). The basic shot unit.
- **time** (s) / **frames**: global frame range at the brief's fps. Beat/downbeat/accent frames on cards are **relative to segment start** (frame 0 = segment start).
- **energy** 0–1, normalized to this song: <0.25 very quiet · <0.45 low · <0.62 medium · <0.8 high · ≥0.8 very high.
- **trend** (inside the segment): rising · falling · peak · steady.
- **pace** slow / medium / fast (`pace_score` 0–1) from energy + note density + drum activity. Not raw BPM. If the brief says half-time, plan cutting around the *felt* BPM.
- **arrangement**: vocal-only · sparse (vocal + little backing) · vocals without drums · full band · instrumental.
- **beat** = every beat · **downbeat** = bar start (best cut point) · **accent** = strong hit (flash cut, light pulse, action sync).
- **section** labels with `?` / “（推测）” are guesses. Verify with lyric repetition (repeated lines = chorus).
- **mood**: ≤3 auto tags. A hint only.
- **Wan clips**: each generated clip must be **4n+1 frames and ≤81** (≈5 s at 16 fps). Long segments are already split on downbeats in `clips`: generate `wan_frames`, then trim the extra frames.
- **motion_strength / denoise**: suggested ranges. Start in the middle of the range.

## Step 3 — Translate music → shots (rules)

| Music feature | Shot decision |
|---|---|
| very quiet/low energy + slow pace | static or very slow push-in, long take (1 shot or cut every 4 bars), wide/atmospheric |
| sparse/vocal-only + lyric | close-up or medium close-up **lip-sync** |
| full band + lyric | medium shot lip-sync, cut-aways on downbeats |
| rising trend | dolly-in, crane up, speed ramp into the next downbeat |
| falling trend | pull-back, slow drift away, fade/rack focus |
| peak / drop / pace fast | quick cuts every 1–2 beats, handheld, whip pans on accents |
| instrumental segment | B-roll, landscape, instrument/performance inserts, story beats without lip-sync |
| section change | change scene, location, or color look |
| accent frames | flash cut, light pulse, or a hit of action exactly on that frame |

Cutting rules:
- Cut on **downbeat frames** by default. Fast parts may cut on beats.
- Slow parts: hold through at least 2 bars.
- Every shot ≤81 frames. Use the given `clips` split.

Consistency rules:
- Keep one visual look per section.
- Keep characters and wardrobe consistent across shots. Restate key identity details in every Wan prompt.

## Step 4 — Output format

Use the user's language for descriptions; write Wan prompts in English. One row per shot:

```
| seg | time | frames (clip) | lyric | shot size | visual | camera | lip-sync | cut frames | Wan prompt | motion / denoise |
```

Under each section, add one line: the section's visual idea and how it serves the story.
At the end, add:
- a short list of recurring motifs;
- any shots that need special workflows: lip-sync model, first/last-frame control, audio-reactive nodes driven by `kick` or `beat_trig` schedules.

**Wan prompt recipe:** `[subject + action], [setting], [camera move], [lighting/mood], [style]`. Keep it under ~60 words, concrete and visual. No lyrics text inside prompts.

## Example (abridged)

Card from the brief:
```
### 002｜15.81–19.37s｜帧 253–310（57帧）｜主歌（推测）｜「清晨的风吹过街角」
- 稀疏：人声+少量伴奏；能量较弱（0.41），走向渐弱；节奏慢（0.36）；情绪：放松、快乐、亲密
- 拍点：第0帧起约每7帧一拍；小节首拍帧：[22, 50]
- Wan分段：57帧→生成57帧
```
User intent: “a grown woman returns to her seaside hometown, 90s film look”.
Output row:
```
| 002 | 15.8–19.4 | 253–310 (57f) | 清晨的风吹过街角 | close-up | woman in her 30s singing softly on a beach at dawn, wet sand, gentle waves | slow pull-back | yes | 22, 50 | a woman in her thirties softly singing on a quiet beach at dawn, gentle waves on white sand, slow pull-back, soft warm backlight, 90s 35mm film grain | 0.3 / 0.5 |
```
