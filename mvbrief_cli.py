#!/usr/bin/env python
"""mvbrief <audio> [--lyrics txt] [--fps 16] [-o dir]  — analyze -> export_comfy -> make_brief in one go.
Device: CUDA is used automatically when torch sees a GPU (override: --device cpu|cuda or env MVBRIEF_DEVICE)."""
import argparse, os, subprocess, sys, time
from pathlib import Path
import platform_setup

HERE = str(Path(__file__).resolve().parent)

def run_pipeline(audio, outdir=None, lyrics=None, fps=16, device=None, whisper_model=None, extra=None, log=print, copy_audio=False):
    outdir = str(Path(outdir) if outdir else Path(audio).resolve().parent / "mvbrief_out" / Path(audio).stem)
    platform_setup.setup()
    audio = str(Path(audio).resolve())
    if not platform_setup.ensure_ffmpeg(): log("[mvbrief] note: ffmpeg not found (mp3/wav/flac/ogg still work via soundfile; install ffmpeg or `pip install imageio-ffmpeg` for m4a/aac)")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    if device: env["MVBRIEF_DEVICE"] = device
    py = sys.executable; t0 = time.time()
    cmd = [py, os.path.join(HERE, "analyze.py"), audio, "-o", outdir]
    if lyrics: cmd += ["--lyrics", lyrics]
    if whisper_model: cmd += ["--whisper-model", whisper_model]
    cmd += list(extra or [])
    for step in (cmd, [py, os.path.join(HERE, "export_comfy.py"), outdir, "--fps", str(fps)],
                 [py, os.path.join(HERE, "make_brief.py"), outdir, "--fps", str(fps)],
                 [py, os.path.join(HERE, "make_report.py"), outdir] + (["--copy-audio"] if copy_audio else [])):
        log(f"[mvbrief] {' '.join(os.path.basename(c) if i < 2 else c for i, c in enumerate(step))}")
        subprocess.run(step, check=True, env=env)
    log(f"[mvbrief] done in {time.time() - t0:.0f}s -> {outdir}/agent_brief.md, {outdir}/report.html")
    return outdir

def main():
    ap = argparse.ArgumentParser(prog="mvbrief", description=__doc__)
    ap.add_argument("audio"); ap.add_argument("-o", "--outdir"); ap.add_argument("--lyrics"); ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--device", choices=["cpu", "cuda"]); ap.add_argument("--copy-audio", action="store_true", help="copy audio next to report.html for playback"); ap.add_argument("--whisper-model", help="small (default), medium, large-v3 (good on GPU)")
    a, extra = ap.parse_known_args()   # unknown args (e.g. --no-clap, --language zh) are passed to analyze.py
    run_pipeline(a.audio, a.outdir, a.lyrics, a.fps, a.device, a.whisper_model, extra, copy_audio=a.copy_audio)

if __name__ == "__main__": main()
