#!/usr/bin/env python
"""Export analysis.json -> ComfyUI / Wan friendly audio-reactive curves + per-shot JSON/markdown.
Usage: python export_comfy.py <outdir-of-analyze.py> [--fps 16] [--audio <file>] [--max-frames 81]
Writes <outdir>/comfy/: curves.csv, schedules/<curve>.txt, segments/<id>/{curves.csv,<curve>.txt}, shots.json, shots.md, bands_first30s.png"""
import argparse, json, os, math, numpy as np, librosa

SR = 22050
BANDS = {"sub_low": (20, 150), "low_mid": (150, 500), "mid": (500, 2000), "high": (4000, 11025)}

def band_env(y, fps, lo, hi):
    hop = 256; S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=2048); m = (f >= lo) & (f < hi)
    e = np.sqrt(S[m].sum(0)); return resample_to_fps(e, SR / hop, fps)

def rms_env(y, fps):
    hop = 256; return resample_to_fps(librosa.feature.rms(y=y, hop_length=hop)[0], SR / hop, fps)

def resample_to_fps(x, src_rate, fps, n=None):
    n = n or int(math.ceil(len(x) / src_rate * fps)); out = np.zeros(n)
    for i in range(n):   # max-pool so short transients survive the downsampling
        a, b = int(i / fps * src_rate), int((i + 1) / fps * src_rate); out[i] = x[a:max(b, a + 1)].max() if a < len(x) else 0
    return out

def attack_release(x, fps, attack_s=0.02, release_s=0.25):
    a = math.exp(-1 / (attack_s * fps)) if attack_s > 0 else 0; r = math.exp(-1 / (release_s * fps)); y = np.zeros_like(x); p = 0
    for i, v in enumerate(x):
        c = a if v > p else r; p = c * p + (1 - c) * v; y[i] = p
    return y

def norm01(x, pct=99):
    x = np.asarray(x, float); lo, hi = np.percentile(x, 1), np.percentile(x, pct)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

def impulses(times, n, fps, decay_s=0.25, weights=None):
    y = np.zeros(n); k = np.arange(int(decay_s * fps * 5) + 1); kern = np.exp(-k / max(decay_s * fps, 1e-3))
    for j, t in enumerate(times):
        i = int(round(t * fps))
        if 0 <= i < n:
            w = 1.0 if weights is None else weights[j]; seg = kern[: n - i] * w; y[i:i + len(seg)] = np.maximum(y[i:i + len(seg)], seg)
    return y

def sched(vals, offset=0):
    return ", ".join(f"{i}:({v:.3f})" for i, v in enumerate(vals))

def wan_len(n, max_frames):
    """Wan wants 4k+1 frames; return clip lengths covering n frames, each <= max_frames."""
    nclips = max(1, math.ceil((n - 1) / (max_frames - 1)))
    per = n / nclips; out = []
    for c in range(nclips):
        m = int(round(per)); L = 4 * max(1, math.ceil((m - 1) / 4)) + 1; out.append(min(L, max_frames))
    return out

MOOD_LOOK = {"relaxed": "calm soft light", "gentle": "gentle diffused light", "dreamy": "dreamy haze, soft bloom", "nostalgic": "warm nostalgic film tone",
             "happy": "warm bright light", "bright": "bright airy light", "euphoric": "glowing euphoric light flares", "intimate": "intimate shallow depth of field",
             "romantic": "romantic golden hour glow", "sad": "muted melancholic tones", "dark": "low-key moody shadows", "tense": "tense flickering light",
             "aggressive": "harsh contrast strobing light", "epic": "epic wide scale, dramatic sky", "mysterious": "misty mysterious atmosphere",
             "triumphant": "triumphant soaring light", "hopeful": "hopeful sunrise light", "calm": "still calm atmosphere", "playful": "playful lively colors",
             "energetic": "vivid energetic colors", "melancholic": "blue melancholic tones", "hopeless": "desaturated somber tones", "defeated": "dim fading light",
             "gloomy": "overcast gloomy light", "grounded": "naturalistic grounded look", "cold": "cool blue tones", "serious": "restrained serious mood"}

def prompt_for(s):
    ps = s.get("pace_score", s["camera_motion_hint"].get("pace_score", 0.5)); tr = s["energy"]["trend"]; arr = s["arrangement"]
    ev = {e["type"] for e in s["events"]}; acc = sum(e["type"] == "accent" for e in s["events"])
    if "drop/energy-jump" in ev or (tr == "peak" and ps >= 0.6): cam = "fast handheld camera, quick whip pans, dynamic motion"
    elif tr == "rising": cam = "smooth dolly-in, camera slowly rising" if ps < 0.6 else "fast dolly-in, crane up, accelerating motion"
    elif tr == "falling": cam = "slow pull-back, camera drifting away"
    elif ps < 0.4: cam = "static shot, very slow push-in, minimal motion"
    elif ps < 0.6: cam = "slow lateral tracking shot, gentle motion"
    else: cam = "steady gimbal tracking shot, energetic motion"
    subj = ""
    if s["type"] == "lyric":
        subj = "close-up of the singer singing softly" if ("sparse" in arr or "vocal-only" in arr or ps < 0.4) else "medium shot of the singer singing"
    else: subj = "atmospheric b-roll, wide establishing shot" if ps < 0.45 else "performance b-roll, instruments in motion"
    moods = [m for m, p in s["mood"].get("clap", {}).get("mood", [])[:3]]
    look = ", ".join(dict.fromkeys(MOOD_LOOK[m] for m in moods if m in MOOD_LOOK))
    extra = "subtle light pulses on the beat" if acc >= 2 and ps >= 0.45 else ""
    return ", ".join(x for x in [subj, cam, look, extra] if x)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("outdir"); ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--audio"); ap.add_argument("--max-frames", type=int, default=81); a = ap.parse_args()
    A = json.load(open(os.path.join(a.outdir, "analysis.json"))); fps = a.fps
    cd = os.path.join(a.outdir, "comfy"); os.makedirs(os.path.join(cd, "schedules"), exist_ok=True); os.makedirs(os.path.join(cd, "segments"), exist_ok=True)
    y, _ = librosa.load(a.audio or A["file"], sr=SR); n = int(math.ceil(len(y) / SR * fps))
    sd = os.path.join(a.outdir, "stems"); st = {k: librosa.load(os.path.join(sd, f"{k}.wav"), sr=SR)[0] for k in ["drums", "vocals", "bass"] if os.path.exists(os.path.join(sd, f"{k}.wav"))}
    fit = lambda x: np.pad(x, (0, max(0, n - len(x))))[:n]
    C = {}
    for k, (lo, hi) in BANDS.items(): C[k] = norm01(attack_release(fit(band_env(y, fps, lo, hi)), fps, 0.02, 0.25 if k != "high" else 0.12))
    C["rms"] = norm01(attack_release(fit(rms_env(y, fps)), fps, 0.03, 0.35))
    if "drums" in st:
        C["kick"] = norm01(attack_release(fit(band_env(st["drums"], fps, 20, 150)), fps, 0.01, 0.15))
        C["snare_hat"] = norm01(attack_release(fit(band_env(st["drums"], fps, 2000, 11025)), fps, 0.01, 0.1))
        don = librosa.onset.onset_detect(y=st["drums"], sr=SR, units="time", backtrack=False)
        C["drum_onset_trig"] = impulses(don, n, fps, 0.15)
    if "vocals" in st: C["vocal"] = norm01(attack_release(fit(rms_env(st["vocals"], fps)), fps, 0.05, 0.4))
    if "bass" in st: C["bass_stem"] = norm01(attack_release(fit(rms_env(st["bass"], fps)), fps, 0.03, 0.3))
    C["beat_trig"] = impulses(A["beats"], n, fps, 0.2)
    C["downbeat_trig"] = impulses(A["downbeats"], n, fps, 0.4)
    acc = [(e["t"], e.get("strength", 1)) for s in A["segments"] for e in s["events"] if e["type"] == "accent"]
    C["accent_trig"] = impulses([t for t, _ in acc], n, fps, 0.3, [w for _, w in acc]) if acc else np.zeros(n)
    sec_t = [s["start"] for s in A["sections"][1:]]; C["section_trig"] = impulses(sec_t, n, fps, 1.0)
    names = list(C); M = np.vstack([C[k] for k in names]).T
    hdr = "frame,time," + ",".join(names)
    np.savetxt(os.path.join(cd, "curves.csv"), np.column_stack([np.arange(n), np.arange(n) / fps, M]), delimiter=",", header=hdr, comments="", fmt=["%d", "%.4f"] + ["%.4f"] * len(names))
    for k in names: open(os.path.join(cd, "schedules", f"{k}.txt"), "w").write(sched(C[k]))
    shots, md = [], [f"# ComfyUI / Wan shot list — {os.path.basename(A['file'])}\n", f"fps {fps}, Wan clip length 4n+1 ≤ {a.max_frames} frames. Curves: comfy/curves.csv, schedules/*.txt (global frame index), segments/<id>/ (clip-relative).\n"]
    for i, s in enumerate(A["segments"]):
        f0, f1 = int(round(s["start"] * fps)), int(round(s["end"] * fps)); nf = max(1, f1 - f0)
        clips, cur, mx = [], f0, a.max_frames
        db = [int(round(t * fps)) for t in A["downbeats"] if f0 < t * fps < f1]
        while cur < f1:
            rem = f1 - cur
            if rem <= mx: end = f1
            else:   # cut on the last downbeat that keeps this clip <= max and leaves >= 17 frames for the rest
                cand = [d for d in db if cur + 17 <= d <= cur + mx and f1 - d >= 17]
                end = max(cand) if cand else cur + min(mx, rem - 17)
            real = end - cur; wl = min(4 * math.ceil(max(real - 1, 0) / 4) + 1, mx)
            clips.append(dict(frame_start=cur, frame_end=end, frames=real, wan_frames=wl, extra_frames=wl - real)); cur = end
        beats = [int(round(t * fps)) - f0 for t in A["beats"] if s["start"] <= t < s["end"]]
        downs = [int(round(t * fps)) - f0 for t in A["downbeats"] if s["start"] <= t < s["end"]]
        accs = [int(round(e["t"] * fps)) - f0 for e in s["events"] if e["type"] == "accent"]
        ps = s.get("pace_score", s["camera_motion_hint"].get("pace_score", 0.5))
        ms = (round(max(0.05, 0.1 + 0.8 * ps - 0.1), 2), round(min(1, 0.1 + 0.8 * ps + 0.1), 2))
        dn = (round(0.35 + 0.35 * ps, 2), round(0.5 + 0.35 * ps, 2))
        sid = f"{i:03d}"; segdir = os.path.join(cd, "segments", sid); os.makedirs(segdir, exist_ok=True)
        sl = M[f0:f1]; np.savetxt(os.path.join(segdir, "curves.csv"), np.column_stack([np.arange(len(sl)), sl]), delimiter=",", header="frame," + ",".join(names), comments="", fmt=["%d"] + ["%.4f"] * len(names))
        for k in names: open(os.path.join(segdir, f"{k}.txt"), "w").write(sched(C[k][f0:f1]))
        shot = dict(id=sid, type=s["type"], start=s["start"], end=s["end"], frame_start=f0, frame_end=f1, frames=nf, clips=clips,
                    lyric=s.get("text"), section=s["section"], pace=s["camera_motion_hint"]["pace"], pace_score=ps, energy_trend=s["energy"]["trend"],
                    arrangement=s["arrangement"], wan_prompt=prompt_for(s), motion_strength_range=ms, denoise_range=dn,
                    beat_frames=beats, downbeat_frames=downs, accent_frames=accs, cut_frames=[int(round(t * fps)) - f0 for t in s["suggested_cuts"]],
                    camera_hint=s["camera_motion_hint"]["suggestions"])
        shots.append(shot)
        md.append(f"## {sid} | {s['start']:.2f}-{s['end']:.2f}s | frames {f0}-{f1} ({nf}) | {s['section']} | " + (f'"{s["text"]}"' if s.get("text") else "[instrumental]"))
        md.append("- Clips: " + "; ".join(f"{c['frame_start']}-{c['frame_end']} → Wan {c['wan_frames']}f" + (f" (generate {c['wan_frames']}, trim {c['extra_frames']})" if c['extra_frames'] else "") for c in clips))
        md.append(f"- Prompt: *{shot['wan_prompt']}*")
        md.append(f"- Pace {shot['pace']} ({ps}), trend {shot['energy_trend']}, {shot['arrangement']} | motion {ms[0]}-{ms[1]}, denoise {dn[0]}-{dn[1]}")
        md.append(f"- Beat frames: {beats} | downbeats: {downs} | accents: {accs}\n")
    json.dump(dict(fps=fps, n_frames=n, curves=names, shots=shots), open(os.path.join(cd, "shots.json"), "w"), indent=1, ensure_ascii=False)
    open(os.path.join(cd, "shots.md"), "w").write("\n".join(md))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    show = [k for k in ["sub_low", "low_mid", "mid", "high", "rms", "kick", "snare_hat", "vocal", "beat_trig", "accent_trig"] if k in C]
    m = min(n, 30 * fps); t = np.arange(m) / fps
    fig, ax = plt.subplots(len(show), 1, figsize=(16, 1.3 * len(show)), sharex=True)
    for x, k in zip(ax, show):
        x.plot(t, C[k][:m], lw=0.9); x.set_ylabel(k, rotation=0, ha="right", fontsize=8); x.set_ylim(0, 1.05)
        for d in A["downbeats"]:
            if d < 30: x.axvline(d, color="gray", lw=0.4, alpha=0.5)
        for s in A["segments"]:
            if s["start"] < 30: x.axvline(s["start"], color="red", lw=0.6, alpha=0.6)
    ax[-1].set_xlabel(f"time (s) @ {fps} fps  — grey: downbeats, red: segment starts"); plt.tight_layout(); plt.savefig(os.path.join(cd, "bands_first30s.png"), dpi=100)
    print(f"wrote {cd}: {n} frames, {len(names)} curves, {len(shots)} shots")

if __name__ == "__main__": main()
