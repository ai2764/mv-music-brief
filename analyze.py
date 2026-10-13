#!/usr/bin/env python
"""Music -> time-aligned, LLM-readable analysis for music-video planning.
Usage: python analyze.py <audio> -o <outdir> [--no-stems] [--no-lyrics] [--no-clap] [--no-essentia] [--whisper-model small] [--language en] [--alm]
Primary unit: lyric lines (vocal songs) / musical phrases (instrumental parts)."""
import argparse, json, os, sys, time, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import platform_setup; platform_setup.setup()
import librosa

HERE = os.path.dirname(os.path.abspath(__file__)); MODELS = os.environ.get("MVBRIEF_MODELS", os.path.join(HERE, "models"))
def _dev():
    if os.environ.get("MVBRIEF_DEVICE"): return os.environ["MVBRIEF_DEVICE"]
    try:
        import torch; return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception: return "cpu"
DEV = _dev()
SR = 22050; HOP = 512
STATUS = {}

MOOD_VOCAB = ["sad", "euphoric", "epic", "dreamy", "tense", "romantic", "aggressive", "calm", "nostalgic",
              "happy", "melancholic", "energetic", "mysterious", "playful", "dark", "uplifting", "hopeful", "angry", "peaceful", "triumphant"]
GENRE_VOCAB = ["pop", "rock", "hip hop", "electronic dance", "jazz", "classical", "folk", "country", "r&b", "metal", "ambient", "cinematic orchestral", "blues", "lo-fi"]
INSTR_VOCAB = ["acoustic guitar", "electric guitar", "piano", "synthesizer", "strings", "drums", "bass", "female vocals", "male vocals", "brass", "banjo", "fiddle", "choir"]
MOOD_PAIRS = [("sad", "happy"), ("euphoric", "gloomy"), ("epic", "intimate"), ("dreamy", "grounded"), ("tense", "relaxed"),
              ("romantic", "cold"), ("aggressive", "gentle"), ("calm", "energetic"), ("nostalgic", "futuristic"), ("dark", "bright"),
              ("playful", "serious"), ("mysterious", "straightforward"), ("hopeful", "hopeless"), ("triumphant", "defeated")]
STEMS = ["vocals", "drums", "bass", "other"]

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)
def r(x, n=3): return None if x is None else round(float(x), n)
def t2f(t): return int(np.clip(round(t * SR / HOP), 0, 10**9))

# ---------------- 1. low-level ----------------
HAVE_ESSENTIA = None
def essentia_ok():
    global HAVE_ESSENTIA
    if HAVE_ESSENTIA is None:
        if os.environ.get("MVBRIEF_NO_ESSENTIA"): HAVE_ESSENTIA = False
        else:
            try: import essentia.standard; HAVE_ESSENTIA = True
            except Exception: HAVE_ESSENTIA = False
    return HAVE_ESSENTIA

def beat_this_tempo(path):
    """optional: beat_this (CPJKU) neural beat tracker -> (bpm, beats, downbeats) or None"""
    try:
        from beat_this.inference import File2Beats
        beats, downs = File2Beats(device=os.environ.get("MVBRIEF_DEVICE", "cuda" if _dev() == "cuda" else "cpu"), dbn=False)(path)
        if len(beats) < 8: return None
        return float(60 / np.median(np.diff(beats))), np.asarray(beats), np.asarray(downs)
    except Exception: return None

AUDIO_PATH = {}
def lowlevel(y):
    tempo, beats = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=SR, hop_length=HOP, units="time")
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP)[0]
    n = min(len(rms), len(cent), len(onset_env)); rms, cent, onset_env = rms[:n], cent[:n], onset_env[:n]
    # downbeats: pick 4/4 phase whose beats carry most low-freq onset + RMS
    bf = np.array([min(t2f(b), n - 1) for b in beats], int)
    sc = [np.mean(onset_env[bf[p::4]]) + np.mean(rms[bf[p::4]]) / (rms.max() + 1e-9) * onset_env.mean() for p in range(4)] if len(bf) >= 8 else [0]
    down = beats[int(np.argmax(sc))::4]
    key = None
    try:
        if not essentia_ok(): raise ImportError("essentia not installed/disabled")
        import essentia.standard as es
        k, scale, strength = es.KeyExtractor(profileType="edma")(y.astype(np.float32))  # sample-rate agnostic enough for key
        key = {"key": k, "mode": scale, "strength": r(strength), "method": "essentia KeyExtractor(edma)"}
        STATUS["essentia_key"] = "ok"
    except Exception as e:
        STATUS["essentia_key"] = f"skipped: {e}"
    if key is None:
        chroma = librosa.feature.chroma_cqt(y=y, sr=SR).mean(1)
        maj = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]); mi = np.array([6.33,2.68,3.52,5.38,2.6,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
        names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]; best = max(((np.corrcoef(np.roll(p, i), chroma)[0,1], names[i], m) for i in range(12) for p, m in ((maj,"major"),(mi,"minor"))))
        key = {"key": best[1], "mode": best[2], "strength": r(best[0]), "method": "librosa Krumhansl"}
    bpm_ess = None
    try:
        if not essentia_ok(): raise ImportError("essentia not available")
        import essentia.standard as es
        bpm_ess = float(es.RhythmExtractor2013(method="multifeature")(librosa.resample(y, orig_sr=SR, target_sr=44100).astype(np.float32))[0])
    except Exception: pass
    tempo_raw = tempo
    if bpm_ess is None and AUDIO_PATH.get("p"):
        bt = beat_this_tempo(AUDIO_PATH["p"])
        if bt: bpm_ess = bt[0]; STATUS["beat_this"] = "ok (used as 2nd tempo estimate)"
    tempo, why = reconcile_bpm(tempo, bpm_ess, onset_env)
    if abs(tempo - tempo_raw) / tempo_raw > 0.03:
        _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=SR, hop_length=HOP, units="time", start_bpm=tempo, tightness=400)
        bf = np.array([min(t2f(b), n - 1) for b in beats], int)
        sc = [np.mean(onset_env[bf[p::4]]) + np.mean(rms[bf[p::4]]) / (rms.max() + 1e-9) * onset_env.mean() for p in range(4)] if len(bf) >= 8 else [0]
        down = beats[int(np.argmax(sc))::4]
    return dict(tempo=tempo, tempo_librosa=tempo_raw, tempo_essentia=bpm_ess, tempo_choice=why, beats=np.asarray(beats), downbeats=np.asarray(down), onsets=np.asarray(onsets),
                onset_env=onset_env, rms=rms, centroid=cent, key=key)

def reconcile_bpm(lib, ess, onset_env):
    """candidates = both estimates and their half/double; score by onset-envelope autocorrelation (+ metric-level
    support from the double lag) and a mild log-normal prior around 110 BPM; bonus when both methods agree."""
    ac = librosa.autocorrelate(onset_env - onset_env.mean()); ac = ac / (ac[0] + 1e-9)
    def acs(b):
        i = int(round(60 * SR / HOP / b)); return float(ac[i]) if 1 < i < len(ac) else -1
    cands = {}
    for name, b in [("librosa", lib), ("essentia/beat_this", ess)]:
        if not b: continue
        for f, tag in [(1, ""), (2, " x2"), (0.5, " /2")]:
            if 60 <= b * f <= 190: cands[name + tag] = b * f
    best, bs, table = None, -9, {}
    for k, b in cands.items():
        agree = any(abs(b - o) / o < 0.04 for kk, o in cands.items() if kk.split()[0] != k.split()[0])
        score = acs(b) + 0.5 * acs(b / 2) + 0.15 * np.exp(-0.5 * (np.log2(b / 110) / 0.5) ** 2) + (0.1 if agree else 0)
        table[k] = [round(b, 1), round(score, 3)]
        if score > bs: best, bs = k, score
    return float(cands[best]), dict(chosen=best, candidates_bpm_score=table)

# ---------------- 2. structure ----------------
def structure(path, y, ll, dur):
    try:
        import allin1
        res = allin1.analyze(path, device=DEV)
        STATUS["structure"] = "allin1"
        return [dict(start=s.start, end=s.end, label=s.label) for s in res.segments]
    except Exception as e:
        STATUS["allin1"] = f"skipped: {type(e).__name__}: {str(e)[:120]}"
    STATUS["structure"] = "librosa recurrence + agglomerative (fallback)"
    beats = ll["beats"]; bf = librosa.time_to_frames(beats, sr=SR, hop_length=HOP)
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP); mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=13)
    feat = np.vstack([librosa.util.normalize(librosa.util.sync(chroma, bf), axis=0), librosa.util.normalize(librosa.util.sync(mfcc, bf), axis=1)])
    k = int(np.clip(round(dur / 12), 2, 24)); k = min(k, feat.shape[1] - 1)
    bounds = librosa.segment.agglomerative(feat, k)
    bt = librosa.frames_to_time(librosa.util.fix_frames(bf, x_min=0), sr=SR, hop_length=HOP)
    times = sorted(set([0.0] + [float(bt[b]) for b in bounds if 0 < b < len(bt)] + [dur]))
    # snap to downbeats, merge tiny
    db = ll["downbeats"]; snapped = [0.0]
    for t in times[1:-1]:
        s = float(db[np.argmin(abs(db - t))]) if len(db) else t
        if s - snapped[-1] >= 6: snapped.append(s)
    if dur - snapped[-1] < 6 and len(snapped) > 1: snapped.pop()
    snapped.append(dur)
    segs = [dict(start=a, end=b) for a, b in zip(snapped[:-1], snapped[1:])]
    # cluster sections by timbre/chroma to find repeats, label heuristically
    vecs = []
    for s in segs:
        a, b = t2f(s["start"]), max(t2f(s["end"]), t2f(s["start"]) + 1)
        vecs.append(np.concatenate([chroma[:, a:b].mean(1), mfcc[1:, a:b].mean(1) / 50]))
        s["energy"] = float(ll["rms"][a:b].mean())
    vecs = np.array(vecs); groups = list(range(len(segs)))
    V = (vecs - vecs.mean(0)) / (vecs.std(0) + 1e-9)          # standardise so similarity is relative to this song
    S = np.corrcoef(V) if len(segs) > 1 else np.ones((1, 1))
    off = S[~np.eye(len(segs), dtype=bool)] if len(segs) > 1 else np.array([0.0])
    thr = max(0.3, off.mean() + 0.75 * off.std())
    for i in range(len(segs)):
        for j in range(i):
            if S[i, j] >= thr: groups[i] = groups[j]; break
    for s, g in zip(segs, groups): s["group"] = "ABCDEFGHIJKLMNOP"[g % 16]
    en = np.array([s["energy"] for s in segs]); rank = en.argsort().argsort() / max(len(segs) - 1, 1)  # 0=quietest,1=loudest
    cnt = {g: groups.count(g) for g in groups}
    gE = {g: np.mean([en[i] for i in range(len(segs)) if groups[i] == g]) for g in set(groups)}
    rep_groups = [g for g in gE if cnt[g] > 1]
    chorus_g = max(rep_groups, key=lambda g: gE[g]) if rep_groups else None
    for i, (s, g) in enumerate(zip(segs, groups)):
        if i == 0 and rank[i] <= 0.5 and len(segs) > 2: lab = "intro"
        elif i == len(segs) - 1 and rank[i] <= 0.5 and len(segs) > 2: lab = "outro"
        elif g == chorus_g and rank[i] >= 0.5: lab = "chorus"
        elif rank[i] >= 0.8 and cnt[g] == 1: lab = "climax/bridge"
        elif cnt[g] > 1: lab = "verse"
        else: lab = "bridge" if 0 < i < len(segs) - 1 else "verse"
        s["label"] = lab + "?"; s["energy_rank"] = r(rank[i], 2); s["repeat_count"] = cnt[g]   # '?' = heuristic guess
    return segs

# ---------------- 3. stems ----------------
def stems(path, outdir, dur):
    import torch
    from demucs.pretrained import get_model; from demucs.apply import apply_model; from demucs.audio import AudioFile
    torch.set_num_threads(os.cpu_count())
    m = get_model("htdemucs"); m.eval()
    try:   # decode with librosa/soundfile (no ffprobe needed; works on Windows without ffmpeg for mp3/wav/flac/ogg)
        yy, _ = librosa.load(path, sr=m.samplerate, mono=False)
        if yy.ndim == 1: yy = np.stack([yy, yy])
        wav = torch.from_numpy(yy[:m.audio_channels].astype(np.float32))
    except Exception:
        wav = AudioFile(path).read(streams=0, samplerate=m.samplerate, channels=m.audio_channels)
    ref = wav.mean(0); wav_n = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad(): out = apply_model(m, wav_n[None], device=DEV, split=True, overlap=0.25, progress=False)[0]
    out = out * ref.std() + ref.mean()
    sdir = os.path.join(outdir, "stems"); os.makedirs(sdir, exist_ok=True)
    import soundfile as sf
    res = {}
    for name, src in zip(m.sources, out):
        mono = librosa.resample(src.mean(0).numpy(), orig_sr=m.samplerate, target_sr=SR)
        sf.write(os.path.join(sdir, f"{name}.wav"), src.numpy().T, m.samplerate)
        res[name] = librosa.feature.rms(y=mono, hop_length=HOP)[0]
    return res, sdir

def stem_activity(stem_rms, mix_rms):
    """frame-wise boolean activity per stem: louder than -28 dB re. stem p95 and -35 dB re. mix p95."""
    act = {}; mp = np.percentile(mix_rms, 95) + 1e-9
    for k, v in stem_rms.items():
        p = np.percentile(v, 95) + 1e-9
        act[k] = (20 * np.log10(v / p + 1e-9) > -28) & (20 * np.log10(v / mp + 1e-9) > -35)
    return act

# ---------------- 4. lyrics ----------------
def lyrics(audio_for_asr, model_size, language):
    from faster_whisper import WhisperModel
    m = WhisperModel(model_size, device=DEV, compute_type="float16" if DEV == "cuda" else "int8", cpu_threads=os.cpu_count())
    audio_for_asr = librosa.load(audio_for_asr, sr=16000, mono=True)[0] if isinstance(audio_for_asr, str) else audio_for_asr
    prompt = None
    if language is None:
        lang, lp, _ = m.detect_language(audio_for_asr[: 16000 * 120] if len(audio_for_asr) > 16000 * 30 else audio_for_asr)
        if lp > 0.5: language = lang
    if language == "zh": prompt = "以下是一首中文歌曲的歌词，使用简体中文。"
    segs, info = m.transcribe(audio_for_asr, language=language, initial_prompt=prompt, word_timestamps=True, vad_filter=True, beam_size=5,
                              condition_on_previous_text=False)
    lines = []
    for s in segs:
        txt = s.text.strip()
        if language == "zh" and T2S:
            for x in (s.words or []): x.word = T2S.convert(x.word)
        # sung vocals routinely get no_speech_prob 0.7-0.9, so only drop clearly bad decodes
        if not txt or s.avg_logprob < -1.0 or (s.no_speech_prob > 0.95 and s.avg_logprob < -0.6): continue
        w = [dict(word=x.word.strip(), start=r(x.start), end=r(x.end), p=r(x.probability, 2)) for x in (s.words or []) if x.word.strip()]
        if not w: continue
        for x in w:   # whisper sometimes stretches a word across an instrumental gap
            if x["end"] - x["start"] > 1.5:
                if x is w[0]: x["start"] = r(x["end"] - 0.8)
                else: x["end"] = r(x["start"] + 1.2)
        for chunk in split_lines(w):
            lines.append(dict(start=float(chunk[0]["start"]), end=float(chunk[-1]["end"]), text=join_words([x["word"] for x in chunk]),
                              no_speech_prob=r(s.no_speech_prob), avg_logprob=r(s.avg_logprob), words=chunk))
    return lines, dict(language=info.language, language_prob=r(info.language_probability))

try:
    from opencc import OpenCC; T2S = OpenCC("t2s")    # whisper drifts into traditional script mid-song
except Exception: T2S = None

def align_lyrics(lines, ref_text):
    """Fuzzy-align user lyrics to whisper output at character level (difflib over the whole song, in order),
    so each reference line inherits timestamps of the whisper characters it matches; unmatched lines are interpolated."""
    import difflib, re
    norm = lambda c: re.sub(r"[\W_]", "", c.lower())
    if T2S: ref_text = T2S.convert(ref_text) if any(is_cjk(l["text"]) for l in lines) else ref_text
    ref_lines = [l.strip() for l in ref_text.splitlines() if norm(l.strip())]
    # whisper char stream with per-char times (CJK words split evenly)
    hc, ht = [], []
    for L in lines:
        for w in L["words"]:
            cs = [c for c in w["word"] if norm(c)]
            for k, c in enumerate(cs):
                d = (w["end"] - w["start"]) / max(len(cs), 1); ht.append((w["start"] + k * d, w["start"] + (k + 1) * d)); hc.append(norm(c))
    rc, owner = [], []
    for i, l in enumerate(ref_lines):
        for c in l:
            if norm(c): rc.append(norm(c)); owner.append(i)
    sm = difflib.SequenceMatcher(None, rc, hc, autojunk=False)
    times = [[] for _ in ref_lines]
    for a, b, n in sm.get_matching_blocks():
        for k in range(n): times[owner[a + k]].append(ht[b + k])
    out, matched = [], 0
    for i, l in enumerate(ref_lines):
        nchar = sum(1 for c in l if norm(c)); tt = times[i]
        if len(tt) >= max(1, 0.25 * nchar):
            tt = sorted(tt); s0, e0 = tt[0][0], tt[-1][1]; matched += 1
            out.append(dict(start=float(s0), end=float(e0), text=l, match=r(len(times[i]) / nchar, 2), words=[]))
        else: out.append(dict(start=None, end=None, text=l, match=r(len(tt) / max(nchar, 1), 2), words=[]))
    # interpolate unmatched between neighbours
    for i, L in enumerate(out):
        if L["start"] is None:
            prev = next((o["end"] for o in reversed(out[:i]) if o["start"] is not None), 0.0)
            nxt = next((o["start"] for o in out[i + 1:] if o["start"] is not None), prev + 3)
            L["start"], L["end"], L["interpolated"] = prev, max(prev + 0.5, nxt), True
    for a, b in zip(out, out[1:]):   # monotonic, no overlap
        if b["start"] < a["end"]: b["start"] = a["end"]
        if b["end"] <= b["start"]: b["end"] = b["start"] + 0.5
    return out, f"user lyrics aligned to whisper timing: {matched}/{len(ref_lines)} lines matched (rest interpolated)"

def is_cjk(t): return any("\u3040" <= c <= "\u9fff" or "\uac00" <= c <= "\ud7af" for c in t)
def join_words(ws):
    out = ""
    for w in ws:
        out += w if (out and (is_cjk(w[:1]) or is_cjk(out[-1:]))) or not out else " " + w
    return out

def split_lines(words, gap=0.5, max_len=8.0, min_words=2):
    """split a whisper segment into lyric lines at pauses >= gap s or sentence punctuation (or when too long)."""
    out, cur = [], []
    for i, w in enumerate(words):
        if cur:
            pause = w["start"] - cur[-1]["end"]; punct = cur[-1]["word"][-1:] in ".,!?;:"
            too_long = w["end"] - cur[0]["start"] > max_len
            if pause >= 2.0 and len(cur) == 1 and len(words) > i:   # orphan pickup word before a gap -> belongs to next line
                cur = [dict(cur[0], start=w["start"], end=w["start"], moved_from=cur[0]["start"])]
            elif pause >= 2.0 or (len(cur) >= min_words and (pause >= gap or punct or too_long)):
                out.append(cur); cur = []
        cur.append(w)
    if cur:
        if out and len(cur) < min_words and cur[0]["start"] - out[-1][-1]["end"] < 2.0: out[-1] += cur
        else: out.append(cur)
    return out

# ---------------- 5. emotion ----------------
def essentia_emotion(path):
    import essentia.standard as es
    a = es.MonoLoader(filename=path, sampleRate=16000, resampleQuality=4)()
    emb = es.TensorflowPredictMusiCNN(graphFilename=f"{MODELS}/msd-musicnn-1.pb", output="model/dense/BiasAdd")(a)
    hop = len(a) / 16000 / max(len(emb), 1)  # seconds per embedding (~1.5s hop, 3s patch)
    times = np.arange(len(emb)) * hop + 1.5
    va = es.TensorflowPredict2D(graphFilename=f"{MODELS}/deam-msd-musicnn-2.pb", output="model/Identity")(emb)
    out = {"times": times, "valence": (va[:, 0] - 5) / 4, "arousal": (va[:, 1] - 5) / 4}   # DEAM 1..9 -> -1..1
    for h in ["mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed", "mood_party", "mood_electronic", "mood_acoustic", "danceability"]:
        meta = json.load(open(f"{MODELS}/{h}-msd-musicnn-1.json", encoding="utf-8")); pos = [c for c in meta["classes"] if not c.startswith("non") and not c.startswith("not")][0]
        p = es.TensorflowPredict2D(graphFilename=f"{MODELS}/{h}-msd-musicnn-1.pb", output="model/Softmax")(emb)
        out[pos] = p[:, meta["classes"].index(pos)]
    return out

class Clap:
    def __init__(self):
        import torch; from transformers import ClapModel, ClapProcessor
        self.torch = torch; self.m = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(DEV).eval(); self.p = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.vocab = {"mood": MOOD_VOCAB, "genre": GENRE_VOCAB, "instrument": INSTR_VOCAB}
        prompts = {k: [f"{'This is a' if k!='instrument' else 'Music featuring'} {w} {'song' if k=='mood' else 'music' if k=='genre' else ''}".strip() for w in v] for k, v in self.vocab.items() if k != "mood"}
        with self.torch.no_grad():
            enc = lambda v: self._norm(self._feat(self.m.get_text_features(**self.p(text=v, return_tensors="pt", padding=True).to(DEV))))
            self.text = {k: enc(v) for k, v in prompts.items()}
            cp = os.path.join(MODELS, "clap_calib.json")
            self.calib = {k: np.array(v) for k, v in json.load(open(cp, encoding="utf-8")).items()} if os.path.exists(cp) else None
            self.pos = enc([f"a {a} song" for a, b in MOOD_PAIRS]); self.neg = enc([f"a {b} song" for a, b in MOOD_PAIRS])
    def _feat(self, o): return o if isinstance(o, self.torch.Tensor) else getattr(o, "pooler_output", o[0])
    def _norm(self, x): return x / x.norm(dim=-1, keepdim=True)
    def pair_diff(self, y48):
        with self.torch.no_grad():
            a = self._norm(self._feat(self.m.get_audio_features(**self.p(audio=[y48[:480000]], sampling_rate=48000, return_tensors="pt").to(DEV))))
        return (a @ self.pos.T - a @ self.neg.T)[0].cpu().numpy()
    def tags(self, y48, top=5):
        if len(y48) < 48000 * 2: y48 = np.pad(y48, (0, 48000 * 2 - len(y48)))
        with self.torch.no_grad():
            a = self._norm(self._feat(self.m.get_audio_features(**self.p(audio=[y48[:48000 * 10]], sampling_rate=48000, return_tensors="pt").to(DEV))))
        # mood: 2-way softmax per contrasting pair -> P(pole A vs pole B); report the winning pole of each pair
        d = (a @ self.pos.T - a @ self.neg.T)[0].cpu().numpy()
        if self.calib is not None:   # z-score each pair against a reference set of varied music, then sigmoid
            pa = 1 / (1 + np.exp(-1.7 * (d - self.calib["mean"]) / self.calib["std"]))
        else: pa = 1 / (1 + np.exp(-d * 100))
        moods = [[A, r(p, 2)] if p >= 0.5 else [B, r(1 - p, 2)] for (A, B), p in zip(MOOD_PAIRS, pa)]
        P = dict(zip(MOOD_PAIRS, pa))
        def pp(A_, B_): return float(P[(A_, B_)])   # P(A over B)
        val = np.mean([1 - pp("sad", "happy"), pp("euphoric", "gloomy"), pp("hopeful", "hopeless"), 1 - pp("dark", "bright")]) * 2 - 1
        aro = np.mean([1 - pp("calm", "energetic"), pp("aggressive", "gentle"), pp("tense", "relaxed"), pp("epic", "intimate")]) * 2 - 1
        res = {"va_clap": [r(val, 2), r(aro, 2)], "mood": sorted(moods, key=lambda x: -x[1])[:top + 3], "mood_pairs": {f"{A}|{B}": r(p, 2) for (A, B), p in zip(MOOD_PAIRS, pa)}}
        for k, t in self.text.items():
            pr = (a @ t.T * 100).softmax(-1)[0].cpu().numpy()
            idx = np.argsort(-pr)[:top]; res[k] = [[self.vocab[k][i], r(pr[i])] for i in idx]
        return res

# ---------------- segmenting by lyric lines / phrases ----------------
def relabel_by_lyrics(sections, lines):
    """sections whose lyrics repeat elsewhere in the song (and are not among the quietest) are likely choruses."""
    import re
    norm = lambda t: re.sub(r"[\W_]", "", t.lower())
    texts = [(norm(l["text"]), l["start"]) for l in lines if len(norm(l["text"])) >= (4 if is_cjk(l["text"]) else 10)]
    for s in sections:
        mine = [t for t, st in texts if s["start"] <= st < s["end"]]
        others = [t for t, st in texts if not (s["start"] <= st < s["end"])]
        s["repeated_lyrics"] = sum(any(t == o or (len(t) >= 4 and (t in o or o in t)) for o in others) for t in mine)
        if s["repeated_lyrics"] >= 2 and s["repeated_lyrics"] >= 0.4 * max(len(mine), 1) and s.get("energy_rank", 1) >= 0.4 and not s["label"].startswith("chorus"):
            s["label"] = "chorus?"

def snap(t, grid):
    return float(grid[np.argmin(abs(grid - t))]) if len(grid) else t

def build_segments(lines, sections, ll, dur, vocal_ratio):
    beats, db = ll["beats"], ll["downbeats"]
    spb = 60 / max(ll["tempo"], 1)
    segs = []
    use_lyrics = len(lines) >= 2 and vocal_ratio > 0.05
    if use_lyrics:
        for L in lines:
            s, e = snap(L["start"], beats), snap(L["end"], beats)
            if e <= s: e = s + spb
            segs.append(dict(type="lyric", start=s, end=e, raw_start=r(L["start"]), raw_end=r(L["end"]), text=L["text"], words=L["words"], match=L.get("match")))
        # fix overlaps
        segs.sort(key=lambda x: x["start"])
        for a, b in zip(segs, segs[1:]):
            if a["end"] > b["start"]: a["end"] = b["start"]
        segs = [s for s in segs if s["end"] - s["start"] > 0.2]
    # fill gaps (instrumental) with phrases
    filled, cur = [], 0.0
    for s in segs + [dict(type="_end", start=dur, end=dur)]:
        if s["start"] - cur >= max(spb * 2, 1.0):
            filled += phrase_split(cur, s["start"], sections, db, spb)
        elif filled and s["start"] > cur and s["type"] != "_end":
            s["start"] = cur
        elif s["type"] == "_end" and filled and dur > cur: filled[-1]["end"] = dur
        if s["type"] != "_end": filled.append(s); cur = s["end"]
    if filled and filled[-1]["end"] < dur - 0.05:
        filled += phrase_split(filled[-1]["end"], dur, sections, db, spb)
    return filled, ("lyric_lines+phrases" if use_lyrics else "phrases (instrumental / no reliable lyrics)")

def phrase_split(a, b, sections, db, spb):
    """split instrumental span at section boundaries, then into 4-bar (or 8-bar if slow-changing) phrases on downbeats."""
    cuts = [a] + [s["start"] for s in sections if a + 1 < s["start"] < b - 1] + [b]
    out = []
    bar = spb * 4
    for x, y in zip(cuts[:-1], cuts[1:]):
        dbs = [d for d in db if x + 0.5 < d < y - 0.5]
        pts = [x]
        for d in dbs:
            if d - pts[-1] >= 4 * bar - spb: pts.append(float(d))
        if y - pts[-1] < 2 * bar and len(pts) > 1: pts.pop()
        pts.append(y)
        for p, q in zip(pts[:-1], pts[1:]):
            out.append(dict(type="instrumental" if not out or True else "", start=float(p), end=float(q), text=None))
    return out

# ---------------- per-segment descriptors ----------------
def describe(seg, ll, sections, act, emo, clap, y, dur, curves):
    a, b = seg["start"], seg["end"]; fa, fb = t2f(a), max(t2f(b), t2f(a) + 2)
    E, C, O = curves["energy"], curves["bright"], ll["onset_env"]
    e = E[fa:fb]; L = b - a
    sec = max(sections, key=lambda s: min(b, s["end"]) - max(a, s["start"]))
    beats_in = ll["beats"][(ll["beats"] >= a - 0.02) & (ll["beats"] < b)]
    loc_bpm = 60 / np.median(np.diff(beats_in)) if len(beats_in) >= 3 else ll["tempo"]
    ons = ll["onsets"][(ll["onsets"] >= a) & (ll["onsets"] < b)]
    # trend
    n = len(e); thirds = [e[: n // 3].mean(), e[n // 3: 2 * n // 3].mean(), e[2 * n // 3:].mean()] if n >= 6 else [e.mean()] * 3
    slope = thirds[2] - thirds[0]; gmax = np.percentile(E, 95)
    if e.max() >= 0.92 * gmax and thirds[1] >= max(thirds[0], thirds[2]) - 0.02 and e.mean() > 0.6: trend = "peak"
    elif slope > 0.08: trend = "rising"
    elif slope < -0.08: trend = "falling"
    elif e.mean() > 0.75: trend = "peak"
    else: trend = "steady"
    # stems
    stem_frac = {k: r(v[fa:fb].mean(), 2) for k, v in act.items()} if act else {}
    on = [k for k, v in stem_frac.items() if v >= 0.3]
    if not act: arrangement = "unknown (no stems)"
    elif on == ["vocals"]: arrangement = "vocal-only / a cappella"
    elif "vocals" in on and len(on) == 2 and "drums" not in on: arrangement = "sparse: vocals + " + [x for x in on if x != "vocals"][0]
    elif "vocals" in on and "drums" not in on: arrangement = "vocals over band without drums"
    elif set(on) >= {"vocals", "drums", "bass"}: arrangement = "full band with vocals"
    elif "vocals" not in on and "drums" in on: arrangement = "instrumental groove (" + "+".join(on) + ")"
    elif not on: arrangement = "near-silence"
    else: arrangement = "instrumental: " + "+".join(on)
    drum_density = None
    if curves.get("drum_onsets") is not None:
        d = curves["drum_onsets"]; drum_density = r(((d >= a) & (d < b)).sum() / L, 2)
    # events
    events = []
    for d in ll["downbeats"][(ll["downbeats"] >= a) & (ll["downbeats"] < b)]: events.append(dict(t=r(d), type="downbeat"))
    thr = np.percentile(O, 99); wait = max(1, int(60 / ll["tempo"] * SR / HOP))   # >= 1 beat between accents
    pk = librosa.util.peak_pick(O, pre_max=wait // 2, post_max=wait // 2, pre_avg=wait * 2, post_avg=wait * 2, delta=0.5, wait=wait)
    for p in pk:
        t = p * HOP / SR
        if a <= t < b and O[p] >= thr: events.append(dict(t=r(t), type="accent", strength=r(O[p] / O.max(), 2)))
    for t in curves["drops"]:
        if a - 0.1 <= t < b: events.append(dict(t=r(t), type="drop/energy-jump"))
    for s in sections:
        if a - 0.25 <= s["start"] < b - 0.25 and s["start"] > 0.1: events.append(dict(t=r(s["start"]), type="section-transition", to=s["label"]))
    events.sort(key=lambda x: x["t"])
    # mood
    mood = {}
    if emo is not None:
        msk = (emo["times"] >= a - 1.5) & (emo["times"] <= b + 1.5)
        if not msk.any(): msk = np.argmin(abs(emo["times"] - (a + b) / 2)) == np.arange(len(emo["times"]))
        mood["valence"] = r(emo["valence"][msk].mean(), 2); mood["arousal"] = r(emo["arousal"][msk].mean(), 2)
        mood["essentia_moods"] = {k: r(emo[k][msk].mean(), 2) for k in emo if k not in ("times", "valence", "arousal")}
    if clap is not None:
        c = (a + b) / 2; w0, w1 = max(0, min(a, c - 5)), min(dur, max(b, c + 5)) if L < 4 else b
        if L < 4: w0, w1 = max(0, c - 3), min(dur, c + 3)
        y48 = librosa.resample(y[int(w0 * SR): int(w1 * SR)], orig_sr=SR, target_sr=48000)
        mood["clap"] = clap.tags(y48)
        if emo is None:   # no essentia: valence/arousal estimated from CLAP contrast pairs
            mood["valence"], mood["arousal"] = mood["clap"]["va_clap"]; mood["va_source"] = "clap"
    cuts = cut_points(a, b, ll, E, events)
    out = dict(seg)
    out.update(start=r(a, 2), end=r(b, 2), duration=r(L, 2), section=sec["label"], section_index=sections.index(sec),
               local_bpm=r(loc_bpm, 1), beat_count=int(len(beats_in)), bars_approx=r(len(beats_in) / 4, 1),
               energy=dict(mean=r(e.mean(), 2), max=r(e.max(), 2), trend=trend, thirds=[r(x, 2) for x in thirds]),
               onset_density_per_s=r(len(ons) / L, 2), drum_hits_per_s=drum_density, brightness=r(C[fa:fb].mean(), 2),
               brightness_hz=r(ll["centroid"][fa:fb].mean(), 0), stems_active_fraction=stem_frac, arrangement=arrangement,
               events=events, mood=mood, suggested_cuts=cuts)
    out["camera_motion_hint"] = camera_hint(out, ll["tempo"])
    return out

def cut_points(a, b, ll, E, events):
    c = [e["t"] for e in events if e["type"] in ("downbeat", "drop/energy-jump", "section-transition")]
    # energy peak snapped to beat
    fa, fb = t2f(a), max(t2f(b), t2f(a) + 1)
    pk = (fa + int(np.argmax(E[fa:fb]))) * HOP / SR
    bb = ll["beats"][(ll["beats"] >= a) & (ll["beats"] < b)]
    if len(bb): c.append(r(snap(pk, bb)))
    return sorted(set(x for x in c if a + 0.05 < x < b - 0.05))

FEEL = {"bpm": 120.0}
def detect_feel(ll, act, curves, emo):
    """half-time feel: fast notated tempo but sparse drums/onsets (and calm arousal) -> the music is felt at half tempo."""
    bpm = ll["tempo"]; dur = len(ll["rms"]) * HOP / SR
    onset_rate = len(ll["onsets"]) / dur
    drum_frac = float(act["drums"].mean()) if act else 0.5
    hits = len(curves["drum_onsets"]) / dur if curves.get("drum_onsets") is not None else onset_rate
    arousal = float(np.mean(emo["arousal"])) if emo is not None else 0
    votes = [onset_rate < 1.2 * bpm / 60, drum_frac < 0.5, hits < bpm / 60, arousal < 0]
    half = bpm >= 110 and sum(votes) >= 3
    FEEL["bpm"] = bpm / 2 if half else bpm
    return dict(notated_bpm=r(bpm, 1), felt_bpm=r(FEEL["bpm"], 1), half_time=half, onset_rate=r(onset_rate, 2),
                drum_active_frac=r(drum_frac, 2), drum_hits_per_s=r(hits, 2), arousal=r(arousal, 2))

def pace_score(s, feel_bpm):
    """0..1 motion intensity from energy, onset (note) density and drum activity, relative to the felt pulse."""
    beat_rate = feel_bpm / 60
    dens = min(s["onset_density_per_s"] / (4 * beat_rate), 1)           # ~4 onsets per felt beat = busy
    drums = s["stems_active_fraction"].get("drums", 0.5) if s["stems_active_fraction"] else 0.5
    hits = min((s["drum_hits_per_s"] or 0) / (3 * beat_rate), 1)
    return 0.5 * s["energy"]["mean"] + 0.2 * dens + 0.15 * drums + 0.15 * hits

def camera_hint(s, tempo):
    en, tr = s["energy"]["mean"], s["energy"]["trend"]
    arr = s["arrangement"]; ev = {e["type"] for e in s["events"]}
    feel = FEEL["bpm"]; ps = pace_score(s, feel); s["pace_score"] = r(ps, 2)
    fast = ps >= 0.6; slow = ps < 0.4
    hints, cut_rate = [], None
    if "drop/energy-jump" in ev or (tr == "peak" and en > 0.65):
        hints += ["quick cuts on beats", "handheld energy", "whip pans / smash cuts on accents"]; cut_rate = "every 1-2 beats" if fast else "every bar"
    elif tr == "rising":
        hints += ["dolly-in or crane up across the line", "speed ramp into the next downbeat"]; cut_rate = "every 2 bars, tightening"
    elif tr == "falling":
        hints += ["pull-back / dolly-out", "slow fade or rack focus as energy releases"]; cut_rate = "every 2-4 bars"
    elif slow or (en < 0.35 and not fast):
        hints += ["static long take or slow push-in"]; cut_rate = "hold (one shot) or every 4 bars"
    else:
        hints += ["steady tracking / gimbal move" if fast else "slow lateral slide or gentle push-in"]; cut_rate = "every bar" if fast else "every 2 bars"
    if s["type"] == "lyric" and ("vocal-only" in arr or "sparse" in arr):
        hints.insert(0, "close-up lip-sync (sparse vocal arrangement)")
    elif s["type"] == "lyric":
        hints.append("lip-sync medium/close shot on key words" + (", cut-aways on downbeats" if en > 0.5 else ""))
    else:
        hints.append("instrumental: B-roll / performance inserts" + (" (drum hits)" if (s["drum_hits_per_s"] or 0) > 2 else "") + (", wide/atmospheric" if en < 0.4 else ""))
    if "section-transition" in ev: hints.append("mark the section change with a scene/look change")
    if slow and cut_rate in ("every bar", "every 2 bars"): cut_rate = "every 2-4 bars (slow)"
    return dict(suggestions=hints, cut_rate=cut_rate, pace="fast" if fast else "slow" if slow else "medium",
                pace_score=r(ps, 2), felt_bpm=r(feel, 1))

# ---------------- outputs ----------------
def write_md(A, path):
    o = []; g = A["global"]
    o.append(f"# Music analysis: {os.path.basename(A['file'])}\n")
    o.append(f"- Duration: {g['duration']:.1f}s | Tempo: {g['bpm']:.1f} BPM (librosa {g['bpm_librosa']}, 2nd estimate {g['bpm_essentia']}; chose {g['bpm_choice']['chosen']}) | Key: {g['key']['key']} {g['key']['mode']} ({g['key']['method']})")
    o.append(f"- Feel: {'HALF-TIME, felt at ' + str(g['feel']['felt_bpm']) + ' BPM' if g['feel']['half_time'] else 'felt at notated tempo'} (camera pace uses energy + onset density + drums, not raw BPM) | Lyrics source: {g['lyrics_source']}")
    o.append(f"- Vocal presence: {g['vocal_ratio']:.0%} of the track | Segmentation mode: **{A['segmentation_mode']}** | Lyrics language: {g.get('lyrics_language')}")
    if g.get("mood"): o.append(f"- Overall mood: valence {g['mood'].get('valence')}, arousal {g['mood'].get('arousal')} (-1..1); " + ", ".join(f"{k} {v}" for k, v in g["mood"].get("essentia_moods", {}).items()))
    o.append("- Fields: energy/brightness are 0-1 normalised to this track. Trend = within the segment. Times in seconds. Cuts are beat-snapped.\n")
    o.append("## Sections\n\n| # | time | label | energy | trend | arrangement | top moods |\n|---|---|---|---|---|---|---|")
    for i, s in enumerate(A["sections"]):
        o.append(f"| {i} | {s['start']:.1f}-{s['end']:.1f} | {s['label']} | {s['energy_mean']} | {s['trend']} | {s.get('arrangement','')} | {', '.join(m[0] for m in s.get('clap_mood', [])[:3])} |")
    o.append("\n## Shot-planning segments (lyric lines & instrumental phrases)\n")
    for i, s in enumerate(A["segments"]):
        head = (f'"{s["text"]}"' + (f" (align {s['match']})" if s.get("match") is not None and s["match"] < 0.6 else "")) if s["type"] == "lyric" else "[instrumental]"
        o.append(f"### {i:02d} | {s['start']:.2f}-{s['end']:.2f}s ({s['duration']:.1f}s) | {s['section']} | {head}")
        o.append(f"- Tempo: {s['local_bpm']} BPM, {s['beat_count']} beats (~{s['bars_approx']} bars) | onsets {s['onset_density_per_s']}/s, drum hits {s['drum_hits_per_s']}/s")
        o.append(f"- Energy: mean {s['energy']['mean']}, max {s['energy']['max']}, trend **{s['energy']['trend']}** (thirds {s['energy']['thirds']}) | brightness {s['brightness']} (~{s['brightness_hz']:.0f} Hz)")
        o.append(f"- Arrangement: {s['arrangement']} | stems active: " + ", ".join(f"{k} {v:.0%}" for k, v in s["stems_active_fraction"].items()))
        m = s["mood"]; parts = []
        if "valence" in m: parts.append(f"valence {m['valence']}, arousal {m['arousal']}")
        if "clap" in m: parts.append("mood " + ", ".join(f"{t} {p:.2f}" for t, p in m["clap"]["mood"][:4]) + " | instr " + ", ".join(t for t, _ in m["clap"]["instrument"][:3]))
        if parts: o.append("- Mood: " + " | ".join(parts))
        ev = [f"{e['t']:.2f} {e['type']}" + (f"→{e['to']}" if 'to' in e else "") for e in s["events"]]
        if ev: o.append("- Events: " + "; ".join(ev))
        o.append(f"- Camera: {'; '.join(s['camera_motion_hint']['suggestions'])} (pace {s['camera_motion_hint']['pace']}, cut {s['camera_motion_hint']['cut_rate']})")
        o.append(f"- Suggested cuts: {s['suggested_cuts']}")
        if s.get("description"): o.append(f"- Description: {s['description']}")
        o.append("")
    open(path, "w", encoding="utf-8").write("\n".join(o))

def plot(A, curves, ll, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = np.arange(len(curves["energy"])) * HOP / SR
    fig, ax = plt.subplots(2, 1, figsize=(16, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    cols = plt.cm.tab10.colors
    for i, s in enumerate(A["sections"]):
        ax[0].axvspan(s["start"], s["end"], color=cols[i % 10], alpha=0.12); ax[0].text(s["start"] + 0.3, 1.02, s["label"], fontsize=9)
    ax[0].plot(t, curves["energy"], label="energy (RMS, norm)", color="k", lw=1)
    ax[0].plot(t, curves["bright"], label="brightness", color="tab:orange", lw=0.7, alpha=0.7)
    for d in ll["downbeats"]: ax[0].axvline(d, color="gray", lw=0.4, alpha=0.5)
    for s in A["segments"]:
        for c in s["suggested_cuts"]: ax[0].plot(c, 0.02, "v", color="red", ms=4)
    ax[0].set_ylim(0, 1.1); ax[0].legend(loc="upper right"); ax[0].set_title(os.path.basename(A["file"]) + " — sections, energy, downbeats (grey), suggested cuts (red)")
    for i, s in enumerate(A["segments"]):
        c = "tab:blue" if s["type"] == "lyric" else "tab:green"
        ax[1].barh(0, s["end"] - s["start"], left=s["start"], color=c, alpha=0.6, edgecolor="k")
        ax[1].text(s["start"] + 0.1, 0, (s["text"] or "inst")[:18], fontsize=6, va="center")
    ax[1].set_yticks([]); ax[1].set_xlabel("time (s)"); ax[1].set_title("segments: blue=lyric line, green=instrumental phrase", fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()

def smooth01(x, w=9):
    x = np.convolve(x, np.ones(w) / w, mode="same"); lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("audio"); ap.add_argument("-o", "--outdir", default="out")
    for f in ["stems", "lyrics", "clap", "essentia"]: ap.add_argument(f"--no-{f}", action="store_true")
    ap.add_argument("--lyrics", default=None, help="text file with the real lyrics, one line per sung line; aligned to whisper timing")
    ap.add_argument("--whisper-model", default="small"); ap.add_argument("--language", default=None)
    ap.add_argument("--alm", action="store_true", help="optional audio-language-model descriptions (Qwen2-Audio); needs ~16GB+ free RAM or a GPU")
    args = ap.parse_args(); os.makedirs(args.outdir, exist_ok=True); T0 = time.time(); timing = {}
    STATUS["device"] = DEV
    if args.no_essentia: os.environ["MVBRIEF_NO_ESSENTIA"] = "1"
    def tick(k, t): timing[k] = round(time.time() - t, 1)
    t = time.time(); y, _ = librosa.load(args.audio, sr=SR, mono=True); dur = len(y) / SR; log(f"loaded {dur:.1f}s")
    AUDIO_PATH["p"] = args.audio; ll = lowlevel(y); tick("lowlevel", t); log("lowlevel done", ll["tempo"])
    t = time.time(); sections = structure(args.audio, y, ll, dur); tick("structure", t); log("structure", len(sections))
    curves = {"energy": smooth01(ll["rms"]), "bright": smooth01(ll["centroid"])}
    act, stem_rms, vocal_path = None, None, args.audio
    if not args.no_stems:
        t = time.time()
        try:
            stem_rms, sdir = stems(args.audio, args.outdir, dur); n = len(ll["rms"])
            stem_rms = {k: np.pad(v, (0, max(0, n - len(v))))[:n] for k, v in stem_rms.items()}
            act = stem_activity(stem_rms, ll["rms"]); vocal_path = os.path.join(sdir, "vocals.wav")
            dy, _ = librosa.load(os.path.join(sdir, "drums.wav"), sr=SR)
            curves["drum_onsets"] = librosa.onset.onset_detect(y=dy, sr=SR, hop_length=HOP, units="time")
            STATUS["demucs"] = "ok (htdemucs)"
        except Exception as e: STATUS["demucs"] = f"failed: {e}"
        tick("demucs", t); log("stems done")
    vocal_ratio = float(act["vocals"].mean()) if act else 0.5
    # drops: big energy jump between 2s windows, on a downbeat
    E = curves["energy"]; w = int(2 * SR / HOP); drops = []
    for d in ll["downbeats"]:
        f = t2f(d)
        if w < f < len(E) - w and E[f:f + w].mean() - E[f - w:f].mean() > 0.3: drops.append(float(d))
    curves["drops"] = drops
    lines, linfo = [], {}
    if not args.no_lyrics and vocal_ratio > 0.03:
        t = time.time()
        try: lines, linfo = lyrics(vocal_path, args.whisper_model, args.language); STATUS["whisper"] = f"ok ({args.whisper_model}, on {'vocal stem' if vocal_path != args.audio else 'mix'})"
        except Exception as e: STATUS["whisper"] = f"failed: {e}"
        tick("whisper", t); log("lyrics", len(lines))
        if args.lyrics and lines:
            lines, st = align_lyrics(lines, open(args.lyrics, encoding="utf-8").read()); linfo["source"] = st; STATUS["lyrics_align"] = st
    elif not args.no_lyrics: STATUS["whisper"] = f"skipped: vocals present only {vocal_ratio:.1%} -> instrumental"
    emo = None
    if args.no_essentia: os.environ["MVBRIEF_NO_ESSENTIA"] = "1"
    if not args.no_essentia and essentia_ok():
        t = time.time()
        try: emo = essentia_emotion(args.audio); STATUS["essentia_tf"] = "ok (msd-musicnn + DEAM valence/arousal + mood heads)"
        except Exception as e: STATUS["essentia_tf"] = f"failed: {e}"
        tick("essentia_tf", t)
    else: STATUS["essentia_tf"] = "skipped: essentia not installed/disabled (valence/arousal from CLAP pairs)"
    clap = None
    if not args.no_clap:
        t = time.time()
        try: clap = Clap(); STATUS["clap"] = "ok (laion/clap-htsat-unfused via transformers)"
        except Exception as e: STATUS["clap"] = f"failed: {e}"
        tick("clap_load", t)
    t = time.time()
    feel = detect_feel(ll, act, curves, emo)
    relabel_by_lyrics(sections, lines)
    segs, mode = build_segments(lines, sections, ll, dur, vocal_ratio)
    segs = [describe(s, ll, sections, act, emo, clap, y, dur, curves) for s in segs]
    # section summaries
    for i, s in enumerate(sections):
        fa, fb = t2f(s["start"]), t2f(s["end"]); e = E[fa:fb]
        s.update(start=r(s["start"], 2), end=r(s["end"], 2), energy_mean=r(e.mean(), 2),
                 trend="rising" if e[-len(e)//3:].mean() - e[:len(e)//3].mean() > 0.1 else "falling" if e[:len(e)//3].mean() - e[-len(e)//3:].mean() > 0.1 else "steady",
                 lyrics=" / ".join(x["text"] for x in segs if x["type"] == "lyric" and x["section_index"] == i))
        s.pop("energy", None)
        if act:
            fr = {k: r(v[fa:fb].mean(), 2) for k, v in act.items()}; s["stems_active_fraction"] = fr
            s["arrangement"] = "+".join(k for k, v in fr.items() if v >= 0.3) or "sparse"
        if emo is not None:
            m = (emo["times"] >= s["start"]) & (emo["times"] <= s["end"])
            if m.any(): s["valence"], s["arousal"] = r(emo["valence"][m].mean(), 2), r(emo["arousal"][m].mean(), 2)
        if clap is not None:
            yy = librosa.resample(y[int(s["start"] * SR): int(s["end"] * SR)], orig_sr=SR, target_sr=48000); c = s["start"] * 0
            tg = clap.tags(yy[max(0, len(yy) // 2 - 240000): len(yy) // 2 + 240000]); s["clap_mood"], s["clap_genre"], s["clap_instrument"] = tg["mood"], tg["genre"], tg["instrument"]
    tick("segments+tags", t); del clap; import gc; gc.collect()
    if args.alm:
        STATUS["alm"] = alm_status()
    gm = {}
    if emo is None and any("valence" in s["mood"] for s in segs):
        vs = [s["mood"]["valence"] for s in segs if "valence" in s["mood"]]; as_ = [s["mood"]["arousal"] for s in segs if "arousal" in s["mood"]]
        gm = dict(valence=r(np.mean(vs), 2), arousal=r(np.mean(as_), 2), source="clap")
        for i, s in enumerate(sections):
            ss = [x["mood"] for x in segs if x["section_index"] == i and "valence" in x["mood"]]
            if ss: s["valence"], s["arousal"] = r(np.mean([m["valence"] for m in ss]), 2), r(np.mean([m["arousal"] for m in ss]), 2)
    if emo is not None: gm = dict(valence=r(emo["valence"].mean(), 2), arousal=r(emo["arousal"].mean(), 2), essentia_moods={k: r(emo[k].mean(), 2) for k in emo if k not in ("times", "valence", "arousal")})
    A = dict(file=os.path.abspath(args.audio), segmentation_mode=mode, components=STATUS, timing_s=timing,
             global_=None)
    A.pop("global_")
    A["global"] = dict(duration=r(dur, 2), bpm=r(ll["tempo"], 2), bpm_essentia=r(ll["tempo_essentia"], 2), bpm_librosa=r(ll["tempo_librosa"], 2), bpm_choice=ll["tempo_choice"], key=ll["key"], vocal_ratio=r(vocal_ratio, 3),
                       lyrics_language=linfo.get("language"), feel=feel, lyrics_source=linfo.get("source", "whisper"), mood=gm, n_beats=len(ll["beats"]), n_downbeats=len(ll["downbeats"]))
    A["sections"] = sections; A["segments"] = segs; A["lyrics_raw"] = lines
    A["beats"] = [r(b) for b in ll["beats"]]; A["downbeats"] = [r(b) for b in ll["downbeats"]]
    step = int(0.5 * SR / HOP)
    A["curves_0.5s"] = dict(t=[r(i * HOP / SR, 2) for i in range(0, len(E), step)], energy=[r(x, 3) for x in E[::step]], brightness=[r(x, 3) for x in curves["bright"][::step]],
                           **({"valence_t": [r(x, 2) for x in emo["times"]], "valence": [r(x, 3) for x in emo["valence"]], "arousal": [r(x, 3) for x in emo["arousal"]]} if emo is not None else {}),
                           **({f"stem_{k}_active": [int(x) for x in v[::step]] for k, v in act.items()} if act else {}))
    timing["total"] = round(time.time() - T0, 1)
    json.dump(A, open(os.path.join(args.outdir, "analysis.json"), "w", encoding="utf-8"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    write_md(A, os.path.join(args.outdir, "analysis.md")); plot(A, curves, ll, os.path.join(args.outdir, "analysis.png"))
    log("done in", timing["total"], "s ->", args.outdir); print(json.dumps(dict(components=STATUS, timing_s=timing), indent=1))

def alm_status():
    import psutil
    gb = psutil.virtual_memory().available / 1e9
    try: import torch; gpu = torch.cuda.is_available()
    except Exception: gpu = False
    return f"skipped: Qwen2-Audio-7B needs ~16GB (bf16) RAM/VRAM; this box has {gb:.1f}GB free RAM, GPU={gpu}"

if __name__ == "__main__": main()
