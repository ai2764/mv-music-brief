"""Build models/clap_calib.json: per mood-pair mean/std of CLAP (pos-neg) similarity over 10s windows of varied reference music."""
import json, numpy as np, librosa, analyze
c = analyze.Clap(); D = []
for ex in ["brahms", "choice", "fishin", "nutcracker", "pistachio", "sweetwaltz", "vibeace", "trumpet", "drese", "humpback"]:
    y, _ = librosa.load(librosa.example(ex), sr=48000, duration=60)
    for i in range(0, max(1, len(y) - 480000 + 1), 240000): D.append(c.pair_diff(y[i:i + 480000]))
D = np.array(D); print(len(D), "windows")
json.dump({"mean": D.mean(0).tolist(), "std": (D.std(0) + 1e-4).tolist(), "pairs": analyze.MOOD_PAIRS}, open(f"{analyze.MODELS}/clap_calib.json", "w", encoding="utf-8"))
