"""Pre-download all models so analysis runs offline/fast."""
import os, urllib.request, shutil, librosa
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(D, exist_ok=True)
for u in ["https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb",
          "https://essentia.upf.edu/models/classification-heads/deam/deam-msd-musicnn-2.pb",
          "https://essentia.upf.edu/models/classification-heads/deam/deam-msd-musicnn-2.json"]+[f"https://essentia.upf.edu/models/classification-heads/{m}/{m}-msd-musicnn-1.pb" for m in ["mood_happy","mood_sad","mood_aggressive","mood_relaxed","mood_party","mood_electronic","mood_acoustic","danceability"]]:
    p = os.path.join(D, os.path.basename(u))
    if not os.path.exists(p):
        try: urllib.request.urlretrieve(u, p); print("got", p)
        except Exception as e: print("FAIL", u, e)
from faster_whisper import WhisperModel; WhisperModel("small", device="cpu", compute_type="int8"); print("whisper ok")
from transformers import ClapModel, ClapProcessor
ClapModel.from_pretrained("laion/clap-htsat-unfused"); ClapProcessor.from_pretrained("laion/clap-htsat-unfused"); print("clap ok")
from demucs.pretrained import get_model; get_model("htdemucs"); print("demucs ok")
os.makedirs("test", exist_ok=True)
shutil.copy(librosa.example("fishin"), "test/fishin.ogg"); print("test clip ok")
import subprocess,sys; subprocess.run([sys.executable,'calibrate_clap.py'])
