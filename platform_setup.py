"""Cross-platform runtime setup: UTF-8 console, Windows CUDA DLL dirs (pip nvidia-* wheels), ffmpeg discovery."""
import os, sys, shutil, glob
from pathlib import Path

_done = False

def setup():
    global _done
    if _done: return
    _done = True
    os.environ.setdefault("PYTHONIOENCODING", "utf-8"); os.environ.setdefault("PYTHONUTF8", "1")
    for st in (sys.stdout, sys.stderr):
        try: st.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass
    if os.name == "nt": add_cuda_dll_dirs()
    ensure_ffmpeg()

def add_cuda_dll_dirs():
    """faster-whisper (CTranslate2) needs cuBLAS + cuDNN DLLs; pip wheels put them in site-packages/nvidia/*/bin."""
    import site
    roots = [Path(p) for p in site.getsitepackages() + [site.getusersitepackages()]]
    dirs = []
    for r in roots:
        dirs += [Path(d) for d in glob.glob(str(r / "nvidia" / "*" / "bin"))]
        tl = r / "torch" / "lib"
        if tl.is_dir(): dirs.append(tl)
    for d in dirs:
        try: os.add_dll_directory(str(d))
        except Exception: pass
    if dirs: os.environ["PATH"] = os.pathsep.join(str(d) for d in dirs) + os.pathsep + os.environ.get("PATH", "")

def ensure_ffmpeg():
    """Return an ffmpeg path: PATH (winget/choco/apt) first, else imageio-ffmpeg's bundled binary (added to PATH)."""
    exe = shutil.which("ffmpeg")
    if exe: return exe
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe(); d = str(Path(exe).parent)
        # audioread/demucs look for a binary literally named ffmpeg(.exe): provide an alias next to it if needed
        name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        alias = Path(d) / name
        if not alias.exists():
            try: shutil.copy(exe, alias)
            except Exception: pass
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        return str(alias if alias.exists() else exe)
    except Exception:
        return None

if __name__ == "__main__":
    setup(); print("ffmpeg:", ensure_ffmpeg())
