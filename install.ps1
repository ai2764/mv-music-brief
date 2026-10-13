<#
.SYNOPSIS
  Install mv-music-brief on native Windows 11 (NVIDIA GPU such as RTX 4090, Python 3.10-3.12).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\install.ps1
  powershell -ExecutionPolicy Bypass -File .\install.ps1 -Cpu
#>
param(
    [switch]$Cpu,                       # install CPU-only torch
    [string]$Python = ""                # explicit python.exe path (optional)
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Find-Python {
    if ($Python) { return $Python }
    foreach ($v in @("3.11", "3.12", "3.10")) {
        try {
            $p = (& py "-$v" -c "import sys; print(sys.executable)" 2>$null)
            if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() }
        } catch { Write-Verbose "py launcher has no $v" }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw "Python 3.10-3.12 not found. Install with: winget install Python.Python.3.11"
}

$py = Find-Python
$ver = (& $py -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
if (-not (@("3.10", "3.11", "3.12") -contains $ver)) { throw "Python $ver found at $py; need 3.10-3.12 (winget install Python.Python.3.11)" }
Write-Output "Python $ver : $py"

$gpu = $false
if (-not $Cpu) {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
        $gpu = $true
    } else {
        Write-Warning "nvidia-smi not found: installing CPU build. Install the NVIDIA driver for GPU support."
    }
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg not in PATH. Optional (imageio-ffmpeg fallback is installed). Recommended: winget install Gyan.FFmpeg"
}

if (-not (Test-Path ".venv")) { & $py -m venv .venv }
$vpy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $vpy -m pip install -U pip wheel
if ($gpu) {
    & $vpy -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
    # cuBLAS + cuDNN 9 DLLs for faster-whisper / CTranslate2 (loaded via os.add_dll_directory in platform_setup.py)
    & $vpy -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
} else {
    & $vpy -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

# essentia extra is Linux-only (environment marker), so it is skipped here automatically
& $vpy -m pip install -e ".[ui,essentia]"
if ($LASTEXITCODE -ne 0) { throw "pip install -e . failed" }

# Activate hook: put pip-installed CUDA DLL dirs on PATH for any tool run inside the venv
$hook = @'

# --- mv-music-brief: CUDA DLLs from pip nvidia-* wheels ---
$__sp = Join-Path $PSScriptRoot "..\Lib\site-packages\nvidia"
if (Test-Path $__sp) { Get-ChildItem $__sp -Directory | ForEach-Object { $b = Join-Path $_.FullName "bin"; if (Test-Path $b) { $env:PATH = "$b;$env:PATH" } } }
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
'@
$act = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (-not (Select-String -Path $act -Pattern "mv-music-brief" -Quiet)) { Add-Content -Path $act -Value $hook -Encoding UTF8 }

& $vpy fetch_models.py
& $vpy -c "import platform_setup as p; p.setup(); import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
& $vpy -c "import platform_setup as p; p.setup(); import ctranslate2; print('ctranslate2 CUDA devices:', ctranslate2.get_cuda_device_count())"
& $vpy -c "import platform_setup as p; print('ffmpeg:', p.ensure_ffmpeg())"
Write-Output ""
Write-Output "Done. Drag an audio file onto mvbrief.bat, or run:  .\mvbrief.bat song.mp3 --lyrics lyrics.txt"
Write-Output "UI: .\.venv\Scripts\mvbrief-ui.exe   (then open http://127.0.0.1:7860)"
