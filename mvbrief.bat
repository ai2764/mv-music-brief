@echo off
rem mv-music-brief launcher. Drag-and-drop an audio file onto this .bat, or:
rem   mvbrief.bat "song.mp3" --lyrics "lyrics.txt" --fps 16 -o out\song
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "HERE=%~dp0"
if not exist "%HERE%.venv\Scripts\python.exe" (
  echo [mvbrief] .venv not found. Run: powershell -ExecutionPolicy Bypass -File "%HERE%install.ps1"
  pause
  exit /b 1
)
if "%~1"=="" (
  echo Usage: drag an audio file onto mvbrief.bat, or: mvbrief.bat song.mp3 [--lyrics lyrics.txt] [--fps 16] [-o outdir]
  pause
  exit /b 1
)
rem auto-use lyrics.txt / <song>.txt next to the audio when dropped alone
set "EXTRA="
if "%~2"=="" (
  if exist "%~dpn1.txt" set EXTRA=--lyrics "%~dpn1.txt"
)
"%HERE%.venv\Scripts\python.exe" "%HERE%mvbrief_cli.py" %* %EXTRA% --copy-audio
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" (
  echo.
  echo [mvbrief] Output folder: "%~dp1mvbrief_out\%~n1"
  if "%~2"=="" start "" "%~dp1mvbrief_out\%~n1\report.html"
)
echo.
pause
exit /b %RC%
