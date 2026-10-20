@echo off
rem Double-click to start Sticker Forge (needs uv and ffmpeg, see README)
uv run "%~dp0stickermaker.py" %*
if errorlevel 1 pause
