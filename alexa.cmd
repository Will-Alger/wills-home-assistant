@echo off
rem Foreground console session with Alexa (real home). Say "Alexa" to wake.
rem For the invisible always-on version: scripts\install_autostart.ps1
rem NOTE: don't run this while the always-on service is running (mic clash) —
rem scripts\alexa-stop.cmd first.
cd /d "%~dp0"
uv run scripts/m4_realtime.py
pause
