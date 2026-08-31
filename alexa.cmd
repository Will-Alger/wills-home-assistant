@echo off
rem Double-click to talk to Alexa. Say "Alexa" to wake her; Ctrl+C quits.
rem Runs against the fake apartment; drop --fake below once Home Assistant
rem (Milestone 1, the HAOS VM evening) is up and HA_TOKEN is set in .env.
cd /d "%~dp0"
uv run scripts/m4_realtime.py --fake
pause
