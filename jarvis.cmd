@echo off
rem Double-click to talk to Jarvis (typed, for now — voice arrives at M3/M4).
rem Runs against the fake apartment; drop --fake below once Home Assistant
rem (Milestone 1) is up and HA_URL/HA_TOKEN are set in .env.
cd /d "%~dp0"
uv run scripts/m2_repl.py --fake
pause
