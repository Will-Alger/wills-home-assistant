@echo off
rem Stops the always-on Alexa service and the voice app.
cd /d "%~dp0.."
if not exist data mkdir data
type nul > data\stop.flag
if exist data\service.pid (
  for /f "usebackq" %%p in ("data\service.pid") do taskkill /pid %%p /f >nul 2>&1
  del data\service.pid >nul 2>&1
)
echo Alexa stopped. (Autostart shortcut still installed unless you ran uninstall_autostart.ps1)
