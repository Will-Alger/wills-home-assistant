# Installs Alexa as an always-on assistant: a Startup-folder shortcut runs
# the service (via pythonw = no console window) at every Windows logon.
# No admin rights needed. Remove with scripts\uninstall_autostart.ps1.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $root ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pythonw)) { throw "venv not found - run 'uv sync' first" }

$startup = [Environment]::GetFolderPath("Startup")
$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut((Join-Path $startup "Alexa Assistant.lnk"))
$lnk.TargetPath = $pythonw
$lnk.Arguments = "`"$root\scripts\alexa_service.py`""
$lnk.WorkingDirectory = $root
$lnk.Description = "Always-on voice assistant"
$lnk.Save()
Write-Host "Installed: Alexa starts at every logon (shortcut in shell:startup)."
Write-Host "Start her right now too? Running the service in the background..."
Start-Process -FilePath $pythonw -ArgumentList "`"$root\scripts\alexa_service.py`"" -WorkingDirectory $root
Write-Host "Done. Logs: $root\logs\alexa.log · Stop: scripts\alexa-stop.cmd"
