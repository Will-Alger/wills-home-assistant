# Removes the always-on autostart and stops the running service.
$startup = [Environment]::GetFolderPath("Startup")
Remove-Item (Join-Path $startup "Alexa Assistant.lnk") -ErrorAction SilentlyContinue
& (Join-Path $PSScriptRoot "alexa-stop.cmd")
Write-Host "Autostart removed."
