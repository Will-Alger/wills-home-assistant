<#
Home Assistant OS in a Hyper-V VM — run in an ELEVATED PowerShell.

Based on the official Windows install docs (verified 2026-08):
https://www.home-assistant.io/installation/windows/
Generation 2 VM, Secure Boot OFF, external (bridged) virtual switch so HA is
a real LAN peer — that's what makes mDNS/Matter/AirPlay discovery work.
Note: Hyper-V has no USB passthrough (irrelevant here — no Zigbee sticks).

Usage (elevated):
    powershell -ExecutionPolicy Bypass -File scripts\setup_haos_vm.ps1
#>
param(
    # Check https://github.com/home-assistant/operating-system/releases for newer
    [string]$VhdxUrl = "https://github.com/home-assistant/operating-system/releases/download/18.2/haos_ova-18.2.vhdx.zip",
    [string]$VmName = "homeassistant",
    [string]$SwitchName = "HAOS External Switch",
    [int]$MemoryGB = 4,
    [int]$Cpus = 2
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # progress bar makes big downloads ~10x slower

# --- preflight -------------------------------------------------------------
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { throw "Run this from an elevated (Administrator) PowerShell." }

$hyperv = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All
if ($hyperv.State -ne "Enabled") {
    Write-Host "Hyper-V is not enabled. Enabling now (a REBOOT will be required)..."
    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All -All -NoRestart
    Write-Host "Reboot, then run this script again." -ForegroundColor Yellow
    exit 0
}

if (Get-VM -Name $VmName -ErrorAction SilentlyContinue) {
    throw "A VM named '$VmName' already exists."
}

# --- external switch (bridged NIC) ----------------------------------------
if (-not (Get-VMSwitch -Name $SwitchName -ErrorAction SilentlyContinue)) {
    $adapters = Get-NetAdapter -Physical | Where-Object Status -eq "Up"
    if (-not $adapters) { throw "No connected physical network adapter found." }
    if (@($adapters).Count -gt 1) {
        Write-Host "Connected adapters:"; $adapters | Format-Table Name, InterfaceDescription, LinkSpeed
        $adapterName = Read-Host "Adapter name to bridge (prefer Ethernet over Wi-Fi)"
    } else {
        $adapterName = $adapters.Name
    }
    Write-Host "Creating external switch '$SwitchName' on '$adapterName' (network blips a few seconds)..."
    New-VMSwitch -Name $SwitchName -NetAdapterName $adapterName -AllowManagementOS $true | Out-Null
}

# --- download + unpack the HAOS disk image ---------------------------------
$vmDir = "$env:PUBLIC\Documents\Hyper-V\$VmName"
New-Item -ItemType Directory -Force $vmDir | Out-Null
$zipPath = Join-Path $vmDir "haos.vhdx.zip"
if (-not (Get-ChildItem $vmDir -Filter *.vhdx -ErrorAction SilentlyContinue)) {
    # A leftover zip from an interrupted run is likely truncated — start clean.
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Write-Host "Downloading HAOS image (a few hundred MB)..."
    Invoke-WebRequest -Uri $VhdxUrl -OutFile $zipPath
    Write-Host "Extracting..."
    Expand-Archive -Path $zipPath -DestinationPath $vmDir -Force
    Remove-Item $zipPath
}
$vhdx = (Get-ChildItem $vmDir -Filter *.vhdx | Select-Object -First 1).FullName
if (-not $vhdx) {
    throw "No .vhdx found after extraction — the download may have failed; just re-run this script."
}

# --- create the VM (per official docs: Gen 2, existing disk, Secure Boot off)
Write-Host "Creating Generation 2 VM '$VmName'..."
New-VM -Name $VmName -Generation 2 -MemoryStartupBytes ($MemoryGB * 1GB) `
    -VHDPath $vhdx -SwitchName $SwitchName | Out-Null
Set-VMProcessor -VMName $VmName -Count $Cpus
Set-VMFirmware -VMName $VmName -EnableSecureBoot Off
Set-VM -Name $VmName -AutomaticStartAction Start -AutomaticStopAction ShutDown `
    -CheckpointType Disabled

Start-VM -Name $VmName
Write-Host ""
Write-Host "HAOS is booting (first boot takes a few minutes)." -ForegroundColor Green
Write-Host "Then open http://homeassistant.local:8123 to onboard."
Write-Host "If that name doesn't resolve, find the VM's IP in your router's client list."
