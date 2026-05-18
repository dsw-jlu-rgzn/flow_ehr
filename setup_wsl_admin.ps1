# Run this script from an elevated PowerShell window.
# It enables the Windows features required by WSL2. A Windows reboot is
# normally required before installing Ubuntu.

$ErrorActionPreference = "Stop"

Write-Host "Enabling Windows Subsystem for Linux..."
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

Write-Host "Enabling Virtual Machine Platform..."
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

Write-Host ""
Write-Host "WSL features have been requested."
Write-Host "Please reboot Windows, then run from normal PowerShell:"
Write-Host "  wsl --install -d Ubuntu"
Write-Host ""
