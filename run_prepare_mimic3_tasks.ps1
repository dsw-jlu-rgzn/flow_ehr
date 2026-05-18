param(
    [int]$SampleSize = 100,
    [string]$RawDir = "C:\Users\dsw54\Desktop\MIMIC_related\mimic-iii-20260513T124356Z-3-001\mimic-iii",
    [string]$Python = "C:\Users\dsw54\AppData\Local\Programs\Python\Python312\python.exe"
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

if (-not (Test-Path $Python)) {
    throw "Python executable not found: $Python"
}

if (-not (Test-Path $RawDir)) {
    throw "MIMIC-III raw directory not found: $RawDir"
}

& $Python processing\prepare_mimic3_tasks.py `
    --raw-dir $RawDir `
    --output-root data `
    --sample-size $SampleSize `
    --make-tasks

& $Python processing\validate_mimic3_tasks.py --data-root data
