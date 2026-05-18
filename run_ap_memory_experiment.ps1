param(
    [string]$Python = "C:\Users\dsw54\AppData\Local\Programs\Python\Python312\python.exe",
    [string]$InputDir = "data\AP\input",
    [string]$GoldDir = "data\AP\gold",
    [string]$OutDir = "experiments\problemflow_ap\outputs_ap_memory",
    [string]$Method = "all",
    [string]$Llm = "mock",
    [int]$Limit = 0,
    [switch]$AutoregressiveHistory
)

$argsList = @(
    "experiments\problemflow_ap\ap_memory_experiment.py",
    "run-all",
    "--inputdir", $InputDir,
    "--golddir", $GoldDir,
    "--outdir", $OutDir,
    "--method", $Method,
    "--llm", $Llm,
    "--limit", $Limit
)

if ($AutoregressiveHistory) {
    $argsList += "--autoregressive-history"
}

& $Python @argsList
