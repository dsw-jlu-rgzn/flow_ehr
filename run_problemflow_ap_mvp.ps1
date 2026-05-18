param(
    [string]$Python = "C:\Users\dsw54\AppData\Local\Programs\Python\Python312\python.exe",
    [string]$InputDir = "data\AP\input",
    [string]$GoldDir = "data\AP\gold",
    [string]$OutDir = "experiments\problemflow_ap\outputs",
    [string]$Method = "all",
    [string]$Llm = "mock",
    [int]$Limit = 0
)

& $Python experiments\problemflow_ap\problemflow_ap.py run-all `
    --inputdir $InputDir `
    --golddir $GoldDir `
    --outdir $OutDir `
    --method $Method `
    --llm $Llm `
    --limit $Limit
