param(
  [string]$ApiKeyEnv = "DEEPSEEK_API_KEY",
  [string]$ApiUrl = "https://api.deepseek.com/chat/completions",
  [string]$Model = "deepseek-chat",
  [int]$Workers = 2
)

$ErrorActionPreference = "Stop"

$ConfigName = "ap100lowbase65_generated_method2_gen_v2_cui_recall_v1"
$OutputDir = "outputs\ap_memory_gated_scaffold_ap100"
$CasesFile = "$OutputDir\low_base65_cases.txt"
$UmlsDir = "C:\Users\dsw54\Downloads\umls-2026AA-full\2026AA-full"

python modeling\ap_memory_gated_scaffold_generation.py `
  --data-root data_ap100_ap `
  --output-dir $OutputDir `
  --config-name $ConfigName `
  --cases-file $CasesFile `
  --baseline-run-name deepseek_api_full_gen `
  --baseline-setting gen `
  --baseline-method method2 `
  --memory-source baseline_method `
  --prompt-version v2_cui_recall `
  --workers $Workers `
  --temperature 0.0 `
  --max-tokens 2000 `
  --scaffold-max-tokens 4200 `
  --api-url $ApiUrl `
  --api-key-env $ApiKeyEnv `
  --model $Model

python evaluation\evaluate_ap100_umls_cui_f1.py `
  --umls_dir $UmlsDir `
  --cases_file $CasesFile `
  --out_dir "$OutputDir\umls_eval_lowbase65_v2_cui_recall_v1" `
  --method_dirs "v2_cui_recall_v1=$OutputDir\$ConfigName"
