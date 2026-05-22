param(
  [string]$ApiKeyEnv = "QWEN_API_KEY",
  [string]$Model = "Qwen/Qwen3.6-35B-A3B",
  [string]$ApiUrl = "https://api.siliconflow.cn/v1/chat/completions"
)

$OutRoot = "outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_2case_final"

python evaluation/judge_ap_pairwise_llm.py `
  --selected "$OutRoot/selected_2case.json" `
  --method-a-dir "$OutRoot/base_outputs_2case" `
  --method-b-dir "$OutRoot/truth_revised_outputs/ap_delta_truth_verifier_revise_2case" `
  --method-a-name base `
  --method-b-name delta_truth_revised `
  --data-root data_ap100_ap/AP `
  --output-csv "$OutRoot/pairwise_base_vs_delta_truth_revised_qwen36.csv" `
  --model $Model `
  --api-url $ApiUrl `
  --api-key-env $ApiKeyEnv `
  --temperature 0 `
  --max-tokens 1800 `
  --retries 3 `
  --sleep-seconds 2
