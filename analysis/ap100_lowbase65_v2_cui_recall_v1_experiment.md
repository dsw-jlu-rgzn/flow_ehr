# AP100 Low-Base65 V2 CUI Recall V1 Experiment

## Version

- Experiment id: `ap100lowbase65_generated_method2_gen_v2_cui_recall_v1`
- Prompt version: `v2_cui_recall`
- Scope: low-base subset, 65 / 653 patient-days selected by lowest base UMLS CUI-F1
- Case list: `outputs/ap_memory_gated_scaffold_ap100/low_base65_cases.txt`

## What Changed From V2

This is a non-judge V2 variant. It does not run generation-time judge revision.

Changes added to `modeling/ap_memory_gated_scaffold_generation.py`:

1. `must_cover_concepts`
   - Extract high-confidence concepts from today's EHR and supported prior major headings.
   - Covers active problems, medications/treatments, procedures/devices, lab/physiology, infection/microbiology, and goals/disposition/status.

2. `active_state_audit`
   - Labels candidate concepts as `active_today`, `worsening_today`, `improving_today`, `resolved_today`, `historical_context_only`, `unsupported_today`, or `unclear`.
   - Prevents stale carry-forward for pressors, ventilation, dialysis/CRRT, antibiotics, anticoagulation, nutrition route, procedures, and CMO/death status.

3. Concrete-heading constraint
   - Requires specific clinical headings rather than generic labels.
   - Example: use `Sustained VT on amiodarone infusion`, not `Arrhythmia`.

4. Coverage self-check inside the generation prompt
   - Before finalizing, the model silently checks high-priority concepts are covered and stale concepts are excluded.

## Baseline Low-Base65 UMLS Results

| Method | Precision | Recall | CUI-F1 | Pred CUI |
|---|---:|---:|---:|---:|
| base | 0.3679 | 0.1959 | 0.2448 | 197.4 |
| V2 | 0.3726 | 0.1983 | 0.2476 | 192.4 |
| V2 judge | 0.3729 | 0.2045 | 0.2542 | 198.2 |

Paired deltas:

| Comparison | Mean F1 Delta | Wins | Losses |
|---|---:|---:|---:|
| V2 - base | +0.0028 | 36 | 29 |
| V2 judge - base | +0.0094 | 35 | 30 |
| V2 judge - V2 | +0.0066 | 33 | 32 |

## Run Status

Completed on the 65-case low-base subset.

Note: the PowerShell script was blocked by local execution policy, so the same
Python commands were run directly. The first run exposed one JSON truncation
case; rerunning with `--scaffold-max-tokens 8000`, `--parse-retries 5`, and
`--workers 1` completed all 65 cases.

To reproduce:

```powershell
$env:DEEPSEEK_API_KEY = "..."
.\scripts\run_ap100_lowbase65_v2_cui_recall_v1.ps1
```

For another OpenAI-compatible provider:

```powershell
$env:MY_API_KEY = "..."
.\scripts\run_ap100_lowbase65_v2_cui_recall_v1.ps1 `
  -ApiKeyEnv MY_API_KEY `
  -ApiUrl "https://provider.example.com/v1/chat/completions" `
  -Model "provider-model-name"
```

## Expected Evaluation Output

- Generated notes: `outputs/ap_memory_gated_scaffold_ap100/ap100lowbase65_generated_method2_gen_v2_cui_recall_v1/`
- Scaffolds: `outputs/ap_memory_gated_scaffold_ap100/scaffolds/ap100lowbase65_generated_method2_gen_v2_cui_recall_v1/`
- Summary: `outputs/ap_memory_gated_scaffold_ap100/ap100lowbase65_generated_method2_gen_v2_cui_recall_v1_summary.csv`
- UMLS eval: `outputs/ap_memory_gated_scaffold_ap100/umls_eval_lowbase65_v2_cui_recall_v1/`

## Final UMLS Results

| Method | Precision | Recall | CUI-F1 | Pred CUI |
|---|---:|---:|---:|---:|
| base | 0.3679 | 0.1959 | 0.2448 | 197.4 |
| V2 | 0.3726 | 0.1983 | 0.2476 | 192.4 |
| V2 judge | 0.3729 | 0.2045 | 0.2542 | 198.2 |
| V2 CUI recall V1 | 0.3704 | 0.2036 | 0.2528 | 199.8 |

Paired deltas:

| Comparison | Mean F1 Delta | Wins | Losses | Ties |
|---|---:|---:|---:|---:|
| V2 CUI recall V1 - base | +0.0080 | 34 | 30 | 1 |
| V2 CUI recall V1 - V2 | +0.0052 | 34 | 31 | 0 |
| V2 CUI recall V1 - V2 judge | -0.0014 | 30 | 35 | 0 |

Interpretation:

- The non-judge optimized V2 moved in the intended direction: higher recall
  than old V2 (0.2036 vs 0.1983) and higher CUI-F1 (0.2528 vs 0.2476).
- It nearly matched V2 judge (0.2528 vs 0.2542), but did not exceed it.
- The predicted CUI count increased from 192.4 to 199.8, close to V2 judge's
  198.2, without a large precision collapse.

Largest improvements over old V2:

| Case | base | V2 | V2 CUI recall V1 | V2 judge |
|---|---:|---:|---:|---:|
| 198275 day 11 | 0.245 | 0.168 | 0.418 | 0.252 |
| 190481 day 3 | 0.259 | 0.164 | 0.369 | 0.314 |
| 199046 day 14 | 0.228 | 0.233 | 0.370 | 0.240 |
| 110458 day 8 | 0.266 | 0.268 | 0.401 | 0.264 |
| 119898 day 5 | 0.223 | 0.178 | 0.302 | 0.337 |

Largest regressions vs old V2:

| Case | base | V2 | V2 CUI recall V1 | V2 judge |
|---|---:|---:|---:|---:|
| 176840 day 10 | 0.235 | 0.433 | 0.258 | 0.272 |
| 125899 day 5 | 0.249 | 0.237 | 0.143 | 0.273 |
| 148910 day 5 | 0.223 | 0.298 | 0.208 | 0.139 |
| 128729 day 13 | 0.264 | 0.274 | 0.209 | 0.272 |
| 184018 day 13 | 0.268 | 0.292 | 0.227 | 0.257 |

## Success Criteria

Primary:

- `v2_cui_recall_v1` CUI-F1 > old V2 CUI-F1 on low-base65. Met.
- Target improvement: +0.01 absolute F1 over old V2, preferably mainly through recall. Not met; observed +0.0052.

Secondary:

- Predicted CUI count should increase moderately from old V2's 192.4, but not exceed base/V2 judge by a large margin. Met: 199.8.
- Precision should not fall below old V2 by more than 0.02. Met: 0.3704 vs 0.3726.
- Case-level wins over base should exceed old V2's 36 / 65. Not met: 34 wins, 30 losses, 1 tie.
