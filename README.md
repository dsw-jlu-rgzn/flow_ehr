# Flow-EHR: Longitudinal Clinical Text Generation and Evidence-Grounded Revision

This repository contains code for longitudinal EHR text generation experiments on
MIMIC-style clinical data. It extends the original A&P / discharge-summary
generation pipeline with evidence filtering, no-training embedding baselines,
ProblemFlow multi-agent generation, and the current **ProblemFlow V6** framework:

> high-coverage direct draft + certainty gate + coverage guard + verifier +
> minimal evidence-constrained revision.

The project is designed for reproducible experiments on:

- **A&P generation**: daily Assessment & Plan generation from structured EHR events and note context.
- **Discharge Summary generation**: diagnosis, hospital course, and discharge instruction generation.
- **Evidence-grounded evaluation**: ROUGE-L, SapBERT, CUI-F1 when UMLS is available, lightweight clinical semantic metrics, and LLM-as-a-judge protocols.

No MIMIC or UMLS data is committed to this repository.

## Repository Layout

```text
analysis/                         Research notes and experiment reports
evaluation/                       AP/DS evaluators and clinical semantic metrics
experiments/problemflow_ap/       ProblemFlow AP experiments, including V5/V6
modeling/                         Generation, prefiltering, and DeepSeek scripts
processing/                       MIMIC preprocessing and task construction
scripts/                          Utility scripts, including QuickUMLS indexing
run_*.ps1 / run_*.sh              Convenience runners
```

Large or licensed local artifacts are intentionally ignored:

```text
data/
data_backup*/
.venv*/
evaluation/*outputs*/
experiments/**/outputs*/
```

## Main Contributions in This Working Version

### 1. MIMIC task preparation

The repository includes scripts to regenerate AP/DS tasks from raw MIMIC data:

```bash
python processing/prepare_mimic3_tasks.py --help
python processing/validate_mimic3_tasks.py --data-root data
```

For the current local setup, AP and DS task files are expected under:

```text
data/AP/input/
data/AP/gold/
data/DS/input/
data/DS/full_input/
data/DS/gold/
```

### 2. DeepSeek generation support

DeepSeek API generation scripts are provided for AP/DS and ProblemFlow
experiments. Set the key through an environment variable rather than editing it
into files:

```powershell
$env:DEEPSEEK_API_KEY = "..."
```

### 3. ProblemFlow AP experiments

ProblemFlow AP is implemented in:

```text
experiments/problemflow_ap/problemflow_ap.py
```

Supported methods include:

```text
direct
trend
problemflow
problemflow_v2
problemflow_v3
problemflow_v4
problemflow_v5
problemflow_v6
```

Run a small smoke test:

```powershell
& "C:\Users\dsw54\AppData\Local\Programs\Python\Python312\python.exe" `
  experiments\problemflow_ap\problemflow_ap.py run-all `
  --inputdir data\AP\input `
  --golddir data\AP\gold `
  --outdir experiments\problemflow_ap\outputs_deepseek_v6_smoke `
  --method problemflow_v6 `
  --llm deepseek `
  --limit 5
```

Run full AP V6:

```powershell
& "C:\Users\dsw54\AppData\Local\Programs\Python\Python312\python.exe" `
  experiments\problemflow_ap\problemflow_ap.py run-all `
  --inputdir data\AP\input `
  --golddir data\AP\gold `
  --outdir experiments\problemflow_ap\outputs_deepseek_v6_full `
  --method problemflow_v6 `
  --llm deepseek
```

## ProblemFlow V6

V6 is the current best-balanced AP generation design in this repo.

### Motivation

Direct LLM generation usually gives better lexical overlap and broader clinical
coverage, but it can introduce unsupported clinical claims. Evidence-first
ProblemFlow variants improve grounding but may become too conservative and lose
ROUGE-L. V6 targets this trade-off.

### Pipeline

```text
Input context + EHR evidence
-> Evidence Agent
-> Problem Detector
-> Certainty Gate
-> Coverage Guard
-> Direct LLM Writer
-> Draft A&P
-> Verifier
-> Unsupported claim list
-> Minimal Reviser
-> Final A&P
```

Key idea:

```text
Preserve the high-coverage direct draft whenever it is supported, and only make
local edits to unsupported, over-specific, or weakly grounded claims.
```

More details are documented in:

```text
analysis/problemflow_v6_technical_report_zh.md
```

## Evaluation

### Official-style AP evaluation

```bash
python evaluation/evaluate_ap.py \
  --gen_dir data/AP/generated/DG/deepseek_api_full/gt \
  --gt_dir data/AP/gold
```

Metrics:

- ROUGE-L
- SapBERT F1
- CUI-F1, if QuickUMLS/UMLS is configured

### Official-style DS evaluation

```bash
python evaluation/evaluate_ds.py \
  --gen_dir data/DS/generated/DG/deepseek_api_full \
  --gt_dir data/DS/gold
```

### Lightweight clinical semantic evaluation

This evaluator does not require UMLS, QuickUMLS, SapBERT, or a downloaded model:

```bash
python evaluation/evaluate_clinical_semantics.py \
  --samples experiments/problemflow_ap/outputs_deepseek_v6_full/data/ap_samples.jsonl \
  --run direct=experiments/problemflow_ap/outputs_deepseek_direct_full/generations/direct.jsonl \
  --run problemflow_v5=experiments/problemflow_ap/outputs_deepseek_v5_full/generations/problemflow_v5.jsonl \
  --run problemflow_v6=experiments/problemflow_ap/outputs_deepseek_v6_full/generations/problemflow_v6.jsonl \
  --outdir evaluation/clinical_semantics_outputs/direct_vs_v5_vs_v6
```

It reports:

- ROUGE-L
- Clinical Concept F1
- Problem F1
- Treatment F1
- Grounded Concept Rate
- Unsupported Concept Rate
- Numeric Claim Support Rate
- Trend F1
- Evidence Trend Accuracy

Current local AP comparison on 57 samples:

| Method | N | ROUGE-L | Concept F1 | Problem F1 | Treatment F1 | Grounded | Unsupported | Numeric Support | Trend F1 | Evidence Trend Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | 57 | 7.66 | 80.70 | 83.52 | 75.23 | 96.92 | 3.08 | 75.00 | 6.84 | 61.70 |
| problemflow_v5 | 57 | 6.06 | 75.69 | 76.06 | 74.78 | 98.05 | 1.95 | 84.89 | 30.70 | 69.91 |
| problemflow_v6 | 57 | 7.34 | 78.66 | 81.08 | 74.06 | 98.10 | 1.90 | 76.20 | 7.26 | 59.39 |

Interpretation: V6 recovers much of direct generation's ROUGE-L and coverage
while keeping unsupported concepts lower than direct generation.

## QuickUMLS / UMLS Setup

QuickUMLS itself can be installed in Linux/WSL, but UMLS Metathesaurus data must
be downloaded separately from NLM under a UMLS license.

The expected UMLS files are:

```text
MRCONSO.RRF
MRSTY.RRF
```

After placing the UMLS `META` directory locally, build the QuickUMLS index:

```bash
cd /mnt/c/Users/dsw54/Desktop/codex_related/flow_ehr
. .venv-eval/bin/activate
scripts/build_quickumls_index.sh data/umls/2024AB/META data/quickumls/2024AB
```

Then evaluate with CUI-F1 enabled:

```bash
export QUICKUMLS_PATH="$PWD/data/quickumls/2024AB"
python evaluation/evaluate_ap.py --gen_dir ... --gt_dir ...
python evaluation/evaluate_ds.py --gen_dir ... --gt_dir ...
```

## Research Directions

The main research directions explored in this repository include:

- Flow matching / no-training embedding prefilters for AP evidence selection.
- ProblemFlow multi-agent AP generation.
- Draft-preserving evidence-constrained revision.
- LLM-as-a-judge evaluation for clinical usefulness and faithfulness.
- Admission-local style memory for future ROUGE-L improvement.

Relevant notes:

```text
analysis/research_directions_beyond_flow_matching_zh.md
analysis/problemflow_v6_technical_report_zh.md
analysis/ap_ds_data_and_experiment_report_zh.md
```

## Security and Data Policy

Do not commit:

- MIMIC data
- UMLS data
- generated full outputs with sensitive content
- API keys
- local virtual environments

Use environment variables for API keys:

```powershell
$env:DEEPSEEK_API_KEY = "..."
```

## Citation

This project builds on the paper:

```bibtex
@inproceedings{longitudinal-ehr-generation-2025,
  title = {Large Language Models with Temporal Reasoning for Longitudinal Clinical Summarization and Prediction},
  booktitle = {Findings of EMNLP},
  year = {2025}
}
```
