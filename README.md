# Clinical Summarization — Longitudinal EHR Generation

This repository contains code for the paper **"Large Language Models with Temporal Reasoning for Longitudinal Clinical Summarization and Prediction"** (EMNLP 2025 Findings).

**Link**: https://aclanthology.org/2025.findings-emnlp.1128/

---

## 📋 Table of Contents

- [Task Overview](#task-overview)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Pipeline: Step by Step](#pipeline-step-by-step)
  - [1. Target Population Extraction](#1-target-population-extraction)
  - [2. Chronology Construction](#2-chronology-construction)
  - [3. Model Generation](#3-model-generation)
  - [4. Evaluation](#4-evaluation)
- [Running with nohup (Important!)](#running-with-nohup-important)
- [Troubleshooting](#troubleshooting)
- [Flow Matching + Embedding Experiment](#flow-matching--embedding-experiment)
- [Citation](#citation)

---

## 🎯 Task Overview

Three NLP generation/prediction tasks on MIMIC-IV clinical data:

| Task | Input | Output | Script |
|------|-------|--------|--------|
| **Assessment & Plan (A&P)** | Multi-day structured EHR + previous notes | Daily A&P sections of progress notes | `event_ap_fix.py` |
| **Discharge Summary (DS)** | Last 48h structured EHR | Diagnosis + Hospital Course + Discharge Instructions | `event_ds_fix.py` |
| **EHRShot Diagnosis** | Patient data + diagnosis query | Binary prediction | `evaluation/evaluate_ehrshot.py` |

Available models: `mistral`, `qwen`, `deepseek`, `llama3`, `llama2`

---

## 📁 Project Structure

```
longitudinal_clinical_summarization/
├── modeling/                        # Generation scripts
│   ├── event_ap_fix.py              # ✅ A&P generation (fixed)
│   ├── event_ds_fix.py              # ✅ Discharge summary (fixed)
│   ├── event_ap.py / event_ds.py    # Original scripts (broken - login hangs, missing args)
│   ├── ds_gen.py / ap_gen.py        # Direct generation (original paper)
│   ├── RAG_script_ds.py / RAG_script_ap.py  # RAG version
│   ├── flow_match_clinical_framework.py     # Flow Matching experiment (v1)
│   └── flow_match_clinical_v2.py            # Flow Matching experiment (v2, with text embedding)
├── processing/                      # Data preprocessing
│   ├── get_target_population.py     # Filter MIMIC-IV patients
│   ├── get_chronologies_AP.py       # Build A&P chronologies
│   ├── get_chronologies_AP_fix.py   # Fixed version
│   ├── get_chronologies_DS.py       # Build DS chronologies
│   ├── get_chronologies_DS_fix.py   # Fixed version
│   └── get_chronologies_DS_full_fix.py
├── evaluation/                      # Metrics
│   ├── evaluate_ap.py               # ROUGE-L / SapBERT / CUI-F1 for A&P
│   └── evaluate_ds.py               # Same metrics for DS (section-level)
├── setup.sh                         # Full pipeline setup
├── run.sh                           # Original generation runner
└── evaluate.sh                      # Original evaluation runner

data/
├── AP/                              # Assessment & Plan task
│   ├── input/                       # input_{admission_id}.csv
│   └── gold/                        # gt_{admission_id}.csv
├── DS/                              # Discharge Summary task
│   ├── input/                       # 24_both_{admission_id}.csv
│   └── gold/                        # gtsummary_{admission_id}.txt
└── MIMIC-IV/target/                 # Filtered MIMIC-IV tables
```

---

## 🔧 Data Preparation

### Prerequisites

1. Request MIMIC-IV access from [PhysioNet](https://physionet.org/content/mimiciv/3.1/)
2. Download and place `mimic-iv-3.1.zip` and `note.tar.gz` in the project root
3. Install UMLS for CUI evaluation (see [UMLS Setup](#umls-setup) below)

### Step 1: Filter target population

```bash
conda run -n safevla python processing/get_target_population_fix.py
```

This extracts a subset of patients from MIMIC-IV with sufficient data (output in `data/MIMIC-IV/target/`).

### Step 2: Build patient chronologies

```bash
# Discharge Summary chronologies (last 48 hours)
conda run -n safevla python processing/get_chronologies_DS_fix.py

# Assessment & Plan chronologies (multi-day structured data)
conda run -n safevla python processing/get_chronologies_AP_fix.py
```

**Input data format:**

For A&P (`data/AP/input/input_23056393.csv`):
```
DAY | REL_TIME | TIME | TEXT | IS_NOTE
1   | Day 1 08:00 | ... | HR: 88 bpm | 0
1   | Day 1 09:00 | ... | WBC: 12.5 | 0
1   | Day 1 10:00 | ... | Assessment: ... | 1   ← IS_NOTE=1 = doctor's progress note
2   | Day 2 08:00 | ... | HR: 92 bpm | 0
```

For DS (`data/DS/input/24_both_24274249.csv`):
```
REL_TIME | TIME | TEXT
3 hours  | ...  | 12.5 mg Metoprolol administered.
2 hours  | ...  | Hemoglobin 7.80 g/dL.
```

---

## 🤖 Model Generation

### Option A: Event-based generation (**Recommended**, our fixed version)

These scripts implement a **two-stage pipeline**:
1. **Event Extraction (EE)** — LLM extracts key clinical events from raw EHR
2. **Generation (Gen)** — LLM generates the final clinical text using extracted events

#### A&P Generation (Assessment & Plan)

```bash
# Method -1: no history, Method 1: previous note only, Method 2: all history
# Setting gt: uses real doctor notes as context, gen: uses model-generated notes

nohup HF_TOKEN=hf_xxxxxxxxxx CUDA_VISIBLE_DEVICES=7 \
  conda run -n safevla python modeling/event_ap_fix.py \
  --inputdir data/AP/input \
  --outputdir data/AP/generated \
  --setting gt --model mistral \
  > logs/event_ap.log 2>&1 &
```

- Output: `data/AP/generated/EE/mistral/gt/method-1|method1|method2/genpns_{id}.csv`

#### Discharge Summary Generation

```bash
nohup HF_TOKEN=hf_xxxxxxxxxx CUDA_VISIBLE_DEVICES=7 \
  conda run -n safevla python modeling/event_ds_fix.py \
  --inputdir data/DS/input \
  --outputdir data/DS/generated \
  --model mistral \
  > logs/event_ds.log 2>&1 &
```

- Output: `data/DS/generated/EE/mistral/gtsummary_{id}.txt`

### Option B: Direct generation (Original paper)

These run the original paper scripts (`ds_gen.py`, `ap_gen.py`):

```bash
# DS
CUDA_VISIBLE_DEVICES=0,1 conda run -n safevla python modeling/ds_gen.py --outputdir results/directgen_discharge_sum

# AP (runs all 3 methods)
CUDA_VISIBLE_DEVICES=0,1 conda run -n safevla python modeling/ap_gen.py --method -1 --outputdir results/directgen_assessment_plan/method-1
CUDA_VISIBLE_DEVICES=0,1 conda run -n safevla python modeling/ap_gen.py --method 1  --outputdir results/directgen_assessment_plan/method1
CUDA_VISIBLE_DEVICES=0,1 conda run -n safevla python modeling/ap_gen.py --method 2  --outputdir results/directgen_assessment_plan/method2
```

Or use the provided runner:
```bash
bash run.sh ds
bash run.sh ap
```

### Option C: RAG generation

```bash
bash run.sh rag_ds
bash run.sh rag_ap
```

---

## 📊 Evaluation

### Evaluate A&P generation

```bash
conda run -n safevla python evaluation/evaluate_ap.py \
  --gen_dir data/AP/generated/EE/mistral/gt \
  --gt_dir data/AP/gold
```

Evaluates all 3 methods (method-1, method1, method2) with:
- **ROUGE-L F1** — lexical overlap
- **SapBERT F1** — semantic similarity via PubMedBERT embeddings
- **CUI-F1** — clinical concept overlap (requires UMLS)

### Evaluate DS generation

```bash
conda run -n safevla python evaluation/evaluate_ds.py \
  --gen_dir data/DS/generated/EE/mistral \
  --gt_dir data/DS/gold
```

Evaluates 3 sections separately (Diagnosis / Hospital Course / Discharge Instructions).

---

## 📌 Running with nohup (Important!)

**Scripts take 30–60 minutes to finish** on Mistral-7B (6 patients × ~20 days × 2 passes each).

Do NOT run directly — use `nohup` to avoid timeout:

```bash
# Recommended: create a logs directory first
mkdir -p logs

# Start AP generation
nohup HF_TOKEN= CUDA_VISIBLE_DEVICES=7 \
  conda run -n safevla python modeling/event_ap_fix.py \
  --inputdir data/AP/input --outputdir data/AP/generated --setting gt --model mistral \
  > logs/event_ap_fix.log 2>&1 &

# Start DS generation
nohup HF_TOKEN= CUDA_VISIBLE_DEVICES=7 \
  conda run -n safevla python modeling/event_ds_fix.py \
  --inputdir data/DS/input --outputdir data/DS/generated --model mistral \
  > logs/event_ds_fix.log 2>&1 &

# Check progress
tail -f logs/event_ap_fix.log

# Check GPU usage
watch -n 2 nvidia-smi

# Check if process is still running
ps aux | grep event_ap_fix
```

---

## ⚠️ Known Issues & Fixes

> We encountered several issues with the original codebase and created `_fix.py` versions.

| Issue | Original Code | Fix |
|-------|--------------|-----|
| 🟡 **login() hangs** | `login()` blocks forever on servers without interactive terminal | `try_login()` uses `HF_TOKEN` env var instead |
| 🟡 **Missing HF mirror** | Directly downloads from `huggingface.co` (blocked in some regions) | Auto-sets `HF_ENDPOINT=https://hf-mirror.com` |
| 🟡 **Missing `--outputdir`** | `event_ap.py` uses `args.outputdir` but never defines it | Added `--outputdir` argument with default |
| 🟡 **Model loaded per method** | LLM re-loaded inside the file loop (~30s × N files wasted) | Model loaded once at the start |
| 🟡 **No GPU memory cleanup** | OOM after processing several patients | `del + torch.cuda.empty_cache() + gc.collect()` after each inference |
| 🟡 **BitsAndBytes version** | Old `bitsandbytes` fails with `'NoneType' object has no attribute 'quant_type'` | Upgraded to `bitsandbytes>=0.49.2` |
| 🟢 **DS output naming** | Output filename doesn't match gold naming convention | Changed to `gtsummary_{id}.txt` |

### How the fixed versions differ

| Feature | `event_ap.py` (original) | `event_ap_fix.py` (fixed) |
|---------|------------------------|---------------------------|
| `--outputdir` arg | ❌ Missing | ✅ Added (default: `data/AP/generated`) |
| Login | `login()` hangs | `try_login()` with HF_TOKEN |
| Model loading | Inside loop (N times) | Once at startup |
| HF mirror | None | Auto `hf-mirror.com` |
| GPU memory | Never cleaned | Cleaned after each call |
| DS output name | `24_both_{id}.txt` | `gtsummary_{id}.txt` |

---

## 🔬 Flow Matching + Embedding Experiment

This is an experimental framework for **unsupervised EHR compression using Flow Matching**.

### Idea

Instead of feeding raw EHR text directly into the LLM, use a small Flow Matching model to:
1. Encode daily EHR events into **semantic embeddings** (via Sentence-BERT)
2. Compress the sequence into a **patient course vector** (via Flow Matching)
3. Convert to **soft prompts** injected into the LLM's embedding space

### Architecture

```
EHR text → Sentence-BERT (384d) → FlowMatchingEncoder (128d) 
  → Projector (16 tokens × 4096d) → LLM generates clinical text
```

### Training stages

| Stage | What | Loss | Data needed |
|-------|------|------|-------------|
| **1** | Flow Matching pre-training | MSE (predict velocity field) | EHR text only (no labels) |
| **2** | Projector training | CE Loss (text generation) | (EHR, doctor note) pairs |
| **3** (optional) | Full fine-tune with LoRA | CE Loss | Same as Stage 2 |

### Files

- `modeling/flow_match_clinical_v2.py` — Complete implementation
- Uses `sentence-transformers/all-MiniLM-L6-v2` for text embedding (frozen, 80M params)
- Flow Encoder: ~15M trainable params, Projector: ~5M trainable params
- LLM (Mistral-7B): frozen, loaded with 4-bit quantization

### Run

```bash
# Stage 1: Unsupervised Flow Matching (no labels needed)
conda run -n safevla python modeling/flow_match_clinical_v2.py --stage 1 --epochs 100

# Demo with random initialization
conda run -n safevla python modeling/flow_match_clinical_v2.py --stage 0
```

---

## 📝 Evaluation Metrics Explained

### ROUGE-L F1
Measures **n-gram overlap** between generated and ground-truth text. Higher = more lexically similar.

### SapBERT-F1 (Semantic F1)
Uses **SapBERT** (PubMedBERT fine-tuned with contrastive learning) to compute token-level semantic similarity. Captures **paraphrased but clinically equivalent** content.

### CUI-F1
Extracts **Concept Unique Identifiers** (from UMLS Metathesaurus) from both texts and computes exact concept overlap. Requires a working UMLS/QuickUMLS installation.

---

## 🔧 UMLS Setup

Required for CUI-F1 evaluation:

```bash
# 1. Download UMLS from: https://www.nlm.nih.gov/research/umls/
# 2. Install QuickUMLS
pip install quickumls

# 3. Point QuickUMLS to your UMLS installation in the evaluation scripts
quickumls = QuickUMLS("/path/to/QuickUMLS/", ...)
```

---

## 📚 Citation

```bibtex
@misc{kruse2025largelanguagemodelstemporal,
  title={Large Language Models with Temporal Reasoning for Longitudinal Clinical Summarization and Prediction}, 
  author={Maya Kruse and Shiyue Hu and Nicholas Derby and Yifu Wu and Samantha Stonbraker and Bingsheng Yao and Dakuo Wang and Elizabeth Goldberg and Yanjun Gao},
  year={2025},
  eprint={2501.18724},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2501.18724}, 
}
```

---

## 🙏 Acknowledgement

This work is supported by National Library of Medicine R00 LM014308.
