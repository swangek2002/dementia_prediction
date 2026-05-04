# SurvivEHR Project Knowledge Base

> **Last updated**: 2026-05-02
> **Purpose**: Comprehensive reference for any agent or collaborator joining this project.
> **Owner**: swangek (HKUST)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Sources & Formats](#3-data-sources--formats)
4. [Model Architecture](#4-model-architecture)
5. [Training Pipeline](#5-training-pipeline)
6. [Experiment History & Results](#6-experiment-history--results)
7. [HES Integration Approaches](#7-hes-integration-approaches)
8. [Key Lessons Learned](#8-key-lessons-learned)
9. [Configuration Reference](#9-configuration-reference)
10. [File Reference](#10-file-reference)

---

## 1. Project Overview

**SurvivEHR** is a survival analysis framework built on **FastEHR**, using transformer-based foundation models for **competing-risk survival prediction** on electronic health records (EHR). The primary application is **dementia prediction** using UK CPRD (Clinical Practice Research Datalink) GP records, optionally augmented with HES (Hospital Episode Statistics) hospital data.

### Core Paradigm
- **Self-supervised pretraining** on GP data (next-event prediction) across ~11M patient records
- **Supervised fine-tuning** for specific survival outcomes (dementia onset vs death as competing risks)
- **Evaluation** via time-dependent concordance index (C_td), integrated Brier score (IBS), and integrated negative binomial log-likelihood (INBLL)

### Research Question
Can integrating hospital admission data (HES) with GP records improve dementia prediction beyond GP-only models?

### Key Findings (as of 2026-05-02)
- **⚠️ CRITICAL: Temporal leakage discovered (2026-04-29)** — All HES static/dual experiments prior to 2026-04-30 suffered from temporal leakage in `build_hes_summary_features.py`. HES records from AFTER the index date (age 72) were included in feature computation, giving models access to future information. Previously reported results (hes_static v1: 0.836, dual v1: 0.845, hes_static v2: 0.875) are ALL INVALID. See Section 6.3 for full details.
- **Best model (corrected, dual backbone)**: Dual-backbone v2 (GP + HES backbones, gated fusion, 22-dim clean HES static) achieves **Dementia C_td = 0.743**, a **+0.010** improvement over baseline (0.733). Trained and tested on temporally-correct data.
- **Baseline (hes_aug)**: Dementia C_td = 0.733 — NOT affected by temporal leakage (uses HES only for label augmentation, not features)
- Sequence-level fusion of HES events into GP sequences **hurts** performance (0.720) due to modality clash and truncation
- Late fusion (independent backbones + gated fusion layer) is the correct approach for multi-modal EHR data
- The real improvement from HES static features is modest (~+0.010), not dramatic — most of the previously reported gains were artifacts of temporal leakage

---

## 2. Repository Structure

```
/Data0/swangek_data/991/
├── CPRD/                           # Main project directory
│   ├── src/                        # Model source code
│   │   ├── models/
│   │   │   ├── survival/
│   │   │   │   └── task_heads/
│   │   │   │       ├── causal.py   # SurvStreamGPTForCausalModelling (main model class)
│   │   │   │       └── dual_backbone.py  # DualBackboneSurvModel + FusionLayer (dual-backbone)
│   │   │   ├── transformer/        # Transformer blocks
│   │   │   └── TTE/
│   │   │       └── base.py         # TTETransformer (wraps DataEmbeddingLayer + transformer blocks)
│   │   └── modules/
│   │       ├── data_embeddings/
│   │       │   └── data_embedding_layer.py  # DataEmbeddingLayer (token + static embedding)
│   │       ├── head_layers/        # Output heads (survival, value prediction)
│   │       ├── positions/          # Positional encoding
│   │       └── transformers/       # Transformer layer implementations
│   │
│   ├── examples/modelling/SurvivEHR/  # Experiment scripts & configs
│   │   ├── run_experiment.py          # Main entry point (Hydra-based)
│   │   ├── run_dual_experiment.py     # Dual-backbone entry point
│   │   ├── setup_finetune_experiment.py  # Model setup, checkpoint loading
│   │   ├── setup_dual_finetune_experiment.py  # Dual-backbone experiment setup
│   │   ├── dual_data_module.py        # DualCollateWrapper, HES cache, HES tokenizer
│   │   ├── confs/                     # Hydra YAML configs (see Section 9)
│   │   ├── build_dementia_cr_hes_aug.py     # Dataset builder: GP + HES label augmentation
│   │   ├── build_dementia_cr_hes_static.py  # Dataset builder: GP + HES labels + HES static features
│   │   ├── build_dementia_cr_hes_fusion.py  # Dataset builder: GP + HES sequence fusion (FAILED APPROACH)
│   │   ├── build_hes_summary_features.py    # HES summary feature extractor
│   │   ├── build_hes_events.py              # HES→Read v2 event translator (for fusion)
│   │   ├── build_omop_mapping.py            # OMOP ICD-10→Read v2 mapping
│   │   ├── prepare_hes_fusion_db.py         # Fused database builder
│   │   ├── verify_hes_fusion_dataset.py     # Dataset verification utility
│   │   ├── create_subset_eval.py            # Subset evaluation dataset creator
│   │   └── hes_dementia_lookup.py           # HES dementia diagnosis lookup builder
│   │
│   ├── data/                        # Data directory
│   │   ├── example_exercise_database.db        # Original GP SQLite database (~11M events)
│   │   ├── hes_pretrain_database.db            # HES-only SQLite DB (3.9M events, 420K patients, ICD-10 3-char)
│   │   ├── example_exercise_database_hes_fusion.db  # Fused GP+HES database (DO NOT USE for new experiments)
│   │   ├── hesin.csv                # HES admission records (eid, admidate, etc.)
│   │   ├── hesin_diag.csv           # HES diagnosis records (eid, diag_icd10, etc.)
│   │   ├── hesin_oper.csv           # HES operation records
│   │   ├── death.csv                # Death records
│   │   ├── death_cause.csv          # Death cause records
│   │   ├── gp_clinical.csv          # GP clinical events
│   │   ├── gp_registrations.csv     # GP registration periods
│   │   ├── gp_scripts.csv           # GP prescriptions
│   │   ├── omop_*.csv               # OMOP CDM mapped tables
│   │   ├── omop_icd10_to_readv2.pickle  # ICD-10 to Read v2 mapping via OMOP
│   │   ├── hes_summary_features.pickle  # Per-patient HES summary features (22 dims v2, 449K patients)
│   │   ├── hes_events_for_db.pickle     # Translated HES events (for fusion approach)
│   │   ├── ready_for_code_*.csv         # Pre-processed tables for DB building
│   │   └── FoundationalModel/           # Dataset directories
│   │       ├── PreTrain/                    # GP Pretrain dataset & meta info
│   │       │   ├── meta_information_custom.pickle  # Vocabulary, encoders, token info
│   │       │   └── practice_id_splits.pickle       # Train/val/test practice splits
│   │       ├── PreTrain_HES/               # HES Pretrain dataset & meta info
│   │       │   └── meta_information.pickle         # HES vocabulary (1,501 ICD-10 tokens)
│   │       ├── FineTune_Dementia_CR/                # Baseline dementia CR dataset
│   │       ├── FineTune_Dementia_CR_hes_aug/        # GP + HES label augmentation
│   │       ├── FineTune_Dementia_CR_hes_static/     # GP + HES labels + HES static features (corrected)
│   │       ├── FineTune_Dementia_CR_hes_fusion/     # GP + HES sequence fusion (FAILED)
│   │       ├── FineTune_Dementia_CR_idx60/          # Index age 60 experiment
│   │       ├── FineTune_Dementia_CR_idx70/          # Index age 70 experiment
│   │       ├── FineTune_Dementia_CR_idx74/          # Index age 74 experiment
│   │       ├── FineTune_Dementia_CR_idx75/          # Index age 75 experiment
│   │       ├── FineTune_Dementia_CR_idx68_cv/       # 5-fold CV at index age 68
│   │       └── cv5_splits/                          # 5-fold CV practice splits
│   │
│   ├── output/
│   │   ├── checkpoints/             # Model checkpoints (see Section 10)
│   │   └── wandb/                   # Weights & Biases run logs
│   │
│   ├── run_hes_static_pipeline.sh   # Pipeline script for hes_static
│   ├── run_hes_fusion_pipeline.sh   # Pipeline script for fusion v5 (FAILED)
│   ├── run_hes_fusion_train_only.sh # Train-only pipeline for fusion v5
│   ├── run_dual_pipeline.sh         # Pipeline script for dual-backbone
│   ├── finetune_cr_hes_static_log.txt   # Training log for hes_static run
│   ├── finetune_cr_hes_fusion_log.txt   # Training log for fusion v5 (first attempt)
│   ├── finetune_cr_hes_fusion_train_only_log.txt  # Training log for fusion v5 (successful train)
│   ├── finetune_cr_dual_log.txt     # Training log for dual-backbone fine-tune
│   └── test_cr_dual_log.txt         # Test log for dual-backbone evaluation
│
└── FastEHR/                         # Foundation model framework
    └── FastEHR/
        ├── dataloader/
        │   ├── foundational_loader.py  # FoundationalDataModule - core dataloader
        │   └── utils/
        │       └── study_criteria.py   # Study inclusion criteria (index_inclusion_method)
        ├── database/                   # Database building utilities
        └── adapters/                   # Model adapters (PEFT, LoRA, etc.)
```

---

## 3. Data Sources & Formats

### 3.1 CPRD GP Data
- **Database**: `example_exercise_database.db` (SQLite)
- **Tables**:
  - `event_table`: Patient events with columns (PATIENT_ID, EVENT, DATE, VALUE, ...)
  - `static_table`: Patient demographics (PATIENT_ID, SEX, IMD, ETHNICITY, YEAR_OF_BIRTH, COUNTRY, HEALTH_AUTH, PRACTICE_ID)
- **Coding system**: Read v2 codes (UK GP coding standard)
- **Total tokens in vocabulary**: 108,118 (from `meta_information_custom.pickle`)
- **Total tokens in pretrain data**: ~373.9 million

### 3.2 HES Hospital Data
- **Source files**: `hesin.csv` (admissions) + `hesin_diag.csv` (diagnoses) + `hesin_oper.csv` (operations)
- **Coding system**: ICD-10 (diagnosis), OPCS-4 (procedures)
- **Key columns in hesin.csv**: `dnx_hesin_id`, `eid` (patient ID), `admidate`, `disdate`, `admimeth`, etc.
- **Key columns in hesin_diag.csv**: `dnx_hesin_id`, `eid`, `diag_icd10`, `level` (1=primary, 2+=secondary)
- **Patient linkage**: `eid` field in HES maps to `PATIENT_ID` in CPRD
- **Total patients with HES records**: ~449,095
- **HES pretrain database**: `hes_pretrain_database.db` — 3,898,992 events, 419,966 patients, 1,499 ICD-10 codes (3-char truncated), level=1 primary diagnoses only, 0.2 GB

### 3.3 Dataset Format (Parquet)
Each fine-tuning dataset is stored as partitioned Parquet files:
```
FineTune_*/
├── split=train/
│   └── COUNTRY=UK/HEALTH_AUTH=*/PRACTICE_ID=*/CHUNK=*/  # Nested partitions
│       └── *.parquet
├── split=val/
│   └── ... (same structure)
├── split=test/
│   └── ...
├── file_row_count_dict_train.pickle
├── file_row_count_dict_val.pickle
└── file_row_count_dict_test.pickle
```

**Parquet row columns**:
- `PATIENT_ID`: Integer patient identifier
- `EVENT`: List of event codes (Read v2 strings)
- `DATE`: List of event dates (datetime64)
- `VALUE`: List of event values (optional numeric)
- Static demographic columns: `SEX`, `IMD`, `ETHNICITY`, `YEAR_OF_BIRTH`, `COUNTRY`, `HEALTH_AUTH`, `PRACTICE_ID`
- (For hes_static_v2): 22 `HES_*` columns (float32) — see Section 7.3 for full list

### 3.4 Study Cohort Definition
- **Index event**: Patient reaches age `INDEX_ON_AGE` (default: 72)
- **Study period**: 1998-01-01 to 2022-10-31
- **Age at entry range**: 50-90 years
- **Minimum GP registration**: 1 year
- **Minimum events**: 5 events before index date
- **Outcomes**: 31 dementia Read v2 codes (risk 0) + DEATH (risk 1)
- **Practice-based splitting**: Patients split by `PRACTICE_ID` into train/val/test to prevent data leakage
  - Split assignments stored in `practice_id_splits.pickle`

### 3.5 Dementia Outcome Codes (31 Read v2 codes)
```
F110., Eu00., Eu01., Eu02z, Eu002, E00.., Eu023, Eu00z, Eu025, Eu01z,
E001., F1100, Eu001, E004., Eu000, Eu02., Eu013, E000., Eu01y, E001z,
F1101, Eu020, E004z, E0021, Eu02y, Eu012, Eu011, E00z., E0040, E003., E0020
```

### 3.6 HES Dementia ICD-10 Codes (for label augmentation)
```
Prefixes: F00, F01, F02, G30
Exact: F03
```
These are used to identify dementia diagnoses in HES records to relabel censored GP patients.

---

## 4. Model Architecture

### 4.1 Overview
```
Input Sequence → DataEmbeddingLayer → Transformer Blocks (x6) → Survival Head
                                                                      ↓
                                                              Competing Risk Output
                                                              (DeSurv distribution)
```

### 4.2 DataEmbeddingLayer (`data_embedding_layer.py`)
- **Token embedding**: `nn.Embedding(vocab_size, n_embd)` — vocab_size=108,118, n_embd=384
- **Positional encoding**: Temporal positional encoding based on event dates
- **Static projection**: `nn.Linear(num_static_covariates, n_embd)` — projects patient-level features
  - Static embedding is added to ALL token embeddings in the sequence
  - For baseline: `nn.Linear(27, 384)` (27 demographic features)
  - For hes_static v1: `nn.Linear(35, 384)` (27 demographic + 8 HES features)
  - For hes_static v2: `nn.Linear(49, 384)` (27 demographic + 22 HES features)

### 4.3 Static Covariates (27 baseline dimensions)
The `_parquet_row_to_static_covariates()` function in `foundational_loader.py` builds:
1. **SEX**: One-hot encoded (multiple categories)
2. **IMD**: One-hot encoded (deprivation index quintiles)
3. **ETHNICITY**: One-hot encoded (multiple categories)
4. **YEAR_OF_BIRTH**: Continuous, scaled
5. Total: 27 dimensions after one-hot encoding

For hes_static, 8 additional HES features are appended (see Section 7.3).

### 4.4 Transformer Configuration
```yaml
block_type: "Neo"          # GPT-Neo style blocks
block_size: 512            # Context window (max sequence length)
n_layer: 6                 # Number of transformer layers
n_head: 6                  # Number of attention heads
n_embd: 384                # Embedding dimension
dropout: 0.1               # Dropout rates
attention_type: "global"   # Global attention (not local)
```

### 4.5 Survival Head
- **Type**: Competing Risk (`SurvLayer: "cr"`)
- **Method**: DeSurv (deep survival distribution estimation)
- **Risk 0**: Dementia (31 codes)
- **Risk 1**: Death (1 code: "DEATH")
- **Time scaling**: `supervised_time_scale: 5.0` — scales target ages for training

### 4.6 Sequence Processing Pipeline
1. Patient events sorted chronologically
2. **Repeating events deduplication** (`repeating_events=False`): Keeps only last occurrence of each event code
3. **Truncation** to `block_size=512`: Keeps the **last** 512 tokens (most recent events)
4. **Global diagnoses rescue** (`global_diagnoses=True`): After truncation, diagnosis events from the truncated prefix are collected and prepended to the sequence (preventing loss of historical diagnoses)
5. Outcome event appended at end of sequence (or censoring marker)

### 4.7 Dual-Backbone Architecture (BEST)

```
Patient i:
  GP data → [GP Backbone (pretrained, 108K vocab, 512 block)] → h_gp (384-dim) ─┐
                                                                                  ├─ Gated Fusion → h_fused (384-dim) → ODESurvCR Head
  HES data → [HES Backbone (pretrained, 1.5K vocab, 256 block)] → h_hes (384-dim) ─┘
```

**Key components**:
- **GP backbone**: TTETransformer, 108,118 vocab (Read v2), block_size=512, 49-dim static covariates (27 base + 22 HES summary)
- **HES backbone**: TTETransformer, 1,501 vocab (ICD-10 3-char), block_size=256, 27-dim static covariates
- **Gated Fusion**: `gate = σ(W_g · [h_gp; h_hes])`, `h_fused = gate * W_gp(h_gp) + (1-gate) * W_hes(h_hes)`
- **No HES patients**: h_hes = zero vector, fusion learns to rely on h_gp
- **Total params**: 106M trainable (2x backbone + fusion + survival head)
- **Model file**: `CPRD/src/models/survival/task_heads/dual_backbone.py`

**Data flow at runtime** (via `DualCollateWrapper` in `dual_data_module.py`):
1. GP DataModule loads GP sequences normally (reuses hes_static dataset)
2. HES sequence cache (419,966 patients' ICD-10 sequences) loaded into memory at startup
3. `DualCollateWrapper` wraps collate_fn: for each patient in batch, looks up HES sequence from cache
4. Batch contains both GP inputs (`tokens`, `ages`, `values`, `static_covariates`, `attention_mask`) and HES inputs (`hes_tokens`, `hes_ages`, `hes_values`, `hes_static_covariates`, `hes_attention_mask`)

### 4.8 Key Architectural Insight
The `block_size=512` truncation is a critical constraint:
- GP-only sequences: median ~151 tokens, comfortably fits
- After HES fusion: median ~731 tokens, ~60% get truncated
- This truncation destroys the temporal patterns the pretrained backbone learned
- **This is why sequence fusion fails and late fusion (dual-backbone / static features) works better**

---

## 5. Training Pipeline

### 5.1 Pretrain → Fine-tune Flow
```
1. GP PRETRAIN (already done):
   config_CompetingRisk11M.yaml
   → Self-supervised next-event prediction on full GP dataset
   → Checkpoint: crPreTrain_small_1337.ckpt
   → batch_size=16, 15 epochs, ~11M events

2. HES PRETRAIN (already done):
   config_HES_Pretrain.yaml
   → Self-supervised next-event prediction on HES ICD-10 sequences
   → Checkpoint: crPreTrain_HES_1337.ckpt
   → batch_size=64, 8 epochs, ~3.9M events

3a. SINGLE-BACKBONE FINE-TUNE:
   config_FineTune_Dementia_CR_*.yaml
   → Load GP pretrained backbone
   → Add competing-risk survival head
   → Train on dementia cohort with supervision
   → Checkpoint: crPreTrain_small_1337_FineTune_*.ckpt

3b. DUAL-BACKBONE FINE-TUNE (BEST):
   config_FineTune_Dementia_CR_dual.yaml
   → Load GP + HES pretrained backbones
   → Add gated fusion layer + competing-risk survival head
   → Train on dementia cohort with GP+HES inputs
   → Checkpoint: crPreTrain_small_1337_FineTune_Dementia_CR_dual.ckpt

4. EVAL:
   config_*_eval.yaml
   → Load fine-tuned checkpoint
   → Run test set evaluation only (train=False, test=True)
   → IMPORTANT: Always use single GPU for eval
```

### 5.2 run_experiment.py Key Logic
- **Line 64**: `cfg.data.num_static_covariates = batch['static_covariates'].shape[1]` — dynamically reads covariate dimension from data
- **Line 256-280**: Fine-tune mode selection:
  - `load_from_finetune`: Resume from fine-tuned checkpoint (for eval)
  - `load_from_pretrain`: Load pretrained backbone, create new survival head
  - `no_load`: Train from scratch (SFT mode)
- **Line 289-293**: `last.ckpt` auto-resume — PyTorch Lightning automatically resumes from `last.ckpt` if it exists. **CRITICAL**: Always delete `last.ckpt` before starting a new training run!

### 5.3 setup_finetune_experiment.py Key Logic
- **`load_from_pretrain` case**: Handles checkpoint weight loading with size mismatch support
  - Creates model with current config dimensions
  - Manually loads checkpoint, handling `static_proj` size changes
  - Pretrained weights for original 27 columns preserved; new columns zero-initialized
  - Uses `strict=False` for `load_state_dict`

### 5.4 Training Hyperparameters (hes_static, BEST)
```yaml
batch_size: 32
accumulate_grad_batches: 16  # Effective batch = 32*16 = 512 (overridden at runtime for single-GPU)
learning_rate: 5e-5          # Backbone LR
head_learning_rate: 5e-4     # Head LR (10x backbone)
num_epochs: 25
early_stop: True
early_stop_patience: 10
scheduler: reduceonplateau
learning_rate_decay: 0.8
val_check_interval: 1.0
seed: 1337
```

### 5.5 GPU Usage
- **4-GPU DDP**: `CUDA_VISIBLE_DEVICES=0,1,2,3` (when all GPUs available)
- **Single GPU**: `CUDA_VISIBLE_DEVICES=0` with higher `accumulate_grad_batches` to compensate
- **Eval**: Always single GPU (`CUDA_VISIBLE_DEVICES=0`)
- **Hardware**: NVIDIA GPUs with ~24GB VRAM each

### 5.6 WandB Logging
- **Project**: SurvivEHR
- **Entity**: swangek2002-hong-kong-university-of-science-and-technology
- **Dashboard**: https://wandb.ai/swangek2002-hong-kong-university-of-science-and-technology/SurvivEHR

---

## 6. Experiment History & Results

### 6.1 Complete Results Table

#### Pretraining Phase (March 2026)
Many early runs were pretraining experiments with `config_CompetingRisk11M`, testing batch sizes, optimizers, adaptive softmax, etc. The final pretrained checkpoint is `crPreTrain_small_1337.ckpt`.

#### Ablation Studies (March 2026)

| Date | Config | Description | Dementia C_td | Death C_td | Overall C_td |
|------|--------|-------------|---------------|------------|--------------|
| 03-21 | FineTune_Dementia_CR | Baseline with SAW | 0.xxx | 0.xxx | 0.xxx |
| 03-22 | FineTune_Dementia_CR_noSAW | No sample-aware weighting | improved | improved | improved |
| 03-22 | FineTune_Dementia_CR_SFT | Scratch fine-tune (no pretrain) | lower | - | - |
| 03-24 | FineTune_Dementia_CR_Combined | Combined approach | - | - | - |
| 03-26 | FineTune_Dementia_CR_idx60 | Index age 60 | - | - | - |
| 03-28 | FineTune_Dementia_CR_idx70 | Index age 70 (train) | - | - | - |
| 03-29 | FineTune_Dementia_CR_idx70 (eval) | Index age 70 (eval) | 0.762 | 0.927 | 0.899 |
| 03-29 | FineTune_Dementia_CR_idx74 | Index age 74 (train) | - | - | - |
| 03-30 | FineTune_Dementia_CR_idx74 (eval) | Index age 74 (eval) | 0.825 | 0.939 | 0.910 |
| 03-30 | FineTune_Dementia_CR_idx75 | Index age 75 (train) | - | - | - |
| 03-31 | FineTune_Dementia_CR_idx75 (eval) | Index age 75 (eval) | 0.786 | 0.946 | 0.925 |

**Key ablation finding**: Pretraining is useful (SFT underperforms), SAW is NOT useful (disabling SAW improves results).

#### 5-Fold Cross-Validation at Index Age 68 (April 2026)

| Fold | Dementia C_td | Death C_td | Overall C_td |
|------|---------------|------------|--------------|
| 0 (train) | 0.778 | 0.814 | 0.804 |
| 0 (eval) | 0.709 | 0.821 | 0.788 |
| 1 (train) | 0.744 | 0.824 | 0.788 |
| 1 (eval) | 0.712 | 0.830 | 0.799 |
| 2 (train) | 0.737 | 0.835 | 0.805 |
| 2 (eval) | 0.712 | 0.828 | 0.796 |
| 3 (train) | 0.657 | 0.838 | 0.799 |
| 3 (eval) | 0.688 | 0.816 | 0.791 |
| 4 (train) | 0.683 | 0.818 | 0.784 |
| 4 (eval) | 0.690 | 0.821 | 0.787 |
| **Mean (eval)** | **~0.702** | **~0.823** | **~0.792** |

#### HES Integration Experiments (April 2026) - KEY RESULTS

| Date | Experiment | Test Population | Dementia C_td | Death C_td | Overall C_td | Notes |
|------|-----------|----------------|---------------|------------|--------------|-------|
| 04-08 | hes_aug (train) | GP patients (8,292) | 0.713 | 0.968 | 0.859 | Training run |
| 04-09 | hes_aug (eval, run 1) | GP patients (8,292) | 0.733 | 0.944 | 0.858 | **Baseline** |
| 04-09 | hes_aug (retrain) | GP patients (8,292) | 0.740 | 0.966 | 0.851 | Second training run |
| 04-10 | hes_aug (eval, run 2) | GP patients (8,292) | 0.731 | 0.945 | 0.854 | Eval of second training |
| 04-11 | hes_fusion v5 (first attempt) | All (22,000+) | 0.547 | 0.735 | 0.711 | Failed - didn't converge |
| 04-11 | hes_fusion v5 (retrain) | All (22,000+) | 0.690 | 0.883 | 0.792 | Training metrics |
| 04-14 | hes_fusion v5 (eval) | All (22,000+) | **0.720** | 0.898 | 0.788 | Below baseline! |
| 04-17 | hes_fusion v5 (subset eval) | GP-only (8,292) | **0.684** | 0.901 | 0.790 | Same patients, much worse |
| 04-17 | hes_static v1 (train+eval, LEAKY) | GP patients (8,292) | ~~0.836~~ | ~~0.944~~ | ~~0.885~~ | ⚠️ **INVALID** — temporal leakage |

#### hes_static_v2 Experiment (April 2026) - ⚠️ INVALIDATED BY TEMPORAL LEAKAGE

| Date | Experiment | Test Population | Dementia C_td | Death C_td | Overall C_td | Notes |
|------|-----------|----------------|---------------|------------|--------------|-------|
| 04-27~28 | hes_static_v2 (train, LEAKY) | GP patients (8,292) | — | — | — | ⚠️ INVALID: trained on leaky features |
| 04-28 | hes_static_v2 (eval, LEAKY) | GP patients (8,292) | ~~0.875~~ | ~~0.961~~ | ~~0.915~~ | ⚠️ **INVALID** — temporal leakage in features |
| 04-29 | hes_static_v2 (eval, leaky model + clean test) | GP patients (8,292) | 0.706 | — | — | Model trained on leaky data evaluated on clean test → BELOW baseline |

**hes_static_v2 results are INVALID** — see Section 6.3 for temporal leakage details.

#### Dual-Backbone Experiments (April 2026) - ⚠️ v1 INVALIDATED, v2 CORRECTED

| Date | Experiment | Test Population | Dementia C_td | Death C_td | Overall C_td | Notes |
|------|-----------|----------------|---------------|------------|--------------|-------|
| 04-22 | HES pretrain | — | — | — | — | test_loss=2.407, 8 epochs (NOT affected by leakage) |
| 04-23~24 | Dual fine-tune v1 (train, LEAKY) | GP patients (8,292) | — | — | — | ⚠️ INVALID: trained on leaky 8-dim features |
| 04-24 | Dual fine-tune v1 (eval, LEAKY) | GP patients (8,292) | ~~0.845~~ | ~~0.949~~ | ~~0.891~~ | ⚠️ **INVALID** — temporal leakage |
| 04-29 | Dual fine-tune v1 (eval, leaky model + clean test) | GP patients (8,292) | 0.704 | — | — | Model trained on leaky data evaluated on clean test → BELOW baseline |
| 04-30~05-01 | **Dual fine-tune v2 (train, CLEAN)** | GP patients (8,292) | — | — | — | 22-dim clean features, trained on temporally-correct data |
| 05-01 | **Dual fine-tune v2 (eval, CLEAN)** | GP patients (8,292) | **0.743** | **0.951** | **0.855** | ✅ **VALID** — clean train + clean test, +0.010 vs baseline |

### 6.2 Summary of Best Results by Approach

| Approach | Dementia C_td | Death C_td | Overall C_td | vs Baseline | Status |
|----------|---------------|------------|-------------|-------------|--------|
| hes_aug (GP + HES labels) | 0.733 | 0.944 | 0.858 | baseline | ✅ Validated (no leakage) |
| hes_fusion v5 (sequence fusion) | 0.720 | 0.898 | 0.788 | -0.013 | ✅ FAILED (no leakage, but bad approach) |
| hes_static v1 (GP + 8 HES static, LEAKY) | ~~0.836~~ | ~~0.944~~ | ~~0.885~~ | ~~+0.103~~ | ⚠️ **INVALID — temporal leakage** |
| dual-backbone v1 (GP + HES, 8-dim static, LEAKY) | ~~0.845~~ | ~~0.949~~ | ~~0.891~~ | ~~+0.112~~ | ⚠️ **INVALID — temporal leakage** |
| hes_static v2 (GP + 22 HES static, LEAKY) | ~~0.875~~ | ~~0.961~~ | ~~0.915~~ | ~~+0.142~~ | ⚠️ **INVALID — temporal leakage** |
| **dual-backbone v2 (GP + HES, 22-dim static, CLEAN)** | **0.743** | **0.951** | **0.855** | **+0.010** | ✅ **BEST (corrected)** |

### 6.3 ⚠️ CRITICAL: Temporal Leakage Discovery (2026-04-29)

#### What Happened

On 2026-04-29, a **critical temporal leakage bug** was discovered in `build_hes_summary_features.py`. The script computed per-patient HES summary features (comorbidity flags, admission counts, etc.) using **ALL** HES records, including those recorded **AFTER** the patient's index date (age 72). In survival analysis, features must only use information available at or before the prediction time point (index date). Using future information constitutes temporal leakage.

#### Root Cause

The original `build_hes_summary_features.py` did not filter HES records by the patient's index date:
- It read all records from `hesin.csv` and `hesin_diag.csv` without date filtering
- Feature 21 (`HES_YEARS_SINCE_LAST_ADMISSION`) was computed relative to a fixed `STUDY_END_DATE` (2022-10-31), not relative to the patient-specific index date
- This affected ALL HES static feature variants (v1 8-dim and v2 22-dim) and ALL models that used them (hes_static, dual-backbone)

#### Scale of Leakage

When temporal filtering was applied (keeping only records with `admidate < index_date`):
- **Admission records**: 4,238,372 → 2,990,537 (70.6% kept, **29.4% were post-index**)
- **Diagnosis records**: 17,421,258 → 10,419,676 (59.8% kept, **40.2% were post-index**)
- **Delirium (F05) prevalence**: 2.0% → 0.5% (**75% of delirium diagnoses were post-index!**)
- Many comorbidity prevalences dropped substantially after filtering

The leakage was especially severe for conditions that manifest late in life (delirium, CKD, heart failure), as these are more likely to be diagnosed after age 72.

#### Why It Inflated Results

1. **Train + Test both leaked**: Both training and test data had leaky features, so the model learned to exploit future information during training, and the same future information was available at test time. This artificially inflated discrimination metrics.
2. **Leaky model on clean test drops BELOW baseline**: When a model trained on leaky data was evaluated on correctly-filtered test data, it performed WORSE than the baseline (0.706 and 0.704 vs 0.733 baseline), because the model learned shortcuts from future information that weren't available in the clean test data.
3. **Baseline (hes_aug) was NOT affected**: The hes_aug approach only uses HES for label augmentation (identifying dementia diagnoses), not for feature computation. Its C_td=0.733 remains valid.

#### The Fix

The temporal filtering was added to `build_hes_summary_features.py` (committed 2026-04-29):

```python
# Step 0: Load year-of-birth to compute per-patient index dates
yob_lookup = load_year_of_birth_lookup()  # from GP database
index_dates = {}
for pid, yob in yob_lookup.items():
    index_dates[str(pid)] = yob + pd.DateOffset(years=INDEX_ON_AGE)

# Step 1: Filter hesin.csv — only keep admissions BEFORE index date
hesin["index_date"] = hesin["eid"].map(index_dates)
hesin = hesin.dropna(subset=["index_date"])
hesin = hesin[hesin["admidate_dt"] < hesin["index_date"]]

# Step 2: Filter hesin_diag.csv — only keep diagnoses from pre-index admissions
pre_index_admissions = set(zip(hesin["eid"], hesin["dnx_hesin_id"]))
diag_filtered = diag[
    diag.apply(lambda r: (r["eid"], r["dnx_hesin_id"]) in pre_index_admissions, axis=1)
]

# Feature 21: Now relative to patient's index date (not STUDY_END_DATE)
years_since = (idx_date - last_adm).days / 365.25
```

#### Impact on All Experiments

| Experiment | Reported C_td | Actual C_td (clean) | Status |
|-----------|--------------|--------------------:|--------|
| hes_aug (baseline) | 0.733 | 0.733 | ✅ Valid (not affected) |
| hes_fusion v5 | 0.720 | 0.720 | ✅ Valid (not affected) |
| hes_static v1 (8-dim) | 0.836 | unknown (not retrained) | ⚠️ INVALID |
| hes_static v2 (22-dim) | 0.875 | unknown (not retrained) | ⚠️ INVALID |
| dual v1 (8-dim static) | 0.845 | unknown (not retrained) | ⚠️ INVALID |
| **dual v2 (22-dim, clean train+test)** | — | **0.743** | ✅ **VALID** |

#### Corrected Best Result

**Dual-backbone v2 (clean)**: Trained and tested on temporally-correct 22-dim HES static features.
- Dementia C_td = **0.743** (vs 0.733 baseline, **+0.010**)
- Death C_td = **0.951**
- Overall C_td = **0.855**
- WandB run: `crPreTrain_small_1337_FineTune_Dementia_CR_dual` (run ID: `0kdxj23q`)

The real improvement from HES static features is **+0.010**, not the previously reported +0.103 to +0.142. While modest, this improvement is genuine and obtained without any data leakage.

---

## 7. HES Integration Approaches

### 7.1 Approach 1: HES Label Augmentation (hes_aug) - BASELINE

**Concept**: Use HES dementia diagnoses to correct GP censoring. Many patients appear censored in GP data but actually have a dementia diagnosis recorded in hospital records.

**Mechanism**:
1. Build GP-only dataset normally via FoundationalDataModule
2. Build lookup: `{patient_id: earliest_HES_dementia_date}` from `hesin.csv` + `hesin_diag.csv`
3. Post-process parquet files: For each censored patient (no dementia/death in GP):
   - If patient has HES dementia diagnosis after index date and before study end
   - Replace last event code with `Eu02z` (unspecified dementia Read v2 code)
   - Replace last event date with HES dementia date
4. Effectively "relabels" censored patients as dementia events

**Results**: 4,097 patients relabeled out of ~134K total.

**Files**:
- `build_dementia_cr_hes_aug.py` — Dataset builder
- `hes_dementia_lookup.py` — HES dementia lookup builder
- `config_FineTune_Dementia_CR_hes_aug.yaml` — Training config
- `config_FineTune_Dementia_CR_hes_aug_eval.yaml` — Eval config

### 7.2 Approach 2: HES Full Sequence Fusion v5 (FAILED)

**Concept**: Translate all HES events to Read v2 codes via OMOP mapping and insert them into GP sequences.

**Implementation**:
1. Build OMOP mapping: `omop_icd10_to_readv2.pickle` (ICD-10 → Read v2 via OMOP CDM)
2. Override mapping for dementia: ICD-10 F00-F03, G30 → mapped to `Eu02z` (unclassified dementia Read v2)
3. Translate HES events to Read v2, align dates (ensure datetime format consistency)
4. Build fused SQLite database (`example_exercise_database_hes_fusion.db`) merging GP + translated HES events
5. Build dataset from fused DB with larger test population (includes HES-only patients)

**Why it failed**:
- **Modality clash**: HES events (hospital, sparse, acute) have different patterns than GP events (primary care, frequent, longitudinal)
- **Sequence bloating**: GP-only patients had median 151 tokens → after fusion, median 731 tokens
- **Truncation damage**: With `block_size=512`, ~60% of fused sequences get truncated, losing the most recent (and most predictive) GP events
- **Backbone corruption**: Training on mixed modality sequences damages the pretrained GP temporal patterns
- **Decisive evidence**: Subset eval on the SAME 8,292 GP patients scored 0.684 (fusion model) vs 0.733 (hes_aug), proving the fusion training itself is harmful

**Files**:
- `build_omop_mapping.py`, `build_hes_events.py`, `prepare_hes_fusion_db.py`
- `build_dementia_cr_hes_fusion.py`, `verify_hes_fusion_dataset.py`, `create_subset_eval.py`
- `config_FineTune_Dementia_CR_hes_fusion.yaml`, `config_FineTune_Dementia_CR_hes_fusion_eval.yaml`
- `config_FineTune_Dementia_CR_hes_fusion_subset_eval.yaml`

### 7.3 Approach 3: HES Static Covariates (hes_static)

**Concept**: Don't touch GP sequences. Condense HES information into summary statistics as additional static covariates. Combine with label augmentation from hes_aug.

**Implementation**:
1. Extract per-patient HES features from `hesin.csv` + `hesin_diag.csv` — **⚠️ MUST filter by index date** (see Section 6.3)
2. Build GP-only dataset (same as hes_aug)
3. Post-process: Add HES_* columns + apply HES label augmentation
4. Model's `static_proj` layer expands from `nn.Linear(27, 384)` to `nn.Linear(49, 384)` (v2)
5. Pretrained weights partially loaded: first 27 columns from checkpoint, remaining zero-initialized

**Critical design decisions**:
- Dementia-related ICD-10 codes (F00-F03, G30) are EXCLUDED from HES features to avoid label leakage
- **Only pre-index-date HES records are used** (temporal filtering added 2026-04-29 to fix leakage bug)

#### v1: 8 HES Features (⚠️ original result 0.836 INVALID due to temporal leakage)

| # | Feature | Type | ICD-10 Codes | Normalization | Rationale |
|---|---------|------|-------------|---------------|-----------|
| 0 | `HES_TOTAL_ADMISSIONS` | Continuous | — | log(1+count)/log(51), cap 1.0 | Hospitalization burden |
| 1 | `HES_TOTAL_UNIQUE_DIAG` | Continuous | — | log(1+count)/log(101), cap 1.0 | Diagnostic complexity |
| 2 | `HES_HAS_STROKE` | Binary | I60-I69 | 0/1 | Vascular dementia risk |
| 3 | `HES_HAS_MI` | Binary | I21-I22 | 0/1 | Cardiovascular risk |
| 4 | `HES_HAS_HEART_FAILURE` | Binary | I50 | 0/1 | Cardiovascular risk |
| 5 | `HES_HAS_DIABETES` | Binary | E10-E14 | 0/1 | Known dementia risk factor |
| 6 | `HES_HAS_DELIRIUM` | Binary | F05 | 0/1 | Strong dementia predictor |
| 7 | `HES_HAS_TBI` | Binary | S06 | 0/1 | Known dementia risk factor |

#### v2: 22 HES Features (⚠️ original result 0.875 INVALID; corrected dual v2 result: 0.743)

Expanded from 8 to 22 dimensions: 8 original + 11 new comorbidities + 3 new continuous features.

**New 11 comorbidity features (indices 8-18)**:

| # | Feature | Type | ICD-10 Codes | Prevalence | Clinical Rationale |
|---|---------|------|-------------|------------|-------------------|
| 8 | `HES_HAS_HYPERTENSION` | Binary | I10-I15 | 36.2% | Vascular dementia core risk; midlife hypertension strongly associated with late-life dementia |
| 9 | `HES_HAS_ATRIAL_FIBRILLATION` | Binary | I48 | 9.2% | Increases vascular dementia risk via stroke pathway |
| 10 | `HES_HAS_CKD` | Binary | N18 | 5.5% | Chronic kidney disease accelerates cognitive decline |
| 11 | `HES_HAS_DEPRESSION` | Binary | F32, F33 | 7.5% | Both prodromal symptom and independent risk factor |
| 12 | `HES_HAS_PARKINSON` | Binary | G20 | 1.0% | Strongly associated with Lewy body dementia |
| 13 | `HES_HAS_EPILEPSY` | Binary | G40, G41 | 1.7% | Bidirectional causal relationship with dementia |
| 14 | `HES_HAS_OBESITY` | Binary | E66 | 9.0% | Midlife obesity increases dementia risk |
| 15 | `HES_HAS_HYPERLIPIDEMIA` | Binary | E78 | 18.7% | Cardiovascular risk chain |
| 16 | `HES_HAS_COPD` | Binary | J44 | 5.3% | Hypoxia-mediated cognitive impairment |
| 17 | `HES_HAS_ALCOHOL` | Binary | F10 | 2.5% | Alcohol-related dementia |
| 18 | `HES_HAS_SLEEP_DISORDER` | Binary | G47 | 2.9% | Sleep apnea associated with dementia |

**New 3 continuous features (indices 19-21)**:

| # | Feature | Type | Source | Normalization | Rationale |
|---|---------|------|--------|---------------|-----------|
| 19 | `HES_MEAN_STAY_DAYS` | Continuous | mean(disdate - admidate) | log(1+days)/log(31), cap 1.0 | Hospitalization severity |
| 20 | `HES_EMERGENCY_RATIO` | Continuous | emergency_count / total | Direct [0,1] | Acute illness burden |
| 21 | `HES_YEARS_SINCE_LAST_ADMISSION` | Continuous | (index_date - last_admidate).years | min(years/20, 1.0) | Recent health status; lower = more recent hospital use. **Note**: computed relative to patient's index date (age 72), NOT study end date |

**⚠️ IMPORTANT**: All prevalence statistics in this section and Appendix B are from the **leaky** (unfiltered) feature extraction. After temporal filtering, prevalences are lower. See Appendix B for both leaky and clean statistics.

**v1 → v2 comparison**: ⚠️ All results below are from leaky features and are INVALID. Only the corrected dual v2 result (clean train + clean test) is valid.

| Metric | v1 8-dim (LEAKY) | v2 22-dim (LEAKY) | v2 22-dim Dual (CLEAN) |
|--------|:-----------------:|:------------------:|:----------------------:|
| num_static_covariates | 35 (27+8) | 49 (27+22) | 49 (27+22) |
| Dementia C_td | ~~0.836~~ ⚠️ | ~~0.875~~ ⚠️ | **0.743** ✅ |
| Death C_td | ~~0.944~~ ⚠️ | ~~0.961~~ ⚠️ | **0.951** ✅ |
| Overall C_td | ~~0.885~~ ⚠️ | ~~0.915~~ ⚠️ | **0.855** ✅ |

**Checkpoint loading** (same mechanism for v1 and v2):
```python
# In setup_finetune_experiment.py, load_from_pretrain case:
# Pretrained static_proj: weight shape (384, 27), bias shape (384,)
# New model static_proj: weight shape (384, 49), bias shape (384,)  [v2]
# Solution: Copy first 27 columns from pretrained, zero-init remaining
new_weight[:384, :27] = pretrained_weight[:384, :27]
new_weight[:384, 27:] = 0  # New HES features start from zero
```

**Files**:
- `build_hes_summary_features.py` — Feature extractor (v2: 22 dims)
- `build_dementia_cr_hes_static.py` — Dataset builder
- `config_FineTune_Dementia_CR_hes_static.yaml` — Training config (num_static_covariates=49)
- `config_FineTune_Dementia_CR_hes_static_eval.yaml` — Eval config
- `run_hes_static_pipeline.sh` — Full pipeline script

**Dataloader modification** (`foundational_loader.py`):
```python
# At end of _parquet_row_to_static_covariates(), before np.hstack:
hes_cols = sorted(col for col in row_df.index if col.startswith("HES_"))
for col in hes_cols:
    val = float(row_df.get(col, 0.0))
    covariates.append(np.asarray(val).reshape((1, -1)))
# Backward-compatible: if no HES_* columns exist, nothing is appended
```

### 7.4 Approach 4: Dual-Backbone Architecture (BEST APPROACH, CORRECTED)

**Concept**: Give GP and HES each their own independent transformer backbone, pretrained separately on their respective modalities, then fuse hidden states via a gated fusion layer during fine-tuning. This avoids the modality clash of sequence fusion while leveraging richer temporal patterns than static features alone.

**Architecture**:
```
GP序列 → [GP Backbone (pretrained on GP)]  → h_gp (384-dim)  ─┐
                                                                ├─ Gated Fusion → h_fused → Survival Head
HES序列 → [HES Backbone (pretrained on HES)] → h_hes (384-dim) ─┘
```

**Implementation (3 phases)**:

**Phase 1: HES Pretrain** (see `PLAN_DUAL_MODEL_ARCHITECTURE.md` Section 3.7)
1. Build HES-only SQLite DB from `hesin.csv` + `hesin_diag.csv` (ICD-10 3-char truncation, level=1 primary diagnosis only)
2. Build HES Pretrain Parquet dataset via `FoundationalDataModule`
3. Self-supervised next-event prediction on HES sequences
4. Result: `crPreTrain_HES_1337.ckpt` (12.2M params, test_loss=2.407)

**Phase 2: Dual Fine-tune**
1. Load GP DataModule (reuses hes_static dataset for GP sequences + labels)
2. Build HES sequence cache from HES DB (419,966 patients in memory)
3. Wrap collate_fn with `DualCollateWrapper` to inject HES inputs per-batch
4. Create `DualFineTuneExperiment` with both backbones + gated fusion + survival head
5. Load GP pretrain weights → `gp_transformer`, HES pretrain weights → `hes_transformer`
6. Fusion layer + survival head trained from scratch with 10x learning rate

**Phase 3: Test**
1. Load fine-tuned checkpoint
2. Single GPU evaluation on test set
3. ~~Result (LEAKY v1): Dementia C_td = 0.845~~ ⚠️ INVALID
4. **Result (CLEAN v2)**: Dementia C_td = **0.743** (+0.010 over baseline), Death C_td = 0.951, Overall C_td = 0.855

**Key design decisions**:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| HES coding | Original ICD-10 (not translated to Read v2) | Avoid translation loss; HES backbone learns ICD-10 semantics directly |
| Fusion type | Gated fusion | Dynamic weighting allows model to ignore HES when absent |
| Fusion timing | Fine-tune only (not pretrain) | Backbones pretrain independently; fusion learned from scratch |
| No HES patients | h_hes = zero vector | Gate learns to rely on h_gp when h_hes ≈ 0 |
| HES static features retained | GP backbone uses 49-dim static (27 base + 22 HES, v2) | Preserves validated signal from hes_static approach |
| Backbone LR vs Head LR | 5e-5 vs 5e-4 (10x) | Pretrained backbones need gentler updates |

**Training details**:
- batch_size=16, accumulate_grad_batches=32, effective_batch=512
- 22 epochs trained (~44 hours), best at epoch 13 (val_loss=0.007)
- 106M trainable params total (2× backbone + fusion + head)
- Model size: 1.1 GB checkpoint

**Files**:
- `build_hes_database.py` — HES SQLite DB builder
- `build_hes_pretrain_dataset.py` — HES Pretrain Parquet builder
- `config_HES_Pretrain.yaml` — HES pretrain config
- `dual_backbone.py` — DualBackboneSurvModel + FusionLayer
- `setup_dual_finetune_experiment.py` — DualFineTuneExperiment + checkpoint loading
- `dual_data_module.py` — DualCollateWrapper + HES tokenizer + HES cache
- `run_dual_experiment.py` — Dual-backbone entry point
- `config_FineTune_Dementia_CR_dual.yaml` — Training config
- `config_FineTune_Dementia_CR_dual_eval.yaml` — Eval config
- `run_dual_pipeline.sh` — Full pipeline script
- `PLAN_DUAL_MODEL_ARCHITECTURE.md` — Detailed architecture plan with implementation records

---

## 8. Key Lessons Learned

### 8.1 Architecture Lessons
1. **block_size=512 is a hard constraint**: Sequences longer than 512 tokens get truncated (last 512 kept). This limits how much additional information can be injected as sequence events.
2. **global_diagnoses=True helps but doesn't solve truncation**: It rescues diagnosis codes from truncated prefix but loses temporal ordering of those rescued events.
3. **Static covariates are a safe injection path**: They don't affect sequence length and are added uniformly to all token embeddings via linear projection.
4. **Pretrained backbone is sensitive**: Mixed-modality training can corrupt learned GP temporal patterns. The pretrain→finetune paradigm works best when the fine-tune data distribution matches pretrain.
5. **Late fusion > early fusion for multi-modal EHR**: Independent backbones with late fusion (gated) preserve each modality's pretrained patterns. Sequence-level fusion destroys them.
6. **Gated fusion handles missing modalities gracefully**: When h_hes is zero (no HES records), the gate learns to rely entirely on h_gp. No special handling needed.
7. **Incremental gains compound**: Static features + dual-backbone are complementary (dual-backbone reuses hes_static dataset as GP input). Corrected dual v2 achieves +0.010 over baseline.
8. **⚠️ Temporal leakage can create dramatic but false improvements**: Using HES records from after the index date inflated C_td by +0.103 to +0.142 — all of which was artificial. The true improvement from HES static features is ~+0.010. Always verify that features used for prediction only contain information available at prediction time.
9. **Leaky training + clean test = WORSE than baseline**: A model trained on temporally-leaky data learns to exploit future information as shortcuts. When tested on correctly-filtered data (without those shortcuts), performance drops BELOW the no-feature baseline (0.706 and 0.704 vs 0.733). This is the hallmark of data leakage.

### 8.2 Training Lessons
1. **Always delete `last.ckpt` before new training**: PyTorch Lightning auto-resumes from it, which can cause immediate termination if max_epochs already reached.
2. **SAW (Sample-Aware Weighting) hurts performance**: Ablation study confirmed disabling SAW improves results for this task.
3. **SFT (Scratch Fine-Tune) underperforms**: Pretraining provides substantial benefit; loading pretrained weights is important.
4. **Single GPU + higher accumulate_grad_batches works**: When not all GPUs are available, increase gradient accumulation to maintain effective batch size.
5. **DDP + occupied GPUs = OOM**: If other GPUs are in use, switch to single GPU training rather than risking OOM.
6. **Eval must use single GPU**: Multi-GPU DDP in test mode causes issues with metric aggregation. Always set `CUDA_VISIBLE_DEVICES=0` for eval.
7. **Dual-backbone training is 2x slower**: ~2 hours/epoch (vs ~1 hour for single backbone) due to doubled forward pass. Budget accordingly.
8. **Differential learning rates essential for dual fine-tune**: Pretrained backbones at 5e-5, new layers (fusion + head) at 5e-4. Prevents catastrophic forgetting of pretrained knowledge.

### 8.3 Data Lessons
1. **HES label augmentation provides ~2-3% C_td improvement** over pure GP labels (from ~0.71 to ~0.73).
2. **HES static features provide modest C_td improvement** (+0.010 with corrected temporal filtering). ~~Previous claims of +0.103 to +0.142 were due to temporal leakage.~~ The actual signal from pre-index HES features is real but small.
3. **Practice-based splitting prevents data leakage**: Patients from the same practice are always in the same split.
4. **Parquet files are nested**: Under `COUNTRY=UK/HEALTH_AUTH=*/PRACTICE_ID=*/CHUNK=*/`. Use `os.walk()` to traverse.
5. **⚠️ Temporal filtering is MANDATORY for survival analysis features**: Any feature derived from external data sources (HES, death records, etc.) MUST be filtered to only include records before the patient's index date. Failure to do so causes temporal leakage. In this project: `admidate < yob + INDEX_ON_AGE` for HES records.

### 8.4 Experimental Design Lessons
1. **Subset evaluation is essential**: When test populations differ between experiments, create a subset eval on the same patients for fair comparison.
2. **Check both train and eval metrics**: Training metrics can look good while eval reveals problems (e.g., fusion v5 train C_td=0.690 but eval on same patients=0.684).
3. **Feature selection matters**: The 8 HES features were chosen based on established clinical risk factors for dementia (cardiovascular risk, diabetes, delirium, TBI).
4. **⚠️ Suspiciously large improvements warrant investigation**: The jump from 0.733 to 0.836 (+0.103) from just 8 static features should have triggered deeper scrutiny. In retrospect, such dramatic gains from simple binary comorbidity flags were a signal of data leakage, not model effectiveness.
5. **Leakage detection method**: Test a model trained on potentially-leaky data against correctly-filtered test data. If performance drops BELOW the no-feature baseline, the model has learned leaky shortcuts. In this project: leaky model scored 0.706 vs baseline 0.733.

---

## 9. Configuration Reference

### 9.1 Available Configs (`CPRD/examples/modelling/SurvivEHR/confs/`)

| Config | Purpose | Key Differences |
|--------|---------|-----------------|
| `config_CompetingRisk11M.yaml` | Pretraining | Full GP dataset, self-supervised |
| `config_FineTune_Dementia_CR.yaml` | Baseline fine-tune | With SAW, idx72 |
| `config_FineTune_Dementia_CR_noSAW.yaml` | No SAW ablation | `sample_weighting.mode: null` |
| `config_FineTune_Dementia_CR_SFT.yaml` | Scratch fine-tune | No pretrained weights |
| `config_FineTune_Dementia_CR_Combined.yaml` | Combined approach | - |
| `config_FineTune_Dementia_CR_EventWeight.yaml` | Event weighting | - |
| `config_FineTune_Dementia_CR_idx60.yaml` | Index age 60 | `INDEX_ON_AGE=60` in dataset |
| `config_FineTune_Dementia_CR_idx70.yaml` | Index age 70 | `INDEX_ON_AGE=70` in dataset |
| `config_FineTune_Dementia_CR_idx74.yaml` | Index age 74 | `INDEX_ON_AGE=74` in dataset |
| `config_FineTune_Dementia_CR_idx75.yaml` | Index age 75 | `INDEX_ON_AGE=75` in dataset |
| `config_FineTune_Dementia_CR_idx68_cv_fold{0-4}.yaml` | 5-fold CV idx68 | Different practice splits |
| `config_FineTune_Dementia_CR_hes_aug.yaml` | HES label aug | GP seq + HES dementia labels |
| `config_FineTune_Dementia_CR_hes_fusion.yaml` | HES seq fusion | Fused DB, expanded test set |
| `config_FineTune_Dementia_CR_hes_static.yaml` | HES static v2 | `num_static_covariates=49` (22 HES features) |
| `config_HES_Pretrain.yaml` | HES backbone pretrain | ICD-10, block_size=256, vocab=1501 |
| `config_FineTune_Dementia_CR_dual.yaml` | **Dual-backbone (BEST, corrected)** | GP+HES backbones, gated fusion |
| Each `*_eval.yaml` | Eval variant | `train: False, test: True` |

### 9.2 Key Config Parameters

```yaml
# Data
data.batch_size: 32
data.num_static_covariates: 49          # 27 (baseline), 35 (hes_static v1), or 49 (hes_static v2)
data.supervised_time_scale: 5.0         # Target age scaling factor
data.global_diagnoses: True             # Rescue truncated diagnosis events
data.repeating_events: False            # Deduplicate events

# Training
optim.num_epochs: 25
optim.learning_rate: 5e-5               # Backbone LR
optim.accumulate_grad_batches: 4        # (override to 16 for single-GPU)
optim.early_stop: True
optim.early_stop_patience: 10
fine_tuning.head.learning_rate: 5e-4    # Head LR (10x backbone)
fine_tuning.sample_weighting.mode: null # SAW disabled (key finding)

# Transformer
transformer.block_size: 512
transformer.n_layer: 6
transformer.n_head: 6
transformer.n_embd: 384
```

---

## 10. File Reference

### 10.1 Checkpoints

| Checkpoint | Description | Status |
|-----------|-------------|--------|
| `crPreTrain_small_1337.ckpt` | Pretrained backbone (15 epochs) | Source for all fine-tunes |
| `crPreTrain_small_1337-epochepoch=09.ckpt` | Pretrain epoch 9 backup | Backup |
| `crPreTrain_small_1337_FineTune_Dementia_CR.ckpt` | Baseline fine-tune (with SAW) | Historic |
| `crPreTrain_small_1337_FineTune_Dementia_CR_noSAW.ckpt` | No SAW fine-tune | Ablation |
| `SFT_small_1337_FineTune_Dementia_CR.ckpt` | Scratch fine-tune (no pretrain) | Ablation |
| `crPreTrain_small_1337_FineTune_Dementia_CR_hes_aug.ckpt` | HES label augmentation | Baseline for HES experiments |
| `crPreTrain_small_1337_FineTune_Dementia_CR_hes_fusion.ckpt` | HES sequence fusion (FAILED) | Do not use |
| `crPreTrain_small_1337_FineTune_Dementia_CR_hes_static_v1.ckpt` | HES static v1 (8-dim, LEAKY) | ⚠️ INVALID — temporal leakage |
| `crPreTrain_small_1337_FineTune_Dementia_CR_hes_static.ckpt` | HES static v2 (22-dim, LEAKY) | ⚠️ INVALID — temporal leakage |
| `crPreTrain_HES_1337.ckpt` | HES backbone pretrained | 8 epochs, 12.2M params (NOT affected) |
| `crPreTrain_small_1337_FineTune_Dementia_CR_dual_v1.ckpt` | Dual-backbone v1 (8-dim static, LEAKY) | ⚠️ INVALID — temporal leakage |
| `crPreTrain_small_1337_FineTune_Dementia_CR_dual.ckpt` | **Dual-backbone v2 (22-dim static, CLEAN)** | ✅ **VALID** — C_td=0.743, clean train+test |
| `crPreTrain_small_1337_FineTune_Dementia_CR_idx{60,70,74,75}.ckpt` | Index age experiments | Index age ablation |
| `crPreTrain_small_1337_FineTune_Dementia_CR_idx68_cv_fold{0-4}.ckpt` | 5-fold CV | Cross-validation |
| `crPreTrain_small_1337_FineTune_Dementia_CR_Combined.ckpt` | Combined approach | Historic |

### 10.2 Key Pipeline Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `CPRD/run_hes_static_pipeline.sh` | Full hes_static pipeline | `bash run_hes_static_pipeline.sh` |
| `CPRD/run_dual_pipeline.sh` | **Dual-backbone pipeline** | `bash run_dual_pipeline.sh` |
| `CPRD/run_hes_fusion_pipeline.sh` | Full fusion pipeline (FAILED) | Not recommended |
| `CPRD/run_hes_fusion_train_only.sh` | Fusion train-only | Not recommended |

### 10.3 Environment

```bash
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
```

Python 3.10, PyTorch + PyTorch Lightning, Hydra for config management, WandB for experiment tracking.

### 10.4 Data Pipeline Commands

```bash
# Option B (hes_static) - RECOMMENDED
cd /Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR

# Step 1: Extract HES features (with temporal filtering — only pre-index records)
$PYTHON build_hes_summary_features.py

# Step 2: Build dataset
$PYTHON build_dementia_cr_hes_static.py

# Step 3: Train (single GPU, adjust accumulate_grad_batches if needed)
CUDA_VISIBLE_DEVICES=0 $PYTHON run_experiment.py \
    --config-name=config_FineTune_Dementia_CR_hes_static \
    optim.accumulate_grad_batches=16

# Step 4: Eval
CUDA_VISIBLE_DEVICES=0 $PYTHON run_experiment.py \
    --config-name=config_FineTune_Dementia_CR_hes_static_eval

# IMPORTANT: Before Step 3, always clean up stale checkpoints:
rm -f /Data0/swangek_data/991/CPRD/output/checkpoints/last.ckpt
rm -f /Data0/swangek_data/991/CPRD/output/checkpoints/crPreTrain_small_1337_FineTune_Dementia_CR_hes_static.ckpt

# Dual-backbone (BEST) — requires hes_static dataset + HES pretrain already done
cd /Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR

# Train (single GPU)
rm -f /Data0/swangek_data/991/CPRD/output/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=0 $PYTHON run_dual_experiment.py \
    --config-name=config_FineTune_Dementia_CR_dual

# Eval (MUST be single GPU)
CUDA_VISIBLE_DEVICES=0 $PYTHON run_dual_experiment.py \
    --config-name=config_FineTune_Dementia_CR_dual_eval
```

---

## Appendix A: Dataset Sizes

| Dataset | Train | Val | Test | Total |
|---------|-------|-----|------|-------|
| PreTrain (GP) | ~450K+ | ~50K+ | ~50K+ | ~550K+ |
| PreTrain_HES | 327,308 | 17,746 | 19,522 | 364,576 |
| FineTune_Dementia_CR (idx72) | ~120K | ~6K | ~8K | ~134K |
| FineTune_Dementia_CR_hes_aug | 119,694 | 5,823 | 8,292 | 133,809 |
| FineTune_Dementia_CR_hes_static | 119,694 | 5,823 | 8,292 | 133,809 |
| FineTune_Dementia_CR_hes_fusion | ~350K+ | ~20K+ | ~22K+ | ~392K+ |

## Appendix B: HES Feature Statistics (v2, 22 dims)

### ⚠️ Leaky vs Clean Feature Statistics

The following table shows feature statistics from BOTH the original leaky extraction (no temporal filtering) and the corrected clean extraction (only pre-index-date records). The differences highlight the scale of temporal leakage.

**Clean features** (used for corrected dual v2 model, 449,095 patients with pre-index HES records):

```
Feature                                   LEAKY           CLEAN (pre-index only)
                                       mean  nonzero     mean  nonzero    Change
HES_TOTAL_ADMISSIONS                  0.4851  1.000     lower  1.000     ↓ fewer admissions
HES_TOTAL_UNIQUE_DIAG                 0.5186  0.995     lower  ~0.99     ↓ fewer diagnoses
HES_HAS_STROKE                        0.0573  0.057     lower  lower     ↓ ~25% of strokes post-index
HES_HAS_MI                            0.0404  0.040     lower  lower     ↓
HES_HAS_HEART_FAILURE                 0.0447  0.045     lower  lower     ↓
HES_HAS_DIABETES                      0.1043  0.104     lower  lower     ↓
HES_HAS_DELIRIUM                      0.0203  0.020     ~0.005 ~0.005   ↓↓↓ 75% were post-index!
HES_HAS_TBI                           0.0077  0.008     lower  lower     ↓
HES_HAS_HYPERTENSION                  0.3618  0.362     lower  lower     ↓
HES_HAS_ATRIAL_FIBRILLATION           0.0921  0.092     lower  lower     ↓
HES_HAS_CKD                           0.0552  0.055     lower  lower     ↓
HES_HAS_DEPRESSION                    0.0749  0.075     lower  lower     ↓
HES_HAS_PARKINSON                     0.0095  0.010     lower  lower     ↓
HES_HAS_EPILEPSY                      0.0166  0.017     lower  lower     ↓
HES_HAS_OBESITY                       0.0897  0.090     lower  lower     ↓
HES_HAS_HYPERLIPIDEMIA                0.1874  0.187     lower  lower     ↓
HES_HAS_COPD                          0.0525  0.052     lower  lower     ↓
HES_HAS_ALCOHOL                       0.0252  0.025     lower  lower     ↓
HES_HAS_SLEEP_DISORDER                0.0286  0.029     lower  lower     ↓
HES_MEAN_STAY_DAYS                    0.1902  0.708     lower  lower     ↓
HES_EMERGENCY_RATIO                   0.2381  0.564     lower  lower     ↓
HES_YEARS_SINCE_LAST_ADMISSION        0.3155  0.999     higher higher    ↑ more distant from index
```

**Key observation**: Delirium (F05) is the most dramatically affected — 75% of delirium diagnoses occurred AFTER the index date. This makes clinical sense: delirium often occurs in late-stage dementia or during the final hospitalization. Including post-index delirium diagnoses gave the model a near-direct indicator of the outcome, which is why the leaky model performed so well.

**Notes**:
- Patients without any HES records get all-zero feature vectors
- Binary features (HES_HAS_*): 0 or 1
- Continuous features: Normalized to [0, 1]
- Exact clean statistics not captured in full detail; the key point is that all comorbidity prevalences decrease after temporal filtering, with delirium showing the most dramatic drop

## Appendix C: Glossary

| Term | Definition |
|------|-----------|
| C_td | Time-dependent concordance index (discrimination metric, higher = better) |
| IBS | Integrated Brier Score (calibration metric, lower = better) |
| INBLL | Integrated Negative Binomial Log-Likelihood (lower = better) |
| SAW | Sample-Aware Weighting (event weighting strategy, disabled in best models) |
| DeSurv | Deep Survival distribution estimation method |
| PEFT | Parameter-Efficient Fine-Tuning (not used in current best model) |
| LLRD | Layer-wise Learning Rate Decay (not used) |
| DDP | Distributed Data Parallel (multi-GPU training) |
| Read v2 | UK GP coding system (replaced by SNOMED CT in modern systems) |
| ICD-10 | International Classification of Diseases, 10th revision (hospital coding) |
| OMOP | Observational Medical Outcomes Partnership Common Data Model |
| CPRD | Clinical Practice Research Datalink (UK GP database) |
| HES | Hospital Episode Statistics (UK hospital records) |
| Index date | Date when patient reaches INDEX_ON_AGE (age 72) |
| Censored | Patient without observed outcome (right-censored in survival analysis) |
| Competing risk | Multiple possible outcomes (dementia vs death) |
