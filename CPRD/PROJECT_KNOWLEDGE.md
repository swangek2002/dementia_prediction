# SurvivEHR Project Knowledge Base

> **Last updated**: 2026-04-27
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

### Key Findings (as of 2026-04-24)
- **Best model**: Dual-backbone architecture (GP + HES backbones with gated fusion) achieves **Dementia C_td = 0.845**, a +0.112 improvement over the GP+HES-labels-only baseline (0.733)
- Previous best: hes_static (8 HES summary features) achieved C_td = 0.836 (+0.103)
- Dual-backbone adds +0.009 on top of hes_static by encoding full ICD-10 sequence temporal patterns via a separate HES transformer
- Sequence-level fusion of HES events into GP sequences **hurts** performance (0.720) due to modality clash and truncation
- Late fusion (independent backbones + gated fusion layer) is the correct approach for multi-modal EHR data

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
│   │   ├── hes_summary_features.pickle  # Per-patient HES summary features (8 dims, 449K patients)
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
│   │       ├── FineTune_Dementia_CR_hes_static/     # GP + HES labels + HES static features (BEST)
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
│   ├── run_dual_pipeline.sh         # Pipeline script for dual-backbone (BEST)
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
- (For hes_static): `HES_TOTAL_ADMISSIONS`, `HES_TOTAL_UNIQUE_DIAG`, `HES_HAS_STROKE`, `HES_HAS_MI`, `HES_HAS_HEART_FAILURE`, `HES_HAS_DIABETES`, `HES_HAS_DELIRIUM`, `HES_HAS_TBI` (float32)

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
  - For hes_static: `nn.Linear(35, 384)` (27 demographic + 8 HES features)

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
- **GP backbone**: TTETransformer, 108,118 vocab (Read v2), block_size=512, 35-dim static covariates (27 base + 8 HES summary)
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
| 04-17 | **hes_static (train+eval)** | GP patients (8,292) | **0.836** | **0.944** | **0.885** | **Previous best** |

#### Dual-Backbone Experiments (April 2026) - CURRENT BEST

| Date | Experiment | Test Population | Dementia C_td | Death C_td | Overall C_td | Notes |
|------|-----------|----------------|---------------|------------|--------------|-------|
| 04-22 | HES pretrain | — | — | — | — | test_loss=2.407, 8 epochs |
| 04-23~24 | Dual fine-tune (train) | GP patients (8,292) | — | — | — | 22 epochs, best at epoch 13, val_loss=0.007 |
| 04-24 | **Dual fine-tune (eval)** | GP patients (8,292) | **0.845** | **0.949** | **0.891** | **BEST RESULT** |

### 6.2 Summary of Best Results by Approach

| Approach | Dementia C_td | Death C_td | Overall C_td | vs Baseline | Status |
|----------|---------------|------------|-------------|-------------|--------|
| hes_aug (GP + HES labels) | 0.733 | 0.944 | 0.858 | baseline | Validated |
| hes_fusion v5 (sequence fusion) | 0.720 | 0.898 | 0.788 | -0.013 | FAILED |
| hes_static (GP + HES static features) | 0.836 | 0.944 | 0.885 | +0.103 | Previous best |
| **dual-backbone (GP + HES backbones, gated fusion)** | **0.845** | **0.949** | **0.891** | **+0.112** | **BEST** |

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

### 7.3 Approach 3: HES Static Covariates (hes_static) - BEST APPROACH

**Concept**: Don't touch GP sequences. Condense HES information into 8 summary statistics as additional static covariates. Combine with label augmentation from hes_aug.

**Implementation**:
1. Extract 8 per-patient HES features from `hesin.csv` + `hesin_diag.csv`
2. Build GP-only dataset (same as hes_aug)
3. Post-process: Add 8 HES_* columns + apply HES label augmentation
4. Model's `static_proj` layer expands from `nn.Linear(27, 384)` to `nn.Linear(35, 384)`
5. Pretrained weights partially loaded: first 27 columns from checkpoint, last 8 zero-initialized

**The 8 HES Features**:

| # | Feature | Type | ICD-10 Codes | Normalization | Rationale |
|---|---------|------|-------------|---------------|-----------|
| 1 | `HES_TOTAL_ADMISSIONS` | Continuous | — | log(1+count)/log(51), cap 1.0 | Hospitalization burden |
| 2 | `HES_TOTAL_UNIQUE_DIAG` | Continuous | — | log(1+count)/log(101), cap 1.0 | Diagnostic complexity |
| 3 | `HES_HAS_STROKE` | Binary | I60-I69 | 0/1 | Vascular dementia risk |
| 4 | `HES_HAS_MI` | Binary | I21-I22 | 0/1 | Cardiovascular risk |
| 5 | `HES_HAS_HEART_FAILURE` | Binary | I50 | 0/1 | Cardiovascular risk |
| 6 | `HES_HAS_DIABETES` | Binary | E10-E14 | 0/1 | Known dementia risk factor |
| 7 | `HES_HAS_DELIRIUM` | Binary | F05 | 0/1 | Strong dementia predictor |
| 8 | `HES_HAS_TBI` | Binary | S06 | 0/1 | Known dementia risk factor |

**Critical design decision**: Dementia-related ICD-10 codes (F00-F03, G30) are EXCLUDED from HES features to avoid label leakage (since hes_aug already uses them for label augmentation).

**Checkpoint loading**:
```python
# In setup_finetune_experiment.py, load_from_pretrain case:
# Pretrained static_proj: weight shape (384, 27), bias shape (384,)
# New model static_proj: weight shape (384, 35), bias shape (384,)
# Solution: Copy first 27 columns from pretrained, zero-init remaining 8
new_weight[:384, :27] = pretrained_weight[:384, :27]
new_weight[:384, 27:35] = 0  # New HES features start from zero
```

**Results**: Dementia C_td = **0.836** (+0.103 over baseline)

**Files**:
- `build_hes_summary_features.py` — Feature extractor
- `build_dementia_cr_hes_static.py` — Dataset builder
- `config_FineTune_Dementia_CR_hes_static.yaml` — Training config (num_static_covariates=35)
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

### 7.4 Approach 4: Dual-Backbone Architecture (BEST APPROACH)

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
3. Result: Dementia C_td = **0.845** (+0.009 over hes_static, +0.112 over baseline)

**Key design decisions**:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| HES coding | Original ICD-10 (not translated to Read v2) | Avoid translation loss; HES backbone learns ICD-10 semantics directly |
| Fusion type | Gated fusion | Dynamic weighting allows model to ignore HES when absent |
| Fusion timing | Fine-tune only (not pretrain) | Backbones pretrain independently; fusion learned from scratch |
| No HES patients | h_hes = zero vector | Gate learns to rely on h_gp when h_hes ≈ 0 |
| HES static features retained | GP backbone still uses 35-dim static (27 base + 8 HES) | Preserves validated signal from hes_static approach |
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
7. **Incremental gains compound**: hes_static (+0.103) captures bulk of HES signal; dual-backbone (+0.009 additional) extracts remaining temporal patterns. Both approaches are complementary (dual-backbone reuses hes_static dataset as GP input).

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
2. **HES static features provide an additional ~10% C_td improvement** (+0.103 from 0.733 to 0.836).
3. **Practice-based splitting prevents data leakage**: Patients from the same practice are always in the same split.
4. **Parquet files are nested**: Under `COUNTRY=UK/HEALTH_AUTH=*/PRACTICE_ID=*/CHUNK=*/`. Use `os.walk()` to traverse.

### 8.4 Experimental Design Lessons
1. **Subset evaluation is essential**: When test populations differ between experiments, create a subset eval on the same patients for fair comparison.
2. **Check both train and eval metrics**: Training metrics can look good while eval reveals problems (e.g., fusion v5 train C_td=0.690 but eval on same patients=0.684).
3. **Feature selection matters**: The 8 HES features were chosen based on established clinical risk factors for dementia (cardiovascular risk, diabetes, delirium, TBI).

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
| `config_FineTune_Dementia_CR_hes_static.yaml` | HES static | `num_static_covariates=35` |
| `config_HES_Pretrain.yaml` | HES backbone pretrain | ICD-10, block_size=256, vocab=1501 |
| `config_FineTune_Dementia_CR_dual.yaml` | **Dual-backbone (BEST)** | GP+HES backbones, gated fusion |
| Each `*_eval.yaml` | Eval variant | `train: False, test: True` |

### 9.2 Key Config Parameters

```yaml
# Data
data.batch_size: 32
data.num_static_covariates: 35          # 27 (baseline) or 35 (hes_static)
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
| `crPreTrain_small_1337_FineTune_Dementia_CR_hes_static.ckpt` | HES static features | Best epoch 16 |
| `crPreTrain_HES_1337.ckpt` | HES backbone pretrained | 8 epochs, 12.2M params |
| `crPreTrain_small_1337_FineTune_Dementia_CR_dual.ckpt` | **Dual-backbone (BEST)** | Best epoch 13, val_loss=0.007 |
| `crPreTrain_small_1337_FineTune_Dementia_CR_dual-v1.ckpt` | Dual-backbone (original save) | Same as above, original filename |
| `crPreTrain_small_1337_FineTune_Dementia_CR_idx{60,70,74,75}.ckpt` | Index age experiments | Index age ablation |
| `crPreTrain_small_1337_FineTune_Dementia_CR_idx68_cv_fold{0-4}.ckpt` | 5-fold CV | Cross-validation |
| `crPreTrain_small_1337_FineTune_Dementia_CR_Combined.ckpt` | Combined approach | Historic |

### 10.2 Key Pipeline Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `CPRD/run_hes_static_pipeline.sh` | Full hes_static pipeline | `bash run_hes_static_pipeline.sh` |
| `CPRD/run_dual_pipeline.sh` | **Dual-backbone pipeline (BEST)** | `bash run_dual_pipeline.sh` |
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

# Step 1: Extract HES features
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

## Appendix B: HES Feature Statistics

From `build_hes_summary_features.py` output (449,095 patients with HES records):
- Patients without any HES records get all-zero feature vectors
- Binary features (HES_HAS_*): 0 or 1
- Continuous features (HES_TOTAL_*): Normalized to [0, 1] via log transform

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
