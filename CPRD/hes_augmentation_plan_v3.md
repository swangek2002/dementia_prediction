# Final Plan: HES-Augmented Dementia Fine-Tuning (v3)
# ====================================================
# Parameters: index_age=72, NO SAW, NO cross-validation, single fixed split
# Uses original practice_id_splits.pickle for train/val/test

## Source Code Findings

The label mechanism works as follows:
1. `_reduce_on_outcome()` filters diagnosis_table for events matching the
   `outcomes` list (31 Read codes + DEATH)
2. It takes the EARLIEST matching event AFTER the index date within study period
3. If no outcome event exists, it takes the LAST event in study period (censored, k=0)
4. Final patient sequence = all events before index + one final event (outcome or censored)

The parquet files store each patient as a row with columns including EVENT (list)
and DATE (list). The LAST element of EVENT determines the label:
- If last EVENT is a dementia Read code -> k=1
- If last EVENT is DEATH -> k=2
- Otherwise -> k=0 (censored)

HES data is in CSV files, NOT in the SQLite database. The inclusion method
only queries the database. Therefore we CANNOT inject HES labels through
the normal pipeline. We must use post-processing.

## Implementation: Post-Process Parquet Files

### Step 0: Read source code (REQUIRED before coding)

The agent MUST read these files to understand the parquet structure:

1. `index_inclusion_method` source — already provided, understood.

2. The `__save_data_to_parquet` method in dataset_polars.py — to understand
   the exact column names and types in the parquet files. Find it in:
   `/Data0/swangek_data/991/FastEHR/FastEHR/dataloader/dataset/dataset_polars.py`

3. One sample parquet file to verify the structure:
   ```python
   import pyarrow.parquet as pq
   import os
   # Use any existing dataset
   test_dir = '/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_idx72_cv/fold0/split=test/'
   for root, dirs, files in os.walk(test_dir):
       for fn in files:
           if fn.endswith('.parquet'):
               df = pq.read_table(os.path.join(root, fn)).to_pandas()
               print(df.columns.tolist())
               print(df.dtypes)
               print(df.iloc[0])  # first patient
               break
       break
   ```

DO NOT write any modification code until you understand the exact parquet
schema (column names, types, how EVENT and DATE are stored as lists).

### Step 1: Create HES Dementia Lookup

File: `/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR/hes_dementia_lookup.py`

```python
"""
hes_dementia_lookup.py
======================
Build a lookup dict: {patient_id (int): earliest_hes_dementia_date (str)}
from the raw HES CSV files.

Only includes patients whose HES dementia diagnosis date is valid
(between 1990-01-01 and 2025-01-01).
"""
import csv

HESIN_PATH = "/Data0/swangek_data/991/CPRD/data/hesin.csv"
HESIN_DIAG_PATH = "/Data0/swangek_data/991/CPRD/data/hesin_diag.csv"

HES_DEMENTIA_ICD10_PREFIXES = ['F00', 'F01', 'F02', 'G30']
HES_DEMENTIA_ICD10_EXACT = ['F03']


def is_dementia_icd10(code):
    """Check if an ICD-10 code is a dementia diagnosis."""
    if not code:
        return False
    for prefix in HES_DEMENTIA_ICD10_PREFIXES:
        if code.startswith(prefix):
            return True
    if code in HES_DEMENTIA_ICD10_EXACT:
        return True
    return False


def build_hes_dementia_lookup():
    """
    Returns dict: {patient_id (int): earliest_dementia_date (str 'YYYY-MM-DD')}
    """
    # Step 1: Load hesin.csv to get admission dates
    hesin_dates = {}
    with open(HESIN_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            hesin_dates[row['dnx_hesin_id']] = row.get('admidate', '').strip()

    # Step 2: Scan hesin_diag.csv for dementia ICD-10 codes
    patient_earliest = {}
    with open(HESIN_DIAG_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            icd = row.get('diag_icd10', '').strip()
            if not is_dementia_icd10(icd):
                continue
            date = hesin_dates.get(row['dnx_hesin_id'], '')
            if not date or date < '1990-01-01' or date > '2025-01-01':
                continue
            pid = int(row['eid'])
            if pid not in patient_earliest or date < patient_earliest[pid]:
                patient_earliest[pid] = date

    print(f"HES dementia lookup: {len(patient_earliest)} patients")
    return patient_earliest


if __name__ == '__main__':
    lookup = build_hes_dementia_lookup()
    dates = sorted(lookup.values())
    print(f"  Earliest: {dates[0]}")
    print(f"  Latest: {dates[-1]}")
```

Test standalone first:
```bash
cd /Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR/
/Data0/swangek_data/conda_envs/survivehr/bin/python hes_dementia_lookup.py
```

### Step 2: Create Dataset Builder with HES Post-Processing

File: `/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR/build_dementia_cr_hes_aug.py`

This script does TWO things:
1. Build a normal GP-only dataset using FoundationalDataModule
2. Post-process the parquet files to relabel censored patients who have HES dementia

Parameters:
- INDEX_ON_AGE = 72
- STUDY_PERIOD = ['1998-01-01', '2022-10-31']  (extended to match HES coverage)
- Use original fixed split: practice_id_splits.pickle
  `/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/practice_id_splits.pickle`
- Output: `/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_aug/`
- 31 pure dementia Read codes (same as all previous experiments)
- NO fold argument (single fixed split)

The post-processing logic for each parquet file:

```
For each patient row in parquet:
    1. Read EVENT list and DATE list
    2. Get last EVENT
    3. If last EVENT is a dementia Read code (k=1) or DEATH (k=2):
       -> Skip, already labeled correctly
    4. If patient_id NOT in HES dementia lookup:
       -> Skip, truly censored
    5. Get HES dementia date
    6. Compute patient's index date = YEAR_OF_BIRTH + 72 years
       (read YEAR_OF_BIRTH from static_table in the database)
    7. If HES dementia date <= index date:
       -> Skip, dementia occurred before observation window
    8. If HES dementia date > '2022-10-31':
       -> Skip, outside study period
    9. REPLACE last EVENT with 'Eu02z' (unspecified dementia Read code)
    10. REPLACE last DATE with HES dementia date
        (convert to same format/type as existing dates in parquet)
    11. Write back modified parquet file
```

WHY 'Eu02z': All 31 dementia Read codes map to the same risk group (risk 0)
in the competing risk model. The specific code does not matter for the loss
calculation. 'Eu02z' = unspecified dementia, safe choice. If the agent prefers,
it can map ICD-10 subtypes to corresponding Read codes:
  F00* -> 'Eu00.' (Alzheimer's)
  F01* -> 'Eu01.' (Vascular)
  F02* -> 'Eu02.' (Other)
  F03/G30* -> 'Eu02z' (Unspecified)
But this is optional and does not affect results.

CRITICAL: The agent must first inspect the actual parquet schema (Step 0)
to know exact column names and date format. DO NOT GUESS.

NOTE on STUDY_PERIOD: We extend to '2022-10-31' so that:
- Patients censored at 2019 in the old setup now have longer observation
- HES dementia diagnoses from 2017-2022 become valid labels
- The model sees longer censoring times for k=0 patients (more accurate)
- Adjust supervised_time_scale in YAML to 5.0 (max follow-up now ~30 years)

### Step 3: Create YAML Configs

Two config files, using idx72 configs as template:

Train config:
`/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR/confs/config_FineTune_Dementia_CR_hes_aug.yaml`

Eval config:
`/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR/confs/config_FineTune_Dementia_CR_hes_aug_eval.yaml`

Key parameters (DIFFERENCES from previous configs highlighted):

```yaml
data:
  batch_size: 32
  unk_freq_threshold: 0.0
  min_workers: 12
  global_diagnoses: True
  repeating_events: False
  path_to_db: /Data0/swangek_data/991/CPRD/data/example_exercise_database.db
  path_to_ds: /Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_aug/
  meta_information_path: /Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/meta_information_custom.pickle
  subsample_training: null
  num_static_covariates: 27
  supervised_time_scale: 5.0  # CHANGED: was 3.0, extended due to longer observation window

experiment:
  type: 'fine-tune'
  project_name: SurvivEHR
  run_id: ${head.SurvLayer}PreTrain_small_${experiment.seed}
  fine_tune_id: FineTune_Dementia_CR_hes_aug  # CHANGED
  notes: "HES-augmented labels, index=72, NO SAW, study period to 2022-10-31"
  tags: ["dementia", "fine-tune", "competing-risk", "idx72", "hes-aug", "no-saw"]
  train: True   # For eval config: set to False
  test: True
  verbose: True
  seed: 1337
  log: True
  log_dir: /Data0/swangek_data/991/CPRD/output/
  ckpt_dir: /Data0/swangek_data/991/CPRD/output/checkpoints/

fine_tuning:
  fine_tune_outcomes:
    - - "F110."
      - "Eu00."
      - "Eu01."
      - "Eu02z"
      - "Eu002"
      - "E00.."
      - "Eu023"
      - "Eu00z"
      - "Eu025"
      - "Eu01z"
      - "E001."
      - "F1100"
      - "Eu001"
      - "E004."
      - "Eu000"
      - "Eu02."
      - "Eu013"
      - "E000."
      - "Eu01y"
      - "E001z"
      - "F1101"
      - "Eu020"
      - "E004z"
      - "E0021"
      - "Eu02y"
      - "Eu012"
      - "Eu011"
      - "E00z."
      - "E0040"
      - "E003."
      - "E0020"
    - - "DEATH"

  custom_outcome_method:
    _target_: null
  custom_stratification_method:
    _target_: null

  use_callbacks:
    hidden_embedding:
      num_batches: 0
      mask_static: False
      mask_value: False
    performance_metrics: True
    rmst: False

  compression_layer: False
  llrd: null

  PEFT:
    method: null
    adapter_dim: 8
  backbone:
    linear_probe_epochs: 0
    unfreeze_top_k: null
  head:
    surv_weight: 1
    value_weight: 0
    learning_rate: 5e-4

  sample_weighting:
    mode: null        # NO SAW — changed from "event_only"
    event_lambda: 1.0 # irrelevant when mode is null
    alpha: 2.0
    tau: 0.33
    w_t_max: 3.0
    w_total_max: 20.0

optim:
  num_epochs: 15
  learning_rate: 5e-5
  scheduler_warmup: False
  scheduler: reduceonplateau
  scheduler_periods: 10000
  learning_rate_decay: 0.8
  val_check_interval: 1.0
  early_stop: True
  early_stop_patience: 10
  log_every_n_steps: 20
  limit_val_batches: 1.0
  limit_test_batches: null
  accumulate_grad_batches: 4

transformer:
  block_type: "Neo"
  block_size: 512
  n_layer: 6
  n_head: 6
  n_embd: 384
  layer_norm_bias: False
  attention_type: "global"
  bias: True
  dropout: 0.1
  attention_dropout: 0.1
  resid_dropout: 0.1
  private_heads: 0

head:
  SurvLayer: "cr"
  surv_weight: 1
  tokens_for_univariate_regression: None
  value_weight: 0.1
```

For eval config: identical except `experiment.train: False`.

### Step 4: Create Pipeline Script

File: `/Data0/swangek_data/991/CPRD/run_hes_aug_pipeline.sh`

```bash
#!/bin/bash
set -e

LOG_DIR="/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
CKPT_DIR="/Data0/swangek_data/991/CPRD/output/checkpoints"
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"

LOG_FILE="${LOG_DIR}/finetune_cr_hes_aug_log.txt"
cd "$WORK_DIR"

echo "============================================================" | tee "$LOG_FILE"
echo "  HES-Augmented Fine-Tuning: idx72, no SAW, study to 2022" | tee -a "$LOG_FILE"
echo "  Start: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

########################################
# Step 1: Build dataset + HES post-processing
########################################
echo "" | tee -a "$LOG_FILE"
echo "===== Step 1: Build HES-augmented dataset =====" | tee -a "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"

$PYTHON build_dementia_cr_hes_aug.py 2>&1 | tee -a "$LOG_FILE"

echo "Dataset build done: $(date)" | tee -a "$LOG_FILE"

########################################
# Step 2: Verify dataset statistics
########################################
echo "" | tee -a "$LOG_FILE"
echo "===== Step 2: Verify dataset stats =====" | tee -a "$LOG_FILE"

$PYTHON verify_hes_aug_dataset.py 2>&1 | tee -a "$LOG_FILE"

echo "Verification done: $(date)" | tee -a "$LOG_FILE"

########################################
# Step 3: Train (4-GPU DDP, 15 epochs max)
########################################
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "" | tee -a "$LOG_FILE"
echo "===== Step 3: Train (4-GPU, 15 epochs max) =====" | tee -a "$LOG_FILE"
echo "Start train: $(date)" | tee -a "$LOG_FILE"

rm -f "${CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_hes_aug.ckpt" 2>/dev/null || true

$PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_hes_aug 2>&1 | tee -a "$LOG_FILE"

echo "End train: $(date)" | tee -a "$LOG_FILE"

########################################
# Step 4: Single-GPU eval (best val_loss checkpoint)
########################################
echo "" | tee -a "$LOG_FILE"
echo "===== Step 4: Single-GPU eval =====" | tee -a "$LOG_FILE"
echo "Start eval: $(date)" | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=0 $PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_hes_aug_eval 2>&1 | tee -a "$LOG_FILE"

echo "End eval: $(date)" | tee -a "$LOG_FILE"
echo "===== ALL DONE =====" | tee -a "$LOG_FILE"
```

### Step 5: Verification Script

File: `/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR/verify_hes_aug_dataset.py`

```python
"""
verify_hes_aug_dataset.py
=========================
Verify HES-augmented dataset statistics before training.
Compares against GP-only baseline.
"""
import os
import pyarrow.parquet as pq

DEMENTIA_CODES = set([
    'F110.','Eu00.','Eu01.','Eu02z','Eu002','E00..','Eu023','Eu00z','Eu025',
    'Eu01z','E001.','F1100','Eu001','E004.','Eu000','Eu02.','Eu013','E000.',
    'Eu01y','E001z','F1101','Eu020','E004z','E0021','Eu02y','Eu012','Eu011',
    'E00z.','E0040','E003.','E0020',
])
DEATH_CODES = set(['DEATH'])

def count_labels(dataset_path):
    total, k1, k2, k0 = 0, 0, 0, 0
    for root, dirs, files in os.walk(dataset_path):
        for fn in files:
            if not fn.endswith('.parquet'):
                continue
            df = pq.read_table(os.path.join(root, fn)).to_pandas()
            for _, row in df.iterrows():
                total += 1
                events = row['EVENT']
                if len(events) > 0:
                    last = events[-1]
                    if last in DEMENTIA_CODES:
                        k1 += 1
                    elif last in DEATH_CODES:
                        k2 += 1
                    else:
                        k0 += 1
    return total, k1, k2, k0

hes_base = '/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_aug/'

print("=" * 70)
print("  HES-Augmented Dataset Statistics (index_age=72, no SAW)")
print("=" * 70)
print(f"{'Split':<8} {'Total':>8} {'Dementia(k=1)':>14} {'Death(k=2)':>11} {'Censored(k=0)':>14} {'Event Rate':>11}")
print("-" * 70)

for split in ['train', 'val', 'test']:
    path = f'{hes_base}split={split}/'
    total, k1, k2, k0 = count_labels(path)
    rate = 100 * k1 / max(1, total)
    print(f"{split:<8} {total:>8} {k1:>14} {k2:>11} {k0:>14} {rate:>10.2f}%")
```

### Step 6: Launch and Monitor

```bash
# Launch in tmux
chmod +x /Data0/swangek_data/991/CPRD/run_hes_aug_pipeline.sh
tmux new-session -d -s hes_aug "bash /Data0/swangek_data/991/CPRD/run_hes_aug_pipeline.sh"

# Monitor
tail -f /Data0/swangek_data/991/CPRD/finetune_cr_hes_aug_log.txt

# Check best val_loss
grep "val_loss.*reached" /Data0/swangek_data/991/CPRD/finetune_cr_hes_aug_log.txt

# Check final eval results (take the SECOND occurrence for 1-GPU eval)
grep "^ .*Test:OutcomePerformanceMetrics_risk0_31codesctd" /Data0/swangek_data/991/CPRD/finetune_cr_hes_aug_log.txt
```

### Step 7: Expected Results Comparison

| Experiment | Index | SAW | Dementia k=1 (test) | Dementia C_td |
|------------|-------|-----|---------------------|---------------|
| GP-only baseline (no SAW, fixed split) | 65 | No | ~60 | 0.708 |
| GP-only 5-fold CV | 72 | λ=6 | ~154 mean/fold | 0.707 ± 0.019 |
| **HES-augmented (this experiment)** | 72 | No | ~??? (expect 3-4x of baseline) | ??? |

## Critical Notes for the Agent

1. DO NOT modify any pretrained model or pretrained checkpoint.
2. DO NOT add ICD-10 codes to the model's vocabulary or input sequences.
3. DO NOT modify any files in `/Data0/swangek_data/991/FastEHR/`.
4. The model input is ALWAYS GP Read code sequences. HES data is ONLY used
   to determine labels (who has dementia and when).
5. Read the actual parquet file schema BEFORE writing post-processing code.
6. If the file content is too large and gets truncated, tell me immediately.
7. Do NOT include Chinese comments in any code.
8. Pre-create split=train, split=val, split=test directories before running
   FoundationalDataModule to avoid FileNotFoundError.
9. Use the ORIGINAL fixed practice_id_splits.pickle (NOT the cv5 splits):
   `/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/practice_id_splits.pickle`
10. Sample weighting mode must be null (NO SAW).
11. Study period extended to '2022-10-31' to capture HES dementia labels.
12. supervised_time_scale = 5.0 (increased from 3.0 due to longer observation window).
