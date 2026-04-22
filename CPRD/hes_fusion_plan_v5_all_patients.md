# Final Plan: HES Full Fusion — All 8,900 Dementia Patients (v5)
# ================================================================
# Include ALL dementia patients by inserting translated HES events
# directly into a COPY of the SQLite database before dataset building.

## Core Strategy Change from v4

v4 tried to post-process parquet files after building. This cannot include
the ~4,400 patients who have NO GP records, because FoundationalDataModule
never creates parquet entries for them.

v5 solution: **Insert translated HES events into the database BEFORE building.**
FoundationalDataModule reads from diagnosis_table. If we insert translated
HES events (as Read v2 codes) into diagnosis_table, these patients will be
picked up automatically — including the 4,400 who previously had zero GP events.

We confirmed earlier that ALL HES patients exist in static_table (449,095 HES
patients, 100% overlap with static_table). So they have demographic info
(SEX, ETHNICITY, IMD, YEAR_OF_BIRTH). They only lack diagnosis_table events.
After insertion, they will have both.

## Implementation Steps

### Step 1: Build OMOP Mapping Dictionary

File: `build_omop_mapping.py`

Scan omop_condition_occurrence.csv (~34M rows). For each OMOP concept_id that
has BOTH ICD-10 and Read v2 source codes, create a mapping entry.

Resolve 1-to-N (one ICD-10 → multiple Read v2) by selecting the Read v2 code
with highest frequency in the pretrained diagnosis_table. This ensures the
model has the best embedding for the selected code.

Validation: We confirmed 24/24 critical disease codes are covered (dementia,
stroke, MI, heart failure, hypertension, diabetes, delirium, depression, TBI).

Output: `omop_icd10_to_readv2.pickle` — a dict {icd10_code: read_v2_code}

```python
# Pseudocode for 1-to-N resolution:
# 1. Get all Read v2 candidates for this ICD-10
# 2. Check which ones exist in diagnosis_table (model knows them)
# 3. Pick the one with highest COUNT(*) in diagnosis_table
# 4. If none exist in diagnosis_table, skip this ICD-10 code
```

Ambiguous code handling: Some codes like 'F001.' look like both ICD-10 and
Read v2. To disambiguate: if a code exists in diagnosis_table, treat it as
Read v2. If not, treat it as ICD-10.

### Step 2: Build HES Events per Patient

File: `build_hes_events.py`

Read hesin.csv (admission dates) and hesin_diag.csv (diagnoses).
For each patient, extract ALL hospital events that can be mapped:
- Include PRIMARY diagnoses (level=1) — always
- Include SECONDARY diagnoses (level=2) — only if mappable via OMOP dict
- Translate each ICD-10 code to Read v2 using the OMOP dictionary
- Record: (patient_id, practice_id, translated_read_v2_code, admission_date)

For PRACTICE_ID assignment:
- If patient exists in static_table with a PRACTICE_ID, use that
- Query: SELECT PRACTICE_ID FROM static_table WHERE PATIENT_ID = ?

Time filtering: Include ALL HES events (not just before index_date).
The index_date filtering will be handled by FoundationalDataModule's
index_inclusion_method — it automatically keeps only events before index
and the first outcome after index.

Output: `hes_events_for_db.pickle` — list of tuples:
  [(practice_id, patient_id, read_v2_code, date_string), ...]

### Step 3: Create Database Copy and Insert HES Events

File: `prepare_hes_fusion_db.py`

1. Copy the original database to a new file:
   ```python
   import shutil
   SRC_DB = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
   DST_DB = "/Data0/swangek_data/991/CPRD/data/example_exercise_database_hes_fusion.db"
   shutil.copy2(SRC_DB, DST_DB)
   ```

2. Open the copy and INSERT translated HES events into diagnosis_table:
   ```python
   conn = sqlite3.connect(DST_DB)
   cur = conn.cursor()

   # Load HES events
   hes_events = pickle.load(open('hes_events_for_db.pickle', 'rb'))

   # Batch insert
   cur.executemany(
       "INSERT INTO diagnosis_table (PRACTICE_ID, PATIENT_ID, EVENT, DATE) VALUES (?, ?, ?, ?)",
       hes_events
   )
   conn.commit()
   ```

3. Verify insertion:
   ```python
   cur.execute("SELECT COUNT(*) FROM diagnosis_table")
   print(f"Total events after insertion: {cur.fetchone()[0]}")
   # Should be ~132M (original) + HES translated events

   cur.execute("SELECT COUNT(DISTINCT PATIENT_ID) FROM diagnosis_table")
   print(f"Total patients after insertion: {cur.fetchone()[0]}")
   # Should be > 230,001 (original) because new HES-only patients added
   ```

CRITICAL: We NEVER modify the original database. All changes are on the copy.
The original database remains untouched for reproducibility.

### Step 4: Build Fine-Tuning Dataset

File: `build_dementia_cr_hes_fusion.py`

Uses FoundationalDataModule pointed at the COPIED database.
The dataset builder is almost identical to previous ones, with these changes:

```python
PATH_TO_DB = "/Data0/swangek_data/991/CPRD/data/example_exercise_database_hes_fusion.db"
OUTPUT_DS = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_fusion/"
INDEX_ON_AGE = 72
STUDY_PERIOD = ["1998-01-01", "2022-10-31"]
# Same 31 dementia Read codes + DEATH as outcomes
# Same practice_id_splits.pickle for train/val/test
# Same meta_information_custom.pickle (from PreTrain, not regenerated)
```

IMPORTANT: Use `overwrite_meta_information` pointing to the ORIGINAL pretrain
meta_information. Do NOT regenerate meta_information from the fused database,
because:
- The pretrained model's vocabulary and static encoders are based on the
  original meta_information
- If we regenerate, the vocabulary mapping and one-hot encoders will change,
  breaking compatibility with the pretrained checkpoint

What FoundationalDataModule will now do automatically:
- For patients with GP + HES events: their sequence will contain both GP
  Read codes and translated HES Read codes, sorted by date (the framework
  sorts events by date internally)
- For patients with HES-only events: their sequence will contain only
  translated HES events
- index_inclusion_method will handle all filtering (index date, study period,
  outcomes, min_events) as usual
- The LAST event in each patient's sequence will be determined by
  _reduce_on_outcome: either a dementia code, DEATH, or the last seen event

NOTE ON DEMENTIA LABELS: Because we inserted HES dementia events (translated
to Read v2 like Eu00., Eu01., Eu02z) directly into diagnosis_table, the
standard _reduce_on_outcome logic will AUTOMATICALLY find them as outcomes.
We do NOT need the separate HES post-processing relabeling step from v3/v4.
The dementia codes are now native Read v2 codes in the database.

### Step 5: Practice ID Assignment for HES-Only Patients

There is a subtlety: FoundationalDataModule splits data by PRACTICE_ID.
The ~4,400 HES-only patients have a PRACTICE_ID in static_table (from their
UKB registration). When we insert their HES events, we must use the same
PRACTICE_ID so they end up in the correct train/val/test split.

The practice_id_splits.pickle assigns practices to splits. A HES-only patient
assigned to a train practice goes to train, val practice goes to val, etc.

If a patient's PRACTICE_ID is not in ANY of the three split lists, they will
be excluded. Check this:
```python
# In prepare_hes_fusion_db.py, verify HES-only patients' practice IDs
# are covered by the splits
splits = pickle.load(open(SPLITS_PATH, 'rb'))
all_split_practices = set(splits['train'] + splits['val'] + splits['test'])

hes_only_practices = set()
for pid in hes_only_patient_ids:
    cur.execute("SELECT PRACTICE_ID FROM static_table WHERE PATIENT_ID=?", (pid,))
    row = cur.fetchone()
    if row:
        hes_only_practices.add(row[0])

uncovered = hes_only_practices - all_split_practices
print(f"HES-only patients in practices not in splits: {len(uncovered)}")
# If non-zero, those patients will be excluded regardless
```

### Step 6: YAML Configuration

Train: `confs/config_FineTune_Dementia_CR_hes_fusion.yaml`
Eval: `confs/config_FineTune_Dementia_CR_hes_fusion_eval.yaml`

Key differences from previous configs:

```yaml
data:
  path_to_db: /Data0/swangek_data/991/CPRD/data/example_exercise_database_hes_fusion.db
  path_to_ds: /Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_fusion/
  meta_information_path: /Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/meta_information_custom.pickle
  supervised_time_scale: 5.0
  num_static_covariates: 27

experiment:
  fine_tune_id: FineTune_Dementia_CR_hes_fusion
  notes: "HES full fusion via OMOP mapping, index=72, no SAW, all 8900 dementia patients"

fine_tuning:
  sample_weighting:
    mode: null  # NO SAW

optim:
  num_epochs: 15
  early_stop: True
  early_stop_patience: 10
```

CRITICAL: `path_to_db` points to the FUSED database copy, NOT the original.
But `meta_information_path` still points to the original pretrain meta_information.

### Step 7: Verification Script

File: `verify_hes_fusion_dataset.py`

Must check:
```
1. Total patients per split and k=1/k=2/k=0 counts
2. Average sequence length per split (should be longer than GP-only)
3. How many patients have BOTH GP and HES events in their sequence
4. How many patients have HES-only events (no original GP events)
5. Compare k=1 counts with previous experiments:
   - GP-only baseline: ~60 test k=1
   - HES label augmented: ~301 test k=1
   - HES fusion (this): should be >= 301, possibly higher
6. Verify no ICD-10 codes leaked into EVENT column (all should be Read v2)
```

### Step 8: Pipeline Script

File: `/Data0/swangek_data/991/CPRD/run_hes_fusion_pipeline.sh`

```bash
#!/bin/bash
set -e

LOG_DIR="/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
CKPT_DIR="/Data0/swangek_data/991/CPRD/output/checkpoints"
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"

LOG_FILE="${LOG_DIR}/finetune_cr_hes_fusion_log.txt"
cd "$WORK_DIR"

echo "============================================================" | tee "$LOG_FILE"
echo "  HES Full Fusion Fine-Tuning" | tee -a "$LOG_FILE"
echo "  idx72, no SAW, OMOP mapping, all dementia patients" | tee -a "$LOG_FILE"
echo "  Start: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Step 1: Build OMOP mapping dictionary
echo "===== Step 1: Build OMOP mapping =====" | tee -a "$LOG_FILE"
$PYTHON build_omop_mapping.py 2>&1 | tee -a "$LOG_FILE"

# Step 2: Build HES events lookup
echo "===== Step 2: Build HES events =====" | tee -a "$LOG_FILE"
$PYTHON build_hes_events.py 2>&1 | tee -a "$LOG_FILE"

# Step 3: Create fused database copy
echo "===== Step 3: Prepare fused database =====" | tee -a "$LOG_FILE"
$PYTHON prepare_hes_fusion_db.py 2>&1 | tee -a "$LOG_FILE"

# Step 4: Build fine-tuning dataset
echo "===== Step 4: Build dataset =====" | tee -a "$LOG_FILE"
$PYTHON build_dementia_cr_hes_fusion.py 2>&1 | tee -a "$LOG_FILE"

# Step 5: Verify dataset
echo "===== Step 5: Verify dataset =====" | tee -a "$LOG_FILE"
$PYTHON verify_hes_fusion_dataset.py 2>&1 | tee -a "$LOG_FILE"

# Step 6: Train (4-GPU DDP, 15 epochs max)
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "===== Step 6: Train =====" | tee -a "$LOG_FILE"
echo "Start train: $(date)" | tee -a "$LOG_FILE"
rm -f "${CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_hes_fusion.ckpt" 2>/dev/null || true
$PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_hes_fusion 2>&1 | tee -a "$LOG_FILE"
echo "End train: $(date)" | tee -a "$LOG_FILE"

# Step 7: Single-GPU eval
echo "===== Step 7: Eval =====" | tee -a "$LOG_FILE"
echo "Start eval: $(date)" | tee -a "$LOG_FILE"
CUDA_VISIBLE_DEVICES=0 $PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_hes_fusion_eval 2>&1 | tee -a "$LOG_FILE"
echo "End eval: $(date)" | tee -a "$LOG_FILE"
echo "===== ALL DONE =====" | tee -a "$LOG_FILE"
```

Launch:
```bash
chmod +x /Data0/swangek_data/991/CPRD/run_hes_fusion_pipeline.sh
tmux new-session -d -s hes_fusion "bash /Data0/swangek_data/991/CPRD/run_hes_fusion_pipeline.sh"
```

Monitor:
```bash
tail -f /Data0/swangek_data/991/CPRD/finetune_cr_hes_fusion_log.txt
```

## Critical Notes

1. NEVER modify the original database. Work on a copy only.
2. Use the ORIGINAL pretrain meta_information_custom.pickle. Do NOT regenerate.
3. The OMOP mapping must only output Read v2 codes that exist in the pretrained
   model's vocabulary (diagnosis_table of ORIGINAL database).
4. path_to_db in YAML must point to the FUSED database copy.
5. Pre-create split=train/val/test directories before FoundationalDataModule.
6. Do NOT include Chinese comments in code.
7. HES events inserted into diagnosis_table must use the patient's PRACTICE_ID
   from static_table so they end up in the correct train/val/test split.
8. The approach makes HES dementia relabeling UNNECESSARY — because HES
   dementia codes (translated to Read v2) are now native events in the
   database. _reduce_on_outcome will find them automatically.
9. Check that inserted HES events don't create duplicate events (same patient,
   same date, same code). Deduplicate if needed.
10. The fused database will be larger than the original (~132M + HES events).
    Ensure sufficient disk space.

## Expected Results Comparison

| Experiment | Test k=1 | Input | C_td |
|------------|----------|-------|------|
| GP-only baseline (idx65) | ~60 | GP only | 0.708 |
| HES label aug (idx72) | ~301 | GP + HES label only | 0.733 |
| **HES fusion (this)** | **~500+** | **GP + HES sequences** | **???** |

Test k=1 should increase beyond 301 because:
- The ~4,400 HES-only patients now have sequences (translated HES events)
  and can enter the dataset
- Some of these patients will land in the test split
- The model sees richer input sequences for ALL patients (GP + HES combined)
