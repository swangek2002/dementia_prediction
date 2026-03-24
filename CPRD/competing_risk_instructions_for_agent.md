# Implementing Competing-Risk Fine-Tuning for Dementia
# ====================================================
# Instructions for the fine-tuning agent.
# The single-risk run can continue in parallel — these instructions
# are for a SECOND, competing-risk run to be done alongside or after.

## ============================
## 1. WHY COMPETING-RISK
## ============================

In single-risk mode, death before dementia is treated as ordinary
censoring (k=0). This means the model treats "died at 68" the same
as "moved away at 68" — it assumes the patient COULD still develop
dementia if observed longer. This systematically overestimates
dementia risk, especially in elderly populations where mortality is
high.

Competing-risk mode explicitly models death as a separate event:
  k=0: censored (alive, no dementia in observation window)
  k=1: dementia onset
  k=2: death before dementia

The model then outputs TWO cumulative incidence functions:
  F_dementia(t) and F_death(t), constrained so that
  F_dementia(t) + F_death(t) <= 1

The original SurvivEHR paper uses competing-risk for the CVD task
(Table 2, Figure 6), where IHD and stroke are competing events.


## ============================
## 2. WHAT NEEDS TO CHANGE
## ============================

Three things need to change:
  A. Rebuild the dataset with death as a second outcome
  B. Change experiment.type in config
  C. Verify the code path handles it correctly


## ============================
## A. REBUILD DATASET
## ============================

### A.1 Find death codes in the database

First, find how death is coded in the UK Biobank primary care data.
In the original paper's CPRD vocabulary, death is a single event
category called "DEATH" (token 106, see Supplementary Table S1).

In the user's 108K UK Biobank vocabulary, death might be coded
differently. Run a query against the SQLite database to find death
codes:

```python
import sqlite3
import polars as pl

db_path = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
conn = sqlite3.connect(db_path)

# Search for death-related events
# Try exact match first
query1 = "SELECT DISTINCT EVENT FROM ... WHERE EVENT LIKE '%DEATH%' OR EVENT LIKE '%death%'"

# Also check the tokenizer vocabulary
# The tokenizer maps EVENT strings to token IDs
# Look for death in the pretrain tokenizer's vocabulary
```

Alternatively, check the pretrain tokenizer directly:

```python
from FastEHR.dataloader.foundational_loader import FoundationalDataModule

dm = FoundationalDataModule(
    path_to_db="...",
    path_to_ds="/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/",
    load=True, ...
)

# Search the tokenizer for death-related tokens
vocab = dm.tokenizer._event_counts
death_rows = vocab.filter(pl.col("EVENT").str.contains("(?i)death|died|deceased|mortality"))
print(death_rows)

# Or check if "DEATH" exists directly
death_token = dm.encode(["DEATH"])
print(f"DEATH token ID: {death_token}")
```

### A.2 Modify build_dementia_finetune_dataset.py

The build script's `fine_tune_outcomes` parameter needs to include
BOTH dementia codes AND death codes. The data pipeline needs to know
which outcome group each code belongs to.

Check how the original SurvivEHR code handles multiple outcome groups.
In the paper's CVD task, the config likely has something like:

```yaml
fine_tuning:
  fine_tune_outcomes:
    - [list of IHD codes]      # → k=1
    - [list of stroke codes]   # → k=2
```

Or it might use a different structure. Look at:
1. `setup_finetune_experiment.py` — how it reads `fine_tune_outcomes`
2. `FoundationalDataset` in supervised mode — how it assigns k values
3. Any existing fine-tune config examples in the codebase (e.g., for
   CVD or hypertension) that show the correct format

The key question is: does `fine_tune_outcomes` accept a LIST OF LISTS
(each sub-list = one competing event group), or a FLAT LIST (all
codes = one event)?

For SINGLE-RISK, it's a flat list:
```yaml
fine_tune_outcomes: [code1, code2, ...]  # all → k=1
```

For COMPETING-RISK, it should be something like:
```yaml
fine_tune_outcomes:
  - [dementia_code1, dementia_code2, ...]  # → k=1
  - [death_code1, death_code2, ...]         # → k=2
```

Search the codebase for how competing-risk outcomes are specified:

```bash
grep -r "fine_tune_outcomes" --include="*.py" --include="*.yaml" -n
grep -r "target_indicies" --include="*.py" -n
```

Look at `single_risk.py` line where `self.target_indicies` is set —
this tells you how the outcome codes are mapped to k values.

Then look at the competing-risk equivalent in `competing_risk.py` —
in competing-risk mode, the DeSurv head has `num_risks` outputs, and
each patient's k value indicates WHICH competing event occurred.

### A.3 Rebuild the dataset

Once you know the correct format, update the build script and rebuild:

```bash
# Add death codes to the outcome definition
# Rebuild with both dementia and death as outcomes
python build_dementia_finetune_dataset.py
```

The output dataset should have:
- Same patients as before (same index dates)
- But labels now include: k=0 (censored), k=1 (dementia), k=2 (death)
- Time-to-event: time from index date to whichever event came FIRST

IMPORTANT: A patient who gets dementia at 70 and dies at 72 should
have k=1, t=70-index. A patient who dies at 68 without dementia
should have k=2, t=68-index. A patient alive at study end without
either event should have k=0, t=study_end-index.


## ============================
## B. CONFIG CHANGES
## ============================

```yaml
experiment:
  type: 'fine-tune'          # NOT 'fine-tune-sr'
  # This triggers competing-risk in run_experiment.py:
  #   case "finetune" | "finetunecr":
  #     risk_model = "competing-risk"

fine_tuning:
  fine_tune_outcomes:
    # Group 1: Dementia codes → k=1
    - [F110., Eu00., Eu01., ...]  # all 41 dementia Read codes
    # Group 2: Death codes → k=2
    - [DEATH, ...]                # death code(s)
```

The `num_risks` for the new DeSurv head should be 2 (dementia + death),
NOT 41+1. All 41 dementia codes should map to the SAME event type (k=1).

### B.1 Verify run_experiment.py routing

In run_experiment.py, the case matching is:

```python
case "finetune" | "finetunecr" | "finetunesr":
    if experiment_type[-2:] == "sr":
        risk_model = "single-risk"
    else:
        risk_model = "competing-risk"
```

So `experiment.type = "fine-tune"` → `experiment_type = "finetune"` →
competing-risk. Good.

### B.2 Verify setup_finetune_experiment.py

Check that `setup_finetune_experiment` correctly:
1. Creates an `ODESurvCompetingRiskLayer` (not SingleRisk) when
   `risk_model="competing-risk"`
2. Sets `num_risks` to the number of competing event GROUPS (2),
   not the number of individual codes (41+death)
3. Passes sample_weights correctly to the competing-risk layer


## ============================
## C. VERIFICATION CHECKLIST
## ============================

Before running, verify:

[ ] Death codes found in the database/vocabulary
[ ] Dataset rebuilt with both dementia (k=1) and death (k=2) labels
[ ] Config uses experiment.type: 'fine-tune' (not fine-tune-sr)
[ ] fine_tune_outcomes has two groups: [dementia_codes] and [death_codes]
[ ] New DeSurv head has num_risks=2
[ ] Training batches show k=0, k=1, AND k=2 labels
[ ] val_loss is non-zero and decreasing (not the trivial k=0 problem)
[ ] PerformanceMetrics can compute C_td for competing-risk output

### Expected outcome

After training, the model should output two CIF curves per patient:
  F_dementia(t): cumulative probability of dementia by time t
  F_death(t): cumulative probability of death (without dementia) by t

The test metrics (C_td, IBS, INBLL) should be computed for the
dementia event specifically, comparable to the paper's Table 2.


## ============================
## D. RUN PLAN
## ============================

1. Keep the current single-risk training running (don't stop it)
2. In parallel, search for death codes in the database
3. Modify build script to include death as second outcome
4. Rebuild dataset → new directory: FineTune_Dementia_CR/
5. Create config_FineTune_Dementia_CR.yaml (copy from current, change
   experiment.type and fine_tune_outcomes)
6. After single-risk finishes AND new dataset is ready, run competing-risk
7. Compare SR vs CR results

This way you get results from BOTH approaches for the thesis.


## ============================
## E. MATCHING PAPER TABLE 2
## ============================

The paper reports results for 5 random seeds with 95% CI.

For the final thesis, you should run:

| Model               | Description                              |
|---------------------|------------------------------------------|
| SurvivEHR-FFT (CR)  | Pretrained + competing-risk fine-tune     |
| SurvivEHR-FFT (SR)  | Pretrained + single-risk fine-tune        |
| SurvivEHR-SFT (CR)  | From scratch + competing-risk fine-tune   |
| RSF                  | Random Survival Forest baseline           |
| DeepHit              | Deep learning baseline                    |
| DeSurv               | DeSurv baseline (cross-sectional input)   |

For each, report: C_td (↑), IBS (↓), INBLL (↓)

The comparison SurvivEHR-FFT vs SurvivEHR-SFT demonstrates the
value of pretraining. The comparison CR vs SR demonstrates the
value of competing-risk modeling for dementia.
