"""
build_dementia_finetune_dataset.py
===================================
Build an indexed fine-tuning dataset for dementia prediction.

Cohort definition (modifiable via the constants below):
  - Index: Age-based (default 65 years) — standard dementia screening threshold
  - Outcomes: All dementia Read codes from the UK Biobank primary care data
  - Censoring: Patients without dementia are right-censored at their last observation
  - Study period: 1998-01-01 to 2019-12-31
  - Age at entry: 50–90 years
  - Minimum registration: 1 year at practice

After running this script, set the config's data.path_to_ds to the output
directory and experiment.type to 'fine-tune'.

Usage:
    python build_dementia_finetune_dataset.py
"""

import torch
import logging
from pathlib import Path

from FastEHR.dataloader import FoundationalDataModule
from FastEHR.dataloader.utils.study_criteria import index_inclusion_method

# ======================== USER CONFIG ========================

# Path to the raw database
PATH_TO_DB = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"

# Where to write the built fine-tuning dataset
OUTPUT_DS_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia/"

# Reuse pre-training splits and tokenizer to avoid data leakage
PRETRAIN_PRACTICE_IDS = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/practice_id_splits.pickle"
PRETRAIN_META_INFO = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/meta_information_custom.pickle"

# ---- Cohort definition ----
# Index on age (years).  65 is the standard threshold for dementia screening.
# Change to an event name (str) if you want event-based indexing instead.
INDEX_ON_AGE = 65

# Dementia outcome tokens — these Read codes cover the main dementia diagnoses
# in UK Biobank primary care.  They will be looked up via dm.encode() at runtime.
DEMENTIA_READ_CODES = [
    "F110.", "Eu00.", "Eu01.", "Eu02z", "Eu002", "E00..",
    "Eu023", "Eu00z", "Eu025", "Eu01z", "E001.", "F1100",
    "Eu001", "E004.", "Eu000", "Eu02.", "Eu013", "E000.",
    "Eu01y", "E001z", "F1101", "Eu020", "E004z", "E0021",
    "Eu02y", "Eu012", "Eu011", "E00z.", "E0040", "E003.",
    "E0020",
    "Eu057", "Eu04.", "Eu053", "Eu04z", "Eu0z.", "Eu060",
    "Eu054", "Eu05y", "Eu052", "Eu062",
]

STUDY_PERIOD = ["1998-01-01", "2019-12-31"]
AGE_AT_ENTRY_RANGE = [50, 90]
MIN_REGISTERED_YEARS = 1
MIN_EVENTS = 5  # require at least 5 events before index for meaningful context

NUM_THREADS = 12
SEED = 1337
# =============================================================


def dementia_inclusion_method():
    """Create an index_inclusion_method for dementia fine-tuning."""
    inclusion = index_inclusion_method(
        index_on=INDEX_ON_AGE,
        outcomes=DEMENTIA_READ_CODES,
        require_outcome=False,  # include censored patients for survival analysis
        exclude_on_events=None,
        exclude_on_events_prior_to_index=None,
        study_period=STUDY_PERIOD,
        age_at_entry_range=AGE_AT_ENTRY_RANGE,
        min_registered_years=MIN_REGISTERED_YEARS,
        min_events=MIN_EVENTS,
    )
    return inclusion.fit


if __name__ == "__main__":
    torch.manual_seed(SEED)
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("  Building Dementia Fine-Tuning Dataset")
    print("=" * 70)
    print(f"  Database:       {PATH_TO_DB}")
    print(f"  Output:         {OUTPUT_DS_PATH}")
    print(f"  Index on age:   {INDEX_ON_AGE}")
    print(f"  Outcomes:       {len(DEMENTIA_READ_CODES)} dementia Read codes")
    print(f"  Study period:   {STUDY_PERIOD}")
    print(f"  Age range:      {AGE_AT_ENTRY_RANGE}")
    print(f"  Min events:     {MIN_EVENTS}")
    print()

    Path(OUTPUT_DS_PATH).mkdir(parents=True, exist_ok=True)

    dm = FoundationalDataModule(
        path_to_db=PATH_TO_DB,
        path_to_ds=OUTPUT_DS_PATH,
        load=False,  # build, not load
        tokenizer="tabular",
        overwrite_practice_ids=PRETRAIN_PRACTICE_IDS,
        overwrite_meta_information=PRETRAIN_META_INFO,
        study_inclusion_method=dementia_inclusion_method(),
        min_workers=NUM_THREADS,
        seed=SEED,
    )

    print(f"\nDataset built successfully!")
    print(f"  Training patients:   {len(dm.train_set)}")
    print(f"  Validation patients: {len(dm.val_set)}")
    print(f"  Test patients:       {len(dm.test_set)}")
    print(f"  Vocab size:          {dm.train_set.tokenizer.vocab_size}")

    # Show a sample batch
    for batch in dm.train_dataloader():
        print(f"\nSample batch keys: {list(batch.keys())}")
        for k, v in batch.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}")
        break

    # Report dementia token IDs
    dementia_tokens = []
    for code in DEMENTIA_READ_CODES:
        try:
            tid = dm.encode([code])[0]
            if tid > 1:
                dementia_tokens.append((code, tid))
        except Exception:
            pass

    print(f"\nDementia tokens found: {len(dementia_tokens)}")
    for code, tid in dementia_tokens[:10]:
        print(f"  {code} -> token {tid}")
    if len(dementia_tokens) > 10:
        print(f"  ... and {len(dementia_tokens) - 10} more")

    dementia_token_ids = [tid for _, tid in dementia_tokens]
    print(f"\nFor fine_tune_outcomes in config, use these Read codes:")
    print(f"  {DEMENTIA_READ_CODES[:5]} ...")
    print(f"\nOr set fine_tune_outcomes to the token IDs directly:")
    print(f"  {dementia_token_ids[:10]} ...")

    print(f"\nDone. You can now run fine-tuning with:")
    print(f"  data.path_to_ds={OUTPUT_DS_PATH}")
    print(f"  experiment.type=fine-tune")
