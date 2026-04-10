"""
build_dementia_cr_idx60_dataset.py
===================================
Build a competing-risk fine-tuning dataset for dementia prediction.
Same as build_dementia_cr_dataset.py but with INDEX_ON_AGE = 60.

Cohort definition:
  - Index: Age 60
  - Outcomes: Dementia Read codes (k=1) + DEATH (k=2)
  - Censoring: Patients without dementia or death → k=0
  - The target event is whichever comes FIRST (dementia or death)
  - Study period: 1998-01-01 to 2019-12-31

Usage:
    python build_dementia_cr_idx60_dataset.py
"""

import torch
import logging
from pathlib import Path

from FastEHR.dataloader import FoundationalDataModule
from FastEHR.dataloader.utils.study_criteria import index_inclusion_method

PATH_TO_DB = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
OUTPUT_DS_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_idx60/"
PRETRAIN_PRACTICE_IDS = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/practice_id_splits.pickle"
PRETRAIN_META_INFO = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/meta_information_custom.pickle"

INDEX_ON_AGE = 60

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
DEATH_CODES = ["DEATH"]

ALL_OUTCOME_CODES = DEMENTIA_READ_CODES + DEATH_CODES

STUDY_PERIOD = ["1998-01-01", "2019-12-31"]
AGE_AT_ENTRY_RANGE = [50, 90]
MIN_REGISTERED_YEARS = 1
MIN_EVENTS = 5
NUM_THREADS = 12
SEED = 1337


def dementia_cr_inclusion_method():
    inclusion = index_inclusion_method(
        index_on=INDEX_ON_AGE,
        outcomes=ALL_OUTCOME_CODES,
        require_outcome=False,
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
    print("  Building Dementia CR Dataset (INDEX_AGE=60)")
    print("=" * 70)
    print(f"  Database:       {PATH_TO_DB}")
    print(f"  Output:         {OUTPUT_DS_PATH}")
    print(f"  Index on age:   {INDEX_ON_AGE}")
    print(f"  Dementia codes: {len(DEMENTIA_READ_CODES)}")
    print(f"  Death codes:    {DEATH_CODES}")
    print(f"  Total outcomes: {len(ALL_OUTCOME_CODES)} codes")
    print(f"  Study period:   {STUDY_PERIOD}")
    print()

    Path(OUTPUT_DS_PATH).mkdir(parents=True, exist_ok=True)

    dm = FoundationalDataModule(
        path_to_db=PATH_TO_DB,
        path_to_ds=OUTPUT_DS_PATH,
        load=False,
        tokenizer="tabular",
        overwrite_practice_ids=PRETRAIN_PRACTICE_IDS,
        overwrite_meta_information=PRETRAIN_META_INFO,
        study_inclusion_method=dementia_cr_inclusion_method(),
        min_workers=NUM_THREADS,
        seed=SEED,
    )

    print(f"\nDataset built successfully!")
    print(f"  Training patients:   {len(dm.train_set)}")
    print(f"  Validation patients: {len(dm.val_set)}")
    print(f"  Test patients:       {len(dm.test_set)}")
    print(f"  Vocab size:          {dm.train_set.tokenizer.vocab_size}")

    death_tid = dm.encode(["DEATH"])[0]
    dementia_tids = set()
    for code in DEMENTIA_READ_CODES:
        try:
            tid = dm.encode([code])[0]
            if tid > 1:
                dementia_tids.add(tid)
        except Exception:
            pass

    print(f"\n  DEATH token ID: {death_tid}")
    print(f"  Dementia token IDs: {len(dementia_tids)} unique")

    print(f"\nDone. Use config_FineTune_Dementia_CR_idx60.yaml to run fine-tuning.")
