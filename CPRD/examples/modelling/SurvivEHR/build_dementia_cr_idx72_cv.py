"""
build_dementia_cr_idx68_cv.py
==============================
Build a competing-risk fine-tuning dataset for dementia prediction.
Index age = 68, with purified dementia labels (31 codes).
Accepts a --fold argument for 5-fold cross-validation.

Usage:
    python build_dementia_cr_idx68_cv.py --fold 0
    python build_dementia_cr_idx68_cv.py --fold 1
    ...
"""

import argparse
import torch
import logging
from pathlib import Path

from FastEHR.dataloader import FoundationalDataModule
from FastEHR.dataloader.utils.study_criteria import index_inclusion_method

PATH_TO_DB = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
OUTPUT_DS_BASE = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_idx72_cv"
SPLITS_BASE = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/cv5_splits"
PRETRAIN_META_INFO = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/meta_information_custom.pickle"

INDEX_ON_AGE = 72

# 31 pure dementia codes (10 non-dementia codes removed)
DEMENTIA_READ_CODES = [
    "F110.", "Eu00.", "Eu01.", "Eu02z", "Eu002", "E00..",
    "Eu023", "Eu00z", "Eu025", "Eu01z", "E001.", "F1100",
    "Eu001", "E004.", "Eu000", "Eu02.", "Eu013", "E000.",
    "Eu01y", "E001z", "F1101", "Eu020", "E004z", "E0021",
    "Eu02y", "Eu012", "Eu011", "E00z.", "E0040", "E003.",
    "E0020",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    fold = args.fold
    output_path = f"{OUTPUT_DS_BASE}/fold{fold}/"
    splits_path = f"{SPLITS_BASE}/practice_id_splits_fold{fold}.pickle"

    torch.manual_seed(SEED)
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print(f"  Building Dementia CR Dataset (INDEX_AGE={INDEX_ON_AGE}, 31 pure codes)")
    print(f"  Fold: {fold}")
    print("=" * 70)
    print(f"  Database:       {PATH_TO_DB}")
    print(f"  Output:         {output_path}")
    print(f"  Splits file:    {splits_path}")
    print(f"  Dementia codes: {len(DEMENTIA_READ_CODES)}")
    print(f"  Death codes:    {DEATH_CODES}")
    print(f"  Total outcomes: {len(ALL_OUTCOME_CODES)} codes")
    print(f"  Study period:   {STUDY_PERIOD}")
    print()

    for split in ["train", "val", "test"]:
        (Path(output_path) / f"split={split}").mkdir(parents=True, exist_ok=True)

    dm = FoundationalDataModule(
        path_to_db=PATH_TO_DB,
        path_to_ds=output_path,
        load=False,
        tokenizer="tabular",
        overwrite_practice_ids=splits_path,
        overwrite_meta_information=PRETRAIN_META_INFO,
        study_inclusion_method=dementia_cr_inclusion_method(),
        min_workers=NUM_THREADS,
        seed=SEED,
    )

    print(f"\nFold {fold} dataset built successfully!")
    print(f"  Training patients:   {len(dm.train_set)}")
    print(f"  Validation patients: {len(dm.val_set)}")
    print(f"  Test patients:       {len(dm.test_set)}")

    print(f"\nDone fold {fold}.")


if __name__ == "__main__":
    main()
