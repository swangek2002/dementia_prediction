"""
build_dementia_cr_hes_fusion.py
================================
Build the HES-fused dementia CR fine-tuning dataset by pointing
FoundationalDataModule at the FUSED database copy (which already contains
translated HES events inside diagnosis_table).

No post-processing of parquet files is needed: the dementia codes are
already native Read v2 events in the database, so _reduce_on_outcome
finds them automatically.
"""

import logging
from pathlib import Path

import torch

from FastEHR.dataloader import FoundationalDataModule
from FastEHR.dataloader.utils.study_criteria import index_inclusion_method

PATH_TO_DB = "/Data0/swangek_data/991/CPRD/data/example_exercise_database_hes_fusion.db"
OUTPUT_DS = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_fusion/"
SPLITS_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/practice_id_splits.pickle"
PRETRAIN_META_INFO = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/meta_information_custom.pickle"

INDEX_ON_AGE = 72

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

STUDY_PERIOD = ["1998-01-01", "2022-10-31"]
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
    torch.manual_seed(SEED)
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("  Building HES-Fused Dementia CR Dataset")
    print(f"  INDEX_AGE={INDEX_ON_AGE}, NO SAW, study period to 2022-10-31")
    print("=" * 70)
    print(f"  Database (FUSED): {PATH_TO_DB}")
    print(f"  Output:           {OUTPUT_DS}")
    print(f"  Splits file:      {SPLITS_PATH}")
    print(f"  Meta info:        {PRETRAIN_META_INFO}")
    print()

    for split in ["train", "val", "test"]:
        (Path(OUTPUT_DS) / f"split={split}").mkdir(parents=True, exist_ok=True)

    dm = FoundationalDataModule(
        path_to_db=PATH_TO_DB,
        path_to_ds=OUTPUT_DS,
        load=False,
        tokenizer="tabular",
        overwrite_practice_ids=SPLITS_PATH,
        overwrite_meta_information=PRETRAIN_META_INFO,
        study_inclusion_method=dementia_cr_inclusion_method(),
        min_workers=NUM_THREADS,
        seed=SEED,
    )

    print(f"\nFused dataset built:")
    print(f"  Training patients:   {len(dm.train_set)}")
    print(f"  Validation patients: {len(dm.val_set)}")
    print(f"  Test patients:       {len(dm.test_set)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
