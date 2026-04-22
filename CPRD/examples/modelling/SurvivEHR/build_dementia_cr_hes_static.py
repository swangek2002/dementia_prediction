"""
build_dementia_cr_hes_static.py
================================
Build a fine-tuning dataset that combines:
  1. Pure GP sequences (same as hes_aug - original DB, no HES events in sequences)
  2. HES label augmentation (relabel censored patients with HES dementia diagnoses)
  3. HES summary statistics as extra columns in parquet files (static covariates)

The model sees GP sequences unchanged (no modality clash / truncation),
but gets HES hospital summary features as static covariates via 8 extra columns
(HES_TOTAL_ADMISSIONS, HES_TOTAL_UNIQUE_DIAG, HES_HAS_STROKE, etc.).

Usage:
    python build_dementia_cr_hes_static.py
"""

import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import torch
import logging
from pathlib import Path

from FastEHR.dataloader import FoundationalDataModule
from FastEHR.dataloader.utils.study_criteria import index_inclusion_method
from hes_dementia_lookup import build_hes_dementia_lookup

# ---- Constants ----
PATH_TO_DB = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
OUTPUT_DS = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static/"
SPLITS_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/practice_id_splits.pickle"
PRETRAIN_META_INFO = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/meta_information_custom.pickle"
HES_FEATURES_PATH = "/Data0/swangek_data/991/CPRD/data/hes_summary_features.pickle"

INDEX_ON_AGE = 72

DEMENTIA_READ_CODES = [
    "F110.", "Eu00.", "Eu01.", "Eu02z", "Eu002", "E00..",
    "Eu023", "Eu00z", "Eu025", "Eu01z", "E001.", "F1100",
    "Eu001", "E004.", "Eu000", "Eu02.", "Eu013", "E000.",
    "Eu01y", "E001z", "F1101", "Eu020", "E004z", "E0021",
    "Eu02y", "Eu012", "Eu011", "E00z.", "E0040", "E003.",
    "E0020",
]
DEMENTIA_READ_CODES_SET = set(DEMENTIA_READ_CODES)
DEATH_CODES = ["DEATH"]
ALL_OUTCOME_CODES = DEMENTIA_READ_CODES + DEATH_CODES

# HES-augmented label code (unspecified dementia Read code)
HES_LABEL_CODE = "Eu02z"

STUDY_PERIOD = ["1998-01-01", "2022-10-31"]
STUDY_END_DATE = pd.Timestamp("2022-10-31")
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


def load_year_of_birth_lookup():
    conn = sqlite3.connect(PATH_TO_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT PATIENT_ID, YEAR_OF_BIRTH FROM static_table")
    rows = cursor.fetchall()
    conn.close()
    yob_lookup = {}
    for pid, yob_str in rows:
        yob_lookup[int(pid)] = pd.Timestamp(yob_str)
    print(f"Year-of-birth lookup: {len(yob_lookup)} patients")
    return yob_lookup


def post_process_parquet_files(hes_lookup, yob_lookup, hes_features, feature_names):
    """
    Walk all parquet files in the dataset directory and:
      1. Add HES summary feature columns to each patient row
      2. Relabel censored patients with HES dementia diagnoses (same as hes_aug)
    """
    total_patients = 0
    relabeled = 0
    skipped_already_labeled = 0
    skipped_no_hes = 0
    skipped_before_index = 0
    skipped_after_study = 0
    patients_with_hes_features = 0

    num_features = len(feature_names)

    for root, dirs, files in os.walk(OUTPUT_DS):
        for fn in files:
            if not fn.endswith('.parquet'):
                continue
            filepath = os.path.join(root, fn)
            table = pq.read_table(filepath)
            df = table.to_pandas()
            modified = False

            # --- Add HES feature columns ---
            hes_feat_arrays = []
            for idx, row in df.iterrows():
                pid = int(row['PATIENT_ID'])
                feat = hes_features.get(pid, np.zeros(num_features, dtype=np.float32))
                hes_feat_arrays.append(feat)
                if pid in hes_features:
                    patients_with_hes_features += 1

            hes_feat_matrix = np.stack(hes_feat_arrays)
            for i, col_name in enumerate(feature_names):
                df[col_name] = hes_feat_matrix[:, i].astype(np.float32)
            modified = True  # Always modified since we add columns

            # --- HES label augmentation (same as hes_aug) ---
            for idx, row in df.iterrows():
                total_patients += 1
                events = row['EVENT']
                dates = row['DATE']

                if len(events) == 0:
                    continue

                last_event = events[-1]

                if last_event in DEMENTIA_READ_CODES_SET or last_event == 'DEATH':
                    skipped_already_labeled += 1
                    continue

                pid = int(row['PATIENT_ID'])

                if pid not in hes_lookup:
                    skipped_no_hes += 1
                    continue

                hes_date = hes_lookup[pid]

                yob = yob_lookup.get(pid)
                if yob is None:
                    skipped_no_hes += 1
                    continue
                index_date = yob + pd.DateOffset(years=INDEX_ON_AGE)

                if hes_date <= index_date:
                    skipped_before_index += 1
                    continue

                if hes_date > STUDY_END_DATE:
                    skipped_after_study += 1
                    continue

                new_events = events.copy()
                new_dates = dates.copy()
                new_events[-1] = HES_LABEL_CODE
                new_dates[-1] = np.datetime64(hes_date, 'us')

                df.at[idx, 'EVENT'] = new_events
                df.at[idx, 'DATE'] = new_dates
                modified = True
                relabeled += 1

            if modified:
                new_table = pa.Table.from_pandas(df, preserve_index=False)
                pq.write_table(new_table, filepath)

    print(f"\n{'='*60}")
    print(f"  Post-Processing Summary")
    print(f"{'='*60}")
    print(f"  Total patients scanned:       {total_patients}")
    print(f"  Patients with HES features:   {patients_with_hes_features}")
    print(f"  Already labeled (skip):       {skipped_already_labeled}")
    print(f"  No HES record (skip):         {skipped_no_hes}")
    print(f"  HES before index date (skip): {skipped_before_index}")
    print(f"  HES after study end (skip):   {skipped_after_study}")
    print(f"  RELABELED (censored -> k=1):  {relabeled}")
    print(f"{'='*60}")


def main():
    torch.manual_seed(SEED)
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print(f"  Building HES-Static Dementia CR Dataset")
    print(f"  GP sequences + HES label aug + HES static features")
    print(f"  INDEX_AGE={INDEX_ON_AGE}, NO SAW, study period to 2022-10-31")
    print("=" * 70)
    print(f"  Database:       {PATH_TO_DB}")
    print(f"  Output:         {OUTPUT_DS}")
    print(f"  HES features:   {HES_FEATURES_PATH}")
    print()

    # Pre-create split directories
    for split in ["train", "val", "test"]:
        (Path(OUTPUT_DS) / f"split={split}").mkdir(parents=True, exist_ok=True)

    # Step 1: Build GP-only dataset using FoundationalDataModule
    print("===== Step 1: Build GP-only dataset =====")
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

    print(f"\nGP-only dataset built:")
    print(f"  Training patients:   {len(dm.train_set)}")
    print(f"  Validation patients: {len(dm.val_set)}")
    print(f"  Test patients:       {len(dm.test_set)}")

    # Step 2: Load HES summary features
    print("\n===== Step 2: Load HES summary features =====")
    with open(HES_FEATURES_PATH, "rb") as f:
        hes_data = pickle.load(f)
    hes_features = hes_data["features"]
    feature_names = hes_data["feature_names"]
    print(f"  Loaded {len(hes_features)} patients with {len(feature_names)} features")
    print(f"  Features: {feature_names}")

    # Step 3: Build HES dementia lookup (for label augmentation)
    print("\n===== Step 3: Build HES dementia lookup =====")
    hes_lookup = build_hes_dementia_lookup()

    # Step 4: Load year-of-birth lookup
    print("\n===== Step 4: Load year-of-birth lookup =====")
    yob_lookup = load_year_of_birth_lookup()

    # Step 5: Post-process parquet files (add HES columns + relabel)
    print("\n===== Step 5: Post-process parquet files =====")
    post_process_parquet_files(hes_lookup, yob_lookup, hes_features, feature_names)

    print("\nDone!")


if __name__ == "__main__":
    main()
