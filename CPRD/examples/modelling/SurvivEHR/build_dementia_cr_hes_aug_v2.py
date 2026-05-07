"""
build_dementia_cr_hes_aug_v2.py
================================
Improved HES-augmented competing-risk dataset for dementia prediction.

Improvements over v1 (build_dementia_cr_hes_aug.py):
  1. Relabel DEATH patients with HES dementia → dementia (use HES date)
  2. Relabel DEATH patients with death-cause dementia (no HES) → dementia (use death date)
  3. Remove prevalent cases: patients with HES dementia BEFORE index date
  4. Keep censored relabeling from v1 (censored + HES dementia → dementia)

Reads from the existing hes_aug v1 dataset and writes to a new directory.

Usage:
    PYTHONPATH=/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD \
    python build_dementia_cr_hes_aug_v2.py
"""

import os
import sqlite3
import shutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from collections import defaultdict

from hes_dementia_lookup import build_hes_dementia_lookup, _is_dementia_icd10

# ---- Paths ----
# We process BOTH hes_aug and hes_static (same labels, hes_static has extra columns)
DATASETS = [
    {
        "input": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_aug/",
        "output": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_aug_v2/",
    },
    {
        "input": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static/",
        "output": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v2/",
    },
]
PATH_TO_DB = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
DEATH_CAUSE_CSV = "/Data0/swangek_data/991/CPRD/data/death_cause.csv"
DEATH_CSV = "/Data0/swangek_data/991/CPRD/data/death.csv"

INDEX_ON_AGE = 72
STUDY_END_DATE = pd.Timestamp("2022-10-31")

DEMENTIA_READ_CODES = [
    "F110.", "Eu00.", "Eu01.", "Eu02z", "Eu002", "E00..",
    "Eu023", "Eu00z", "Eu025", "Eu01z", "E001.", "F1100",
    "Eu001", "E004.", "Eu000", "Eu02.", "Eu013", "E000.",
    "Eu01y", "E001z", "F1101", "Eu020", "E004z", "E0021",
    "Eu02y", "Eu012", "Eu011", "E00z.", "E0040", "E003.",
    "E0020",
]
DEMENTIA_READ_CODES_SET = set(DEMENTIA_READ_CODES)

# HES-augmented label code (unspecified dementia Read code)
HES_LABEL_CODE = "Eu02z"


def load_year_of_birth_lookup():
    """Load {PATIENT_ID: YEAR_OF_BIRTH (pd.Timestamp)} from the database."""
    conn = sqlite3.connect(PATH_TO_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT PATIENT_ID, YEAR_OF_BIRTH FROM static_table")
    rows = cursor.fetchall()
    conn.close()
    yob_lookup = {}
    for pid, yob_str in rows:
        yob_lookup[int(pid)] = pd.Timestamp(yob_str)
    print(f"  Year-of-birth lookup: {len(yob_lookup)} patients")
    return yob_lookup


def build_death_cause_dementia_lookup():
    """
    Build lookup of patients who died OF dementia (ICD-10 in death_cause).
    Returns: {eid (int): death_date (pd.Timestamp)}
    Only includes patients where death_cause has a dementia ICD-10 code.
    """
    dc = pd.read_csv(DEATH_CAUSE_CSV, dtype={"eid": str, "cause_icd10": str})
    dementia_mask = dc["cause_icd10"].apply(_is_dementia_icd10)
    dc_dem = dc[dementia_mask]
    dem_eids = set(dc_dem["eid"].astype(int).unique())

    # Get death dates
    death = pd.read_csv(DEATH_CSV, usecols=["eid", "date_of_death"], dtype={"eid": str})
    death["date_of_death"] = pd.to_datetime(death["date_of_death"], errors="coerce")
    death["eid_int"] = death["eid"].astype(int)
    death = death.dropna(subset=["date_of_death"])

    # Only keep patients with dementia death cause
    death_dem = death[death["eid_int"].isin(dem_eids)]
    # Take earliest death date per patient (should be unique but just in case)
    lookup = death_dem.groupby("eid_int")["date_of_death"].min().to_dict()
    print(f"  Death-cause dementia lookup: {len(lookup)} patients")
    return lookup


def build_all_death_dates_lookup():
    """Build lookup of death dates for all patients. {eid (int): death_date (pd.Timestamp)}"""
    death = pd.read_csv(DEATH_CSV, usecols=["eid", "date_of_death"], dtype={"eid": str})
    death["date_of_death"] = pd.to_datetime(death["date_of_death"], errors="coerce")
    death["eid_int"] = death["eid"].astype(int)
    death = death.dropna(subset=["date_of_death"])
    lookup = death.groupby("eid_int")["date_of_death"].min().to_dict()
    print(f"  All death dates lookup: {len(lookup)} patients")
    return lookup


def post_process(output_ds, hes_lookup, death_cause_lookup, yob_lookup):
    """
    Walk all parquet files and apply corrections:
      1. For DEATH patients with HES dementia (after index, before study end):
         → relabel as dementia, use HES date
      2. For DEATH patients with death-cause dementia (no HES dementia):
         → relabel as dementia, use death date
      3. For censored patients with HES dementia (after index, before study end):
         → relabel as dementia, use HES date (same as v1)
      4. Remove prevalent cases: any patient with HES dementia BEFORE index date
    """
    stats = defaultdict(int)

    for root, dirs, files in os.walk(output_ds):
        for fn in files:
            if not fn.endswith(".parquet"):
                continue
            filepath = os.path.join(root, fn)
            table = pq.read_table(filepath)
            df = table.to_pandas()
            modified = False
            rows_to_drop = []

            for idx, row in df.iterrows():
                stats["total"] += 1
                events = row["EVENT"]
                dates = row["DATE"]

                if len(events) == 0:
                    continue

                last_event = events[-1]
                pid = int(row["PATIENT_ID"])

                # Compute index date
                yob = yob_lookup.get(pid)
                if yob is None:
                    stats["no_yob"] += 1
                    continue
                index_date = yob + pd.DateOffset(years=INDEX_ON_AGE)

                # --- Check for prevalent HES dementia (before index date) ---
                if pid in hes_lookup:
                    hes_date = hes_lookup[pid]
                    if hes_date <= index_date:
                        # Prevalent case: remove this patient
                        rows_to_drop.append(idx)
                        stats["removed_prevalent"] += 1
                        continue

                # --- Already labeled as dementia: skip ---
                if last_event in DEMENTIA_READ_CODES_SET:
                    stats["already_dementia"] += 1
                    continue

                # --- DEATH patients ---
                if last_event == "DEATH":
                    # Priority 1: HES dementia (after index, before study end)
                    if pid in hes_lookup:
                        hes_date = hes_lookup[pid]
                        # We already checked hes_date > index_date above
                        if hes_date <= STUDY_END_DATE:
                            new_events = events.copy()
                            new_dates = dates.copy()
                            new_events[-1] = HES_LABEL_CODE
                            new_dates[-1] = np.datetime64(hes_date, "us")
                            df.at[idx, "EVENT"] = new_events
                            df.at[idx, "DATE"] = new_dates
                            modified = True
                            stats["death_relabeled_hes"] += 1
                            continue
                        else:
                            stats["death_hes_after_study"] += 1

                    # Priority 2: Death-cause dementia (no valid HES)
                    if pid in death_cause_lookup:
                        death_date = death_cause_lookup[pid]
                        if death_date > index_date and death_date <= STUDY_END_DATE:
                            new_events = events.copy()
                            new_dates = dates.copy()
                            new_events[-1] = HES_LABEL_CODE
                            new_dates[-1] = np.datetime64(death_date, "us")
                            df.at[idx, "EVENT"] = new_events
                            df.at[idx, "DATE"] = new_dates
                            modified = True
                            stats["death_relabeled_deathcause"] += 1
                            continue

                    # Not relabeled: keep as DEATH
                    stats["death_kept"] += 1
                    continue

                # --- Censored patients ---
                # (These were already processed by v1 hes_aug, but let's
                #  also handle any that v1 might have missed)
                if pid in hes_lookup:
                    hes_date = hes_lookup[pid]
                    # Already checked hes_date > index_date above
                    if hes_date <= STUDY_END_DATE:
                        new_events = events.copy()
                        new_dates = dates.copy()
                        new_events[-1] = HES_LABEL_CODE
                        new_dates[-1] = np.datetime64(hes_date, "us")
                        df.at[idx, "EVENT"] = new_events
                        df.at[idx, "DATE"] = new_dates
                        modified = True
                        stats["censored_relabeled_hes"] += 1
                        continue

                stats["censored_kept"] += 1

            # Drop prevalent rows
            if rows_to_drop:
                df = df.drop(rows_to_drop).reset_index(drop=True)
                modified = True

            if modified:
                new_table = pa.Table.from_pandas(df, schema=table.schema)
                pq.write_table(new_table, filepath)

    return stats


def verify_dataset(output_ds):
    """Print final label distribution for a dataset."""
    from collections import Counter
    label_counts = Counter()
    split_counts = defaultdict(lambda: Counter())
    total_final = 0
    for root, dirs, files in os.walk(output_ds):
        split = "unknown"
        if "split=train" in root:
            split = "train"
        elif "split=val" in root:
            split = "val"
        elif "split=test" in root:
            split = "test"

        for fn in files:
            if not fn.endswith(".parquet"):
                continue
            df = pq.read_table(os.path.join(root, fn)).to_pandas()
            for _, row in df.iterrows():
                total_final += 1
                events = row["EVENT"]
                if len(events) == 0:
                    label = "empty"
                else:
                    last = events[-1]
                    if last == "DEATH":
                        label = "death"
                    elif last in DEMENTIA_READ_CODES_SET:
                        label = "dementia"
                    else:
                        label = "censored"
                label_counts[label] += 1
                split_counts[split][label] += 1

    print(f"  Total patients (after removal): {total_final}")
    for label in ["dementia", "death", "censored"]:
        print(f"    {label}: {label_counts[label]}")
    print()
    for split in ["train", "val", "test"]:
        sc = split_counts[split]
        total_s = sum(sc.values())
        print(f"  {split} ({total_s}):")
        for label in ["dementia", "death", "censored"]:
            print(f"    {label}: {sc[label]}")


def rebuild_row_count_pickles(output_ds):
    """Rebuild file_row_count_dict pickles after modifying parquets.
    Uses absolute paths as keys (matching the original format)."""
    import pickle
    for split in ["train", "val", "test"]:
        row_count_dict = {}
        split_dir = os.path.join(output_ds, f"split={split}")
        for root, dirs, files in os.walk(split_dir):
            for fn in files:
                if not fn.endswith(".parquet"):
                    continue
                fp = os.path.join(root, fn)
                # Use absolute path as key (matching original pickle format)
                abs_path = os.path.abspath(fp)
                table = pq.read_table(fp)
                row_count_dict[abs_path] = len(table)
        out_path = os.path.join(output_ds, f"file_row_count_dict_{split}.pickle")
        with open(out_path, "wb") as f:
            pickle.dump(row_count_dict, f)
        total = sum(row_count_dict.values())
        print(f"    {split}: {len(row_count_dict)} files, {total} patients")


def main():
    print("=" * 70)
    print("  Building HES-Augmented Dementia CR Dataset V2")
    print("  Corrections: relabel DEATH+HES/deathcause, remove prevalent")
    print("=" * 70)

    # Build lookups (shared across datasets)
    print("\n===== Build lookups =====")
    hes_lookup = build_hes_dementia_lookup()
    death_cause_lookup = build_death_cause_dementia_lookup()
    yob_lookup = load_year_of_birth_lookup()

    for ds_info in DATASETS:
        input_ds = ds_info["input"]
        output_ds = ds_info["output"]
        ds_name = os.path.basename(input_ds.rstrip("/"))
        print(f"\n{'#'*70}")
        print(f"  Processing: {ds_name}")
        print(f"{'#'*70}")

        # Step 1: Copy
        print(f"\n  Step 1: Copy {ds_name} -> v2")
        if os.path.exists(output_ds):
            print(f"    Removing existing v2 directory...")
            shutil.rmtree(output_ds)
        shutil.copytree(input_ds, output_ds)
        print("    Done.")

        # Step 2: Apply corrections
        print(f"\n  Step 2: Apply corrections")
        stats = post_process(output_ds, hes_lookup, death_cause_lookup, yob_lookup)

        print(f"\n  {'='*60}")
        print(f"  V2 Post-Processing Summary ({ds_name})")
        print(f"  {'='*60}")
        print(f"  Total patients scanned:              {stats['total']}")
        print(f"  Already dementia (no change):        {stats['already_dementia']}")
        print(f"  REMOVED prevalent (HES before idx):  {stats['removed_prevalent']}")
        print(f"  DEATH → dementia (HES):              {stats['death_relabeled_hes']}")
        print(f"  DEATH → dementia (death cause):      {stats['death_relabeled_deathcause']}")
        print(f"  DEATH kept as-is:                    {stats['death_kept']}")
        print(f"  DEATH HES after study end:           {stats['death_hes_after_study']}")
        print(f"  Censored → dementia (HES, new):      {stats['censored_relabeled_hes']}")
        print(f"  Censored kept:                       {stats['censored_kept']}")
        print(f"  No YOB:                              {stats['no_yob']}")
        print(f"  {'='*60}")

        # Step 3: Rebuild pickles
        print(f"\n  Step 3: Rebuild row count pickles")
        rebuild_row_count_pickles(output_ds)

        # Step 4: Verify
        print(f"\n  Step 4: Verify final label distribution")
        verify_dataset(output_ds)

    print("\nAll done!")


if __name__ == "__main__":
    main()
