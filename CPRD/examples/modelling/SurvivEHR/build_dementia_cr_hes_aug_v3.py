"""
build_dementia_cr_hes_aug_v3.py
================================
Self-training: V3 dataset = V2 labels + pseudo-labeled dementia patients.

Takes V2 dataset and relabels top 1% DEATH/censored patients (identified by
V2 model inference) as dementia. Uses their original event time (death/censoring
time) as the dementia event time.

Usage:
    PYTHONPATH=/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD \
    python build_dementia_cr_hes_aug_v3.py
"""

import os
import shutil
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from collections import defaultdict

# ---- Paths ----
PSEUDO_CSV = "/Data0/swangek_data/991/CPRD/data/pseudo_dementia_patients_v3.csv"

DATASETS = [
    {
        "input": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_aug_v2/",
        "output": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_aug_v3/",
    },
    {
        "input": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v2/",
        "output": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v3/",
    },
]

DEMENTIA_READ_CODES_SET = set([
    "F110.", "Eu00.", "Eu01.", "Eu02z", "Eu002", "E00..",
    "Eu023", "Eu00z", "Eu025", "Eu01z", "E001.", "F1100",
    "Eu001", "E004.", "Eu000", "Eu02.", "Eu013", "E000.",
    "Eu01y", "E001z", "F1101", "Eu020", "E004z", "E0021",
    "Eu02y", "Eu012", "Eu011", "E00z.", "E0040", "E003.",
    "E0020",
])
HES_LABEL_CODE = "Eu02z"


def relabel_dataset(output_ds, pseudo_pids):
    """Relabel pseudo dementia patients in train split only.
    For these patients, change last EVENT to HES_LABEL_CODE.
    Keep the original last DATE (death/censoring time) as event time.
    """
    stats = defaultdict(int)

    for root, dirs, files in os.walk(output_ds):
        # Only modify train split
        if "split=train" not in root:
            continue

        for fn in files:
            if not fn.endswith(".parquet"):
                continue
            filepath = os.path.join(root, fn)
            table = pq.read_table(filepath)
            df = table.to_pandas()
            modified = False

            for idx, row in df.iterrows():
                pid = int(row["PATIENT_ID"])
                if pid not in pseudo_pids:
                    continue

                events = row["EVENT"]
                if len(events) == 0:
                    continue

                last_event = events[-1]

                # Only relabel DEATH or censored patients
                if last_event in DEMENTIA_READ_CODES_SET:
                    stats["already_dementia"] += 1
                    continue

                new_events = events.copy()
                new_events[-1] = HES_LABEL_CODE
                # Keep original date (death/censoring time) — no change to dates
                df.at[idx, "EVENT"] = new_events
                modified = True
                stats["relabeled"] += 1

                if last_event == "DEATH":
                    stats["from_death"] += 1
                else:
                    stats["from_censored"] += 1

            if modified:
                new_table = pa.Table.from_pandas(df, schema=table.schema)
                pq.write_table(new_table, filepath)

    return stats


def rebuild_row_count_pickles(output_ds):
    """Rebuild file_row_count_dict pickles (absolute paths)."""
    for split in ["train", "val", "test"]:
        row_count_dict = {}
        split_dir = os.path.join(output_ds, f"split={split}")
        for root, dirs, files in os.walk(split_dir):
            for fn in files:
                if not fn.endswith(".parquet"):
                    continue
                fp = os.path.join(root, fn)
                abs_path = os.path.abspath(fp)
                table = pq.read_table(fp)
                row_count_dict[abs_path] = len(table)
        out_path = os.path.join(output_ds, f"file_row_count_dict_{split}.pickle")
        with open(out_path, "wb") as f:
            pickle.dump(row_count_dict, f)
        total = sum(row_count_dict.values())
        print(f"    {split}: {len(row_count_dict)} files, {total} patients")


def verify_dataset(output_ds):
    """Print final label distribution."""
    label_counts = defaultdict(int)
    split_counts = defaultdict(lambda: defaultdict(int))
    total = 0
    for root, dirs, files in os.walk(output_ds):
        split = "unknown"
        if "split=train" in root: split = "train"
        elif "split=val" in root: split = "val"
        elif "split=test" in root: split = "test"

        for fn in files:
            if not fn.endswith(".parquet"):
                continue
            df = pq.read_table(os.path.join(root, fn)).to_pandas()
            for _, row in df.iterrows():
                total += 1
                events = row["EVENT"]
                if len(events) == 0:
                    label = "empty"
                else:
                    last = events[-1]
                    if last == "DEATH": label = "death"
                    elif last in DEMENTIA_READ_CODES_SET: label = "dementia"
                    else: label = "censored"
                label_counts[label] += 1
                split_counts[split][label] += 1

    print(f"  Total: {total}")
    for label in ["dementia", "death", "censored"]:
        print(f"    {label}: {label_counts[label]}")
    for split in ["train", "val", "test"]:
        sc = split_counts[split]
        total_s = sum(sc.values())
        print(f"  {split} ({total_s}):")
        for label in ["dementia", "death", "censored"]:
            print(f"    {label}: {sc[label]}")


def main():
    print("=" * 70)
    print("  Building V3 Dataset (Self-Training: V2 + pseudo labels)")
    print("=" * 70)

    # Load pseudo dementia patient IDs
    pseudo_df = pd.read_csv(PSEUDO_CSV)
    pseudo_pids = set(pseudo_df["patient_id"].astype(int).tolist())
    print(f"\nPseudo dementia patients to relabel: {len(pseudo_pids)}")
    print(f"  From DEATH: {len(pseudo_df[pseudo_df['label']=='death'])}")
    print(f"  From censored: {len(pseudo_df[pseudo_df['label']=='censored'])}")

    for ds_info in DATASETS:
        input_ds = ds_info["input"]
        output_ds = ds_info["output"]
        ds_name = os.path.basename(output_ds.rstrip("/"))
        print(f"\n{'#'*70}")
        print(f"  Processing: {ds_name}")
        print(f"{'#'*70}")

        # Copy V2 -> V3
        print(f"\n  Step 1: Copy V2 -> V3")
        if os.path.exists(output_ds):
            shutil.rmtree(output_ds)
        shutil.copytree(input_ds, output_ds)
        print("    Done.")

        # Relabel
        print(f"\n  Step 2: Relabel pseudo dementia patients (train only)")
        stats = relabel_dataset(output_ds, pseudo_pids)
        print(f"    Relabeled: {stats['relabeled']}")
        print(f"      From DEATH: {stats['from_death']}")
        print(f"      From censored: {stats['from_censored']}")
        print(f"    Already dementia (skipped): {stats['already_dementia']}")

        # Rebuild pickles
        print(f"\n  Step 3: Rebuild row count pickles")
        rebuild_row_count_pickles(output_ds)

        # Verify
        print(f"\n  Step 4: Verify label distribution")
        verify_dataset(output_ds)

    print("\nAll done!")


if __name__ == "__main__":
    main()
