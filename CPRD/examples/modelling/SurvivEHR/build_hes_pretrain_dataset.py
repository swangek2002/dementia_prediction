"""
build_hes_pretrain_dataset.py
=============================
Build HES pretrain dataset using FoundationalDataModule.
Uses the HES-only database and GP practice splits for consistent splitting.

Important: Do NOT pass overwrite_meta_information — let FoundationalDataModule
build new meta info from HES data (ICD-10 vocab, not Read v2).

Output: data/FoundationalModel/PreTrain_HES/
  - Parquet files (split=train/val/test)
  - meta_information_custom.pickle (HES-specific tokenizer vocab)
  - practice_id_splits.pickle

Usage:
    python build_hes_pretrain_dataset.py
"""

import os
import pickle
import torch
import logging
import pandas as pd
from pathlib import Path

from FastEHR.dataloader import FoundationalDataModule

HES_DB = "/Data0/swangek_data/991/CPRD/data/hes_pretrain_database.db"
HES_DS = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain_HES/"
SPLITS = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/practice_id_splits.pickle"

SEED = 1337
NUM_THREADS = 12


def main():
    torch.manual_seed(SEED)
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  Building HES Pretrain Dataset")
    print("=" * 60)
    print(f"  HES DB:     {HES_DB}")
    print(f"  Output:     {HES_DS}")
    print(f"  GP splits:  {SPLITS}")
    print()

    # Verify HES database exists
    if not os.path.exists(HES_DB):
        raise FileNotFoundError(
            f"HES database not found: {HES_DB}\n"
            "Run build_hes_database.py first."
        )

    # Pre-create output and split directories (FoundationalDataModule expects them)
    for split in ["train", "val", "test"]:
        (Path(HES_DS) / f"split={split}").mkdir(parents=True, exist_ok=True)

    print("Building dataset (this may take a while)...")

    # Step 1: Build parquet files from HES database
    #   Use load=False first just to generate parquets + meta_information
    #   include_measurements=False since HES has no measurement tables
    from FastEHR.dataloader.dataset.dataset_polars import PolarsDataset
    polars_dataset = PolarsDataset(path_to_db=HES_DB)
    polars_dataset.fit(
        path=HES_DS,
        overwrite_practice_ids=SPLITS,
        include_measurements=False,
        drop_missing_data=False,
        drop_empty_dynamic=True,
        num_threads=NUM_THREADS,
    )

    # Step 2: Patch meta_information — add empty measurement_tables
    #   The tokenizer expects this key to exist
    meta_path = os.path.join(HES_DS, "meta_information.pickle")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    if "measurement_tables" not in meta:
        meta["measurement_tables"] = pd.DataFrame(
            columns=["event", "count", "count_obs", "digest",
                     "min", "max", "mean", "approx_lqr", "approx_uqr"]
        )
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        print("  Patched meta_information: added empty measurement_tables")

    # Step 3: Now load the dataset properly (load=True)
    dm = FoundationalDataModule(
        path_to_db=HES_DB,
        path_to_ds=HES_DS,
        load=True,
        tokenizer="tabular",
        seed=SEED,
    )

    vocab_size = dm.train_set.tokenizer.vocab_size

    print(f"\nHES pretrain dataset built:")
    print(f"  Train:      {len(dm.train_set)}")
    print(f"  Val:        {len(dm.val_set)}")
    print(f"  Test:       {len(dm.test_set)}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Output:     {HES_DS}")

    # Verify meta_information was created
    meta_path = os.path.join(HES_DS, "meta_information_custom.pickle")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        print(f"\n  Meta info saved: {meta_path}")
        if hasattr(meta, 'keys'):
            print(f"  Meta keys: {list(meta.keys())}")
    else:
        # Check alternative name
        for fn in os.listdir(HES_DS):
            if "meta" in fn.lower():
                print(f"  Meta info found: {os.path.join(HES_DS, fn)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
