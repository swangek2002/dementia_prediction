"""
build_hes_pretrain_dataset.py
=============================
Build HES pretrain dataset using FoundationalDataModule.
Uses the HES-only database and GP practice splits for consistent splitting.

Output: /Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain_HES/
"""

import torch
import logging
from FastEHR.dataloader import FoundationalDataModule

HES_DB = "/Data0/swangek_data/991/CPRD/data/hes_pretrain_database.db"
HES_DS = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain_HES/"
SPLITS = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/practice_id_splits.pickle"

SEED = 1337
NUM_THREADS = 12


def main():
    torch.manual_seed(SEED)
    logging.basicConfig(level=logging.INFO)

    print("Building HES pretrain dataset...")
    dm = FoundationalDataModule(
        path_to_db=HES_DB,
        path_to_ds=HES_DS,
        load=False,
        tokenizer="tabular",
        overwrite_practice_ids=SPLITS,
        # Do NOT pass overwrite_meta_information — let it build new meta info from HES data
        min_workers=NUM_THREADS,
        seed=SEED,
    )

    print(f"HES pretrain dataset built:")
    print(f"  Train: {len(dm.train_set)}")
    print(f"  Val:   {len(dm.val_set)}")
    print(f"  Test:  {len(dm.test_set)}")


if __name__ == "__main__":
    main()
