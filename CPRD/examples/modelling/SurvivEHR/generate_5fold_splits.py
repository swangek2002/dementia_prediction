"""
generate_5fold_splits.py
========================
Generate 5-fold cross-validation splits at the practice_id level.

114 total practices -> 5 folds:
  - Each fold: ~23 practices for test, ~9 for val, ~82 for train
  - Val is drawn from the non-test practices (roughly 10% of remaining)

Output: 5 pickle files in OUTPUT_DIR, each with {'train': [...], 'val': [...], 'test': [...]}

Usage:
    python generate_5fold_splits.py
"""

import pickle
import numpy as np
from pathlib import Path

ORIGINAL_SPLITS_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/practice_id_splits.pickle"
OUTPUT_DIR = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/cv5_splits/"
SEED = 1337
N_FOLDS = 5
VAL_FRACTION = 0.10  # 10% of non-test practices for validation


def main():
    # Load original splits to get all practice IDs
    with open(ORIGINAL_SPLITS_PATH, "rb") as f:
        orig = pickle.load(f)

    all_practices = []
    for split_name in ["train", "val", "test"]:
        all_practices.extend(orig[split_name])

    all_practices = sorted(set(all_practices))
    print(f"Total practices: {len(all_practices)}")

    rng = np.random.RandomState(SEED)
    shuffled = np.array(all_practices)
    rng.shuffle(shuffled)

    # Split into 5 roughly equal folds
    folds = np.array_split(shuffled, N_FOLDS)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx in range(N_FOLDS):
        test_practices = folds[fold_idx].tolist()
        non_test = []
        for j in range(N_FOLDS):
            if j != fold_idx:
                non_test.extend(folds[j].tolist())

        # Shuffle non-test, then split off val
        rng2 = np.random.RandomState(SEED + fold_idx)
        non_test_arr = np.array(non_test)
        rng2.shuffle(non_test_arr)

        n_val = max(1, int(len(non_test_arr) * VAL_FRACTION))
        val_practices = non_test_arr[:n_val].tolist()
        train_practices = non_test_arr[n_val:].tolist()

        splits = {
            "train": train_practices,
            "val": val_practices,
            "test": test_practices,
        }

        out_path = output_dir / f"practice_id_splits_fold{fold_idx}.pickle"
        with open(out_path, "wb") as f:
            pickle.dump(splits, f)

        print(f"Fold {fold_idx}: train={len(train_practices)}, "
              f"val={len(val_practices)}, test={len(test_practices)} -> {out_path}")

    print("\nDone. All 5-fold split files saved.")


if __name__ == "__main__":
    main()
