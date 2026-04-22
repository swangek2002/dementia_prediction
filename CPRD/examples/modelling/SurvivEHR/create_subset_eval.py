"""
create_subset_eval.py
=====================
Create a test-subset of the fusion dataset containing ONLY the patients
that also appear in the hes_aug test set. This enables apple-to-apple
C_td comparison between the fusion model and the hes_aug model on the
exact same patients.

Train/val splits are symlinked from the fusion dataset (unchanged).
"""

import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa

FUSION_DS = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_fusion/"
AUG_DS = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_aug/"
SUBSET_DS = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_fusion_subset_test/"


def get_patient_ids(ds_path, split="test"):
    pids = set()
    split_dir = os.path.join(ds_path, f"split={split}")
    for root, _, files in os.walk(split_dir):
        for fn in files:
            if not fn.endswith(".parquet"):
                continue
            df = pq.read_table(os.path.join(root, fn)).to_pandas()
            for _, row in df.iterrows():
                pids.add(int(row["PATIENT_ID"]))
    return pids


def main():
    print("Collecting hes_aug test patient IDs ...")
    aug_pids = get_patient_ids(AUG_DS, "test")
    print(f"  hes_aug test: {len(aug_pids)} patients")

    # Create subset directory
    subset_test_dir = os.path.join(SUBSET_DS, "split=test")
    os.makedirs(subset_test_dir, exist_ok=True)

    # Filter fusion test parquet to only aug patients
    # Parquet files are nested: split=test/COUNTRY=UK/HEALTH_AUTH=.../PRACTICE_ID=.../CHUNK=.../xxx.parquet
    fusion_test_dir = os.path.join(FUSION_DS, "split=test")
    total_kept = 0
    total_dropped = 0
    for root, _, files in os.walk(fusion_test_dir):
        for fn in sorted(files):
            if not fn.endswith(".parquet"):
                continue
            src_path = os.path.join(root, fn)
            table = pq.read_table(src_path)
            df = table.to_pandas()

            mask = df["PATIENT_ID"].apply(lambda x: int(x) in aug_pids)
            kept = mask.sum()
            dropped = (~mask).sum()
            total_kept += kept
            total_dropped += dropped

            if kept > 0:
                # Preserve the directory structure relative to fusion_test_dir
                rel = os.path.relpath(root, fusion_test_dir)
                dst_dir = os.path.join(subset_test_dir, rel)
                os.makedirs(dst_dir, exist_ok=True)
                filtered_df = df[mask].reset_index(drop=True)
                filtered_table = pa.Table.from_pandas(filtered_df, schema=table.schema)
                pq.write_table(filtered_table, os.path.join(dst_dir, fn))

    print(f"  fusion test filtered: kept={total_kept}, dropped={total_dropped}")

    # Symlink train/val from fusion dataset (needed for DataModule loading)
    for split in ["train", "val"]:
        src = os.path.join(FUSION_DS, f"split={split}")
        dst = os.path.join(SUBSET_DS, f"split={split}")
        if os.path.islink(dst):
            os.unlink(dst)
        elif os.path.isdir(dst):
            shutil.rmtree(dst)
        os.symlink(src, dst)
        print(f"  symlinked {dst} -> {src}")

    # Copy ancillary files needed by DataModule
    for ancillary in [
        "file_row_count_dict_train.pickle",
        "file_row_count_dict_val.pickle",
    ]:
        src = os.path.join(FUSION_DS, ancillary)
        dst = os.path.join(SUBSET_DS, ancillary)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Build new file_row_count_dict for test split
    import pickle
    test_row_counts = {}
    for root, _, files in os.walk(subset_test_dir):
        for fn in sorted(files):
            if not fn.endswith(".parquet"):
                continue
            fp = os.path.join(root, fn)
            n = pq.read_metadata(fp).num_rows
            test_row_counts[fp] = n
    with open(os.path.join(SUBSET_DS, "file_row_count_dict_test.pickle"), "wb") as f:
        pickle.dump(test_row_counts, f)
    print(f"  test row count dict: {sum(test_row_counts.values())} rows in {len(test_row_counts)} files")

    print(f"\nDone. Subset dataset at: {SUBSET_DS}")


if __name__ == "__main__":
    main()
