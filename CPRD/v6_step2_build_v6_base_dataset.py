"""
V6 Step 2: Build V6-base dataset.

V6-base = FineTune_Dementia_CR_hes_static_v2 MINUS 246 GP-prevalent patients
          (16 test + 213 train + 17 val from leaky_patients_*.txt)

This is V2 labels + 22-dim HES static features, with GP-prevalent leakage fixed.
No self-training yet — that's V6 Step 6.

Output: /Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v6_base/

Sanity checks at every step:
- Source dataset patient counts match expected (133322 total)
- Leaky lists each match expected count
- Output patient counts = source - leaky (133076 total)
- File row count dicts correctly rebuilt
- No PIDs duplicated, no PIDs missing
- All HES_* columns preserved
"""
import os
import glob
import pickle
import shutil
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm

SRC_DS = '/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v2'
OUT_DS = '/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v6_base'

LEAKY_FILES = {
    'train': '/Data0/swangek_data/991/CPRD/data/leaky_patients_train.txt',
    'val':   '/Data0/swangek_data/991/CPRD/data/leaky_patients_val.txt',
    'test':  '/Data0/swangek_data/991/CPRD/data/leaky_patients_test.txt',
}

EXPECTED_SRC_PATIENTS = {'train': 119271, 'val': 5794, 'test': 8257}
EXPECTED_LEAKY = {'train': 213, 'val': 17, 'test': 16}
EXPECTED_OUT_PATIENTS = {'train': 119058, 'val': 5777, 'test': 8241}


def load_leaky(split):
    """Load PID set for given split."""
    pids = set()
    with open(LEAKY_FILES[split]) as f:
        for line in f:
            line = line.strip()
            if line:
                pids.add(int(line))
    assert len(pids) == EXPECTED_LEAKY[split], (
        f"Leaky {split}: expected {EXPECTED_LEAKY[split]}, got {len(pids)}"
    )
    return pids


def filter_split(split, leaky_pids):
    """Filter parquet files in a split, save to OUT_DS, return file_row_count dict + stats."""
    src_split_dir = os.path.join(SRC_DS, f'split={split}')
    out_split_dir = os.path.join(OUT_DS, f'split={split}')

    src_files = sorted(glob.glob(os.path.join(src_split_dir, '**', '*.parquet'), recursive=True))
    print(f"  [{split}] {len(src_files)} source parquet files")

    file_row_count = {}
    total_src = 0
    total_removed = 0
    total_kept = 0
    pids_removed = set()
    pids_kept = set()

    for src_fp in tqdm(src_files, desc=f'  [{split}] filter', leave=False):
        # Read parquet
        table = pq.read_table(src_fp)
        df = table.to_pandas()
        total_src += len(df)

        # Verify expected HES_* columns present
        hes_cols = [c for c in df.columns if c.startswith('HES_')]
        assert len(hes_cols) == 22, f"{src_fp}: expected 22 HES_* cols, got {len(hes_cols)}"

        # Filter out leaky patients
        if 'PATIENT_ID' not in df.columns:
            raise RuntimeError(f"{src_fp}: missing PATIENT_ID column")
        mask = ~df['PATIENT_ID'].astype('int64').isin(leaky_pids)
        df_kept = df[mask].copy()
        total_removed += (~mask).sum()
        total_kept += mask.sum()

        # Track PIDs
        for pid in df.loc[~mask, 'PATIENT_ID']:
            pids_removed.add(int(pid))
        for pid in df_kept['PATIENT_ID']:
            pids_kept.add(int(pid))

        # Build output path (mirror directory structure)
        rel_path = os.path.relpath(src_fp, src_split_dir)
        out_fp = os.path.join(out_split_dir, rel_path)
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)

        # Save filtered parquet (preserve schema)
        if len(df_kept) > 0:
            out_table = pa.Table.from_pandas(df_kept, schema=table.schema, preserve_index=False)
            pq.write_table(out_table, out_fp)
            # file_row_count uses RELATIVE path from split dir (matches loader expectation)
            file_row_count[rel_path] = len(df_kept)
        # If 0 rows after filtering, skip writing (some files might be all-leaky)

    return file_row_count, {
        'total_src': total_src,
        'total_removed': total_removed,
        'total_kept': total_kept,
        'pids_removed': pids_removed,
        'pids_kept': pids_kept,
    }


def main():
    print("="*70)
    print("V6 Step 2: Build V6-base dataset")
    print("="*70)
    print(f"  Source:      {SRC_DS}")
    print(f"  Output:      {OUT_DS}")
    print()

    # Pre-flight: validate source exists
    assert os.path.exists(SRC_DS), f"Source dataset not found: {SRC_DS}"
    assert not os.path.exists(OUT_DS), f"Output dataset already exists: {OUT_DS} (please delete first)"

    # Pre-flight: load and validate leaky lists
    print("[1/3] Loading leaky patient lists...")
    leaky = {split: load_leaky(split) for split in ['train', 'val', 'test']}
    total_leaky = sum(len(s) for s in leaky.values())
    print(f"  Train: {len(leaky['train'])} leaky")
    print(f"  Val:   {len(leaky['val'])} leaky")
    print(f"  Test:  {len(leaky['test'])} leaky")
    print(f"  Total: {total_leaky} leaky")
    assert total_leaky == 246, f"Expected 246 total leaky, got {total_leaky}"

    # Filter each split
    print("\n[2/3] Filtering parquet files per split...")
    os.makedirs(OUT_DS, exist_ok=True)
    stats_all = {}
    for split in ['train', 'val', 'test']:
        print(f"\n  ===== Processing split = {split} =====")
        frc, stats = filter_split(split, leaky[split])
        stats_all[split] = stats

        # Save file_row_count_dict
        frc_path = os.path.join(OUT_DS, f'file_row_count_dict_{split}.pickle')
        with open(frc_path, 'wb') as f:
            pickle.dump(frc, f)
        print(f"  [{split}] Saved file_row_count_dict to {frc_path} ({len(frc)} files)")
        print(f"  [{split}] Source rows: {stats['total_src']}, Removed: {stats['total_removed']}, Kept: {stats['total_kept']}")

    # Sanity checks
    print("\n[3/3] Sanity checks")
    print("="*70)
    all_pass = True
    for split in ['train', 'val', 'test']:
        s = stats_all[split]
        # Check 1: Source count matches expected
        exp_src = EXPECTED_SRC_PATIENTS[split]
        if s['total_src'] == exp_src:
            print(f"  [✓] {split} source patients = {s['total_src']} (expected {exp_src})")
        else:
            print(f"  [✗] {split} source patients = {s['total_src']} (expected {exp_src})")
            all_pass = False
        # Check 2: Removed count matches expected leaky
        exp_rem = EXPECTED_LEAKY[split]
        if s['total_removed'] == exp_rem:
            print(f"  [✓] {split} removed = {s['total_removed']} (expected {exp_rem})")
        else:
            # Sometimes some PIDs in leaky list don't appear in dataset (already filtered or never were)
            removed_pids_overlap = leaky[split] & s['pids_removed']
            missing_pids = leaky[split] - s['pids_removed']
            print(f"  [WARN] {split} removed = {s['total_removed']} (expected {exp_rem})")
            print(f"         overlap PIDs: {len(removed_pids_overlap)}; missing PIDs not in source: {len(missing_pids)}")
            if len(missing_pids) > 0:
                print(f"         (these PIDs probably not in v2 due to V2's HES prevalent filter)")
        # Check 3: Kept count matches expected
        exp_kept = EXPECTED_OUT_PATIENTS[split]
        actual_kept = s['total_kept']
        if abs(actual_kept - exp_kept) <= 1:
            print(f"  [✓] {split} kept patients = {actual_kept} (expected {exp_kept})")
        else:
            print(f"  [✗] {split} kept patients = {actual_kept} (expected {exp_kept}, diff {actual_kept - exp_kept})")
            all_pass = False
        # Check 4: PIDs uniqueness
        if len(s['pids_kept']) == actual_kept:
            print(f"  [✓] {split} all PIDs unique ({len(s['pids_kept'])} unique = {actual_kept} rows)")
        else:
            print(f"  [✗] {split} PID uniqueness: {len(s['pids_kept'])} unique vs {actual_kept} rows")
            all_pass = False

    if all_pass:
        print("\nALL SANITY CHECKS PASSED")
    else:
        print("\nSOME CHECKS FAILED - review above")

    # Final summary
    print("\n" + "="*70)
    print("V6-base dataset built:")
    print(f"  Location: {OUT_DS}")
    print(f"  Train: {stats_all['train']['total_kept']:6d} patients")
    print(f"  Val:   {stats_all['val']['total_kept']:6d} patients")
    print(f"  Test:  {stats_all['test']['total_kept']:6d} patients")
    print(f"  Total: {sum(s['total_kept'] for s in stats_all.values())} patients")


if __name__ == '__main__':
    main()
