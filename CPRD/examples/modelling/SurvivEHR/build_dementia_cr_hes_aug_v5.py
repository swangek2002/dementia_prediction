"""
build_dementia_cr_hes_aug_v5.py
================================
Third-round self-training: V5 dataset = V4 labels + new pseudo-labeled dementia patients.

Takes V4 dataset and relabels top 5% non-dementia patients (identified by
V4 model inference) as dementia. Filters censored patients with <2y observation.

More aggressive threshold (5% vs V4's 2%) — this is the deliberate test of whether
self-training continues to improve or starts to overfit.

Includes overlap analysis against earlier V2/V3 pseudo selections.

Usage:
    PYTHONPATH=/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD \
    python build_dementia_cr_hes_aug_v5.py
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
V4_INFERENCE_CSV = "/Data0/swangek_data/991/CPRD/data/train_cif_dementia_v4.csv"
V3_INFERENCE_CSV = "/Data0/swangek_data/991/CPRD/data/train_cif_dementia_v3.csv"
V2_INFERENCE_CSV = "/Data0/swangek_data/991/CPRD/data/train_cif_dementia_v2.csv"
V4_PSEUDO_CSV    = "/Data0/swangek_data/991/CPRD/data/pseudo_dementia_patients_v4.csv"
V3_PSEUDO_CSV    = "/Data0/swangek_data/991/CPRD/data/pseudo_dementia_patients_v3.csv"
V5_PSEUDO_CSV    = "/Data0/swangek_data/991/CPRD/data/pseudo_dementia_patients_v5.csv"

# Selection criteria
TOP_PCT     = 0.05      # top 5% (V4 used 2%, V3 used 1%) — MORE AGGRESSIVE
MIN_FOLLOWUP_YEARS = 2.0  # exclude censored patients with <2y observation

DATASETS = [
    {
        "input": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_aug_v4/",
        "output": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_aug_v5/",
    },
    {
        "input": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v4/",
        "output": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v5/",
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


def select_v5_candidates():
    """Select NEW pseudo-labeled patients for V5 from V4 model inference output.

    Returns:
        v5_pseudo_df: DataFrame with columns [patient_id, label, event_time_years, cif_dementia_at_event]
        threshold: CIF threshold used (top 5%)
        prior_pseudo_pids: set of V3 (771) + V4 (824) pseudo patient IDs (for overlap analysis)
    """
    df = pd.read_csv(V4_INFERENCE_CSV)
    print(f"Loaded V4 inference: {len(df)} patients")
    print(f"  Labels: {df['label'].value_counts().to_dict()}")

    # Pool: non-dementia patients in V4 train (excludes real dementia + V3 pseudo + V4 pseudo,
    # since all of those are now labeled 'dementia' in V4 train)
    pool = df[df['label'].isin(['death', 'censored'])].copy()
    print(f"  Candidate pool (death + censored in V4 train): {len(pool)}")

    # Compute threshold = top 5% CIF (more aggressive than V4's 2%)
    threshold = pool['cif_dementia_at_event'].quantile(1.0 - TOP_PCT)
    print(f"  Top {TOP_PCT*100:.0f}% threshold: CIF >= {threshold:.4f}")

    candidates = pool[pool['cif_dementia_at_event'] >= threshold].copy()
    print(f"  Candidates above threshold: {len(candidates)}")
    print(f"    From DEATH:    {(candidates['label']=='death').sum()}")
    print(f"    From censored: {(candidates['label']=='censored').sum()}")

    # Filter censored with <2y followup (using prediction-point-relative time)
    short_censored = (candidates['label'] == 'censored') & (candidates['event_time_years'] < MIN_FOLLOWUP_YEARS)
    print(f"  Excluding {short_censored.sum()} censored patients with <{MIN_FOLLOWUP_YEARS}y prediction window")

    final = candidates[~short_censored].copy()
    print(f"\n  Final V5 pseudo candidates: {len(final)}")
    print(f"    From DEATH:    {(final['label']=='death').sum()}")
    print(f"    From censored: {(final['label']=='censored').sum()}")

    # Load V3 pseudo (771) + V4 pseudo (824) for overlap analysis
    v3_pseudo_df = pd.read_csv(V3_PSEUDO_CSV)
    v4_pseudo_df = pd.read_csv(V4_PSEUDO_CSV)
    v3_pseudo_pids = set(v3_pseudo_df['patient_id'].astype(int).tolist())
    v4_pseudo_pids = set(v4_pseudo_df['patient_id'].astype(int).tolist())
    prior_pseudo_pids = v3_pseudo_pids | v4_pseudo_pids
    print(f"\n  Prior pseudo pool: V3 ({len(v3_pseudo_pids)}) + V4 ({len(v4_pseudo_pids)}) = {len(prior_pseudo_pids)}")

    return final, threshold, prior_pseudo_pids


def overlap_analysis(v5_candidates, prior_pseudo_pids):
    """Compare V5 candidates against V2's CIF rankings.

    Key questions:
    1. Were V5 candidates ranked HIGH or LOW in V2 model?
       - If high → V3 is reinforcing V2's existing tendencies (mild confirmation bias)
       - If low/middle → V3 is finding genuinely new signal
    2. NOTE: V3's 771 pseudo are LABELED dementia in V3 train, so by construction
       they are NOT in V4's candidate pool. So direct ID-overlap = 0 by design.
    """
    print(f"\n{'='*70}")
    print("OVERLAP ANALYSIS: V5 candidates vs V2's predictions")
    print(f"{'='*70}")

    # Load V2 inference (V2 model on V2 train)
    v2_df = pd.read_csv(V2_INFERENCE_CSV)
    print(f"\nLoaded V2 inference: {len(v2_df)} patients")

    # Direct ID overlap with V3's 771 (should be 0 by construction)
    v5_pids = set(v5_candidates['patient_id'].astype(int).tolist())
    direct_overlap = v5_pids & prior_pseudo_pids
    print(f"\nDirect ID overlap: V5 candidates ({len(v5_pids)}) ∩ (V3+V4 pseudo, {len(prior_pseudo_pids)}) = {len(direct_overlap)}")
    print("  (Expected to be ~0 since V3+V4 pseudo are labeled dementia in V4 train, "
          "thus excluded from V5's candidate pool by construction)")

    # Compute V2 CIF rank for each patient in V2 (among non-dementia in V2)
    v2_pool = v2_df[v2_df['label'].isin(['death', 'censored'])].copy()
    v2_pool = v2_pool.sort_values('cif_dementia_at_event', ascending=False).reset_index(drop=True)
    v2_pool['v2_rank'] = np.arange(len(v2_pool)) + 1
    v2_pool['v2_pct_rank'] = v2_pool['v2_rank'] / len(v2_pool)
    pid_to_v2rank = dict(zip(v2_pool['patient_id'].astype(int), v2_pool['v2_pct_rank']))
    pid_to_v2cif = dict(zip(v2_pool['patient_id'].astype(int), v2_pool['cif_dementia_at_event']))

    # For each V4 candidate, look up V2 rank
    v5_candidates_copy = v5_candidates.copy()
    v5_candidates_copy['patient_id'] = v5_candidates_copy['patient_id'].astype(int)
    v5_candidates_copy['v2_pct_rank'] = v5_candidates_copy['patient_id'].map(pid_to_v2rank)
    v5_candidates_copy['v2_cif'] = v5_candidates_copy['patient_id'].map(pid_to_v2cif)

    # Distribution of V5 candidates across V2's ranking percentiles
    print(f"\nFor V5 candidates ({len(v5_candidates_copy)}), where did V2 rank them?")
    print("  (V2 rank percentile = 0.01 means top 1% in V2's ranking)")

    bins = [0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.01]
    bin_labels = ['top 0-1%', 'top 1-2%', 'top 2-5%', 'top 5-10%', 'top 10-20%', 'top 20-50%', 'bottom 50%']
    has_v2 = v5_candidates_copy['v2_pct_rank'].notna()
    no_v2 = (~has_v2).sum()
    if no_v2 > 0:
        print(f"  Not in V2 pool (was V2 dementia or absent): {no_v2}")

    sub = v5_candidates_copy[has_v2].copy()
    sub['v2_rank_bin'] = pd.cut(sub['v2_pct_rank'], bins=bins, labels=bin_labels, include_lowest=True)
    counts = sub['v2_rank_bin'].value_counts().reindex(bin_labels, fill_value=0)
    print(f"\n  V2 rank bin            | V5 candidates | % of V5 candidates")
    print(f"  {'-'*22} | {'-'*13} | {'-'*18}")
    for label in bin_labels:
        n = counts[label]
        pct = n / len(sub) * 100 if len(sub) > 0 else 0
        print(f"  {label:22s} | {n:13d} | {pct:6.1f}%")

    # Median / mean V2 rank of V5 candidates
    print(f"\n  V2 rank percentile of V5 candidates:")
    print(f"    median: {sub['v2_pct_rank'].median()*100:.2f}%")
    print(f"    mean:   {sub['v2_pct_rank'].mean()*100:.2f}%")
    print(f"    p25:    {sub['v2_pct_rank'].quantile(0.25)*100:.2f}%")
    print(f"    p75:    {sub['v2_pct_rank'].quantile(0.75)*100:.2f}%")

    # Set overlap if we'd used 2% in V2 (sanity check)
    v2_top2pct_pids = set(v2_pool[v2_pool['v2_pct_rank'] <= 0.02]['patient_id'].astype(int).tolist())
    v2_top1pct_pids = set(v2_pool[v2_pool['v2_pct_rank'] <= 0.01]['patient_id'].astype(int).tolist())
    overlap_2pct = v5_pids & v2_top2pct_pids
    overlap_1pct = v5_pids & v2_top1pct_pids
    print(f"\nSet overlap with hypothetical V2 thresholds:")
    print(f"  V5 ∩ V2-top-1%: {len(overlap_1pct)} / {len(v5_pids)} ({len(overlap_1pct)/len(v5_pids)*100:.1f}%)")
    print(f"    (note: V3's 771 actual pseudo were chosen from V2-top-1% with filter)")
    print(f"  V5 ∩ V2-top-2%: {len(overlap_2pct)} / {len(v5_pids)} ({len(overlap_2pct)/len(v5_pids)*100:.1f}%)")

    # Interpretation summary
    pct_in_v2_top5 = (sub['v2_pct_rank'] <= 0.05).mean() * 100
    pct_in_v2_top20 = (sub['v2_pct_rank'] <= 0.20).mean() * 100
    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print(f"  • {pct_in_v2_top5:.0f}% of V5 candidates were already in V2's top 5%")
    print(f"  • {pct_in_v2_top20:.0f}% of V5 candidates were already in V2's top 20%")
    if pct_in_v2_top5 > 70:
        print("  → STRONG OVERLAP: V3 is largely consolidating V2's existing rankings.")
        print("    V4 may show diminishing returns or confirmation bias.")
    elif pct_in_v2_top5 > 40:
        print("  → MODERATE OVERLAP: Some new signal, some reinforcement.")
        print("    V4 expected to give modest improvement.")
    else:
        print("  → LOW OVERLAP: V3 finding genuinely new candidates beyond V2's top picks.")
        print("    V4 has higher chance of meaningful improvement.")
    print(f"{'='*70}")

    return v5_candidates_copy


def relabel_dataset(output_ds, pseudo_pids):
    """Same as V3 builder: relabel pseudo dementia patients in train split only."""
    stats = defaultdict(int)

    for root, dirs, files in os.walk(output_ds):
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
                if last_event in DEMENTIA_READ_CODES_SET:
                    stats["already_dementia"] += 1
                    continue
                new_events = events.copy()
                new_events[-1] = HES_LABEL_CODE
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
    print("  Building V5 Dataset (3rd-Round Self-Training: V4 + new pseudo labels, top 5%)")
    print("=" * 70)

    # Step 1: Select V5 candidates
    print(f"\n[Step 1] Selecting V5 pseudo candidates from V4 inference (top {TOP_PCT*100:.0f}%)")
    v5_candidates, threshold, prior_pseudo_pids = select_v5_candidates()

    # Step 2: Overlap analysis
    print(f"\n[Step 2] Overlap analysis (V5 vs V2 rankings)")
    v5_candidates_with_v2rank = overlap_analysis(v5_candidates, prior_pseudo_pids)

    # Save V5 pseudo CSV
    v5_candidates_with_v2rank.to_csv(V5_PSEUDO_CSV, index=False)
    print(f"\n  Saved V5 pseudo candidates to: {V5_PSEUDO_CSV}")

    pseudo_pids = set(v5_candidates['patient_id'].astype(int).tolist())

    # Step 3-6: Build datasets
    for ds_info in DATASETS:
        input_ds = ds_info["input"]
        output_ds = ds_info["output"]
        ds_name = os.path.basename(output_ds.rstrip("/"))
        print(f"\n{'#'*70}")
        print(f"  Processing: {ds_name}")
        print(f"{'#'*70}")

        print(f"\n  Step A: Copy V4 -> V5")
        if os.path.exists(output_ds):
            shutil.rmtree(output_ds)
        shutil.copytree(input_ds, output_ds)
        print("    Done.")

        print(f"\n  Step B: Relabel new pseudo dementia patients (train only)")
        stats = relabel_dataset(output_ds, pseudo_pids)
        print(f"    Relabeled: {stats['relabeled']}")
        print(f"      From DEATH: {stats['from_death']}")
        print(f"      From censored: {stats['from_censored']}")
        print(f"    Already dementia (skipped): {stats['already_dementia']}")

        print(f"\n  Step C: Rebuild row count pickles")
        rebuild_row_count_pickles(output_ds)

        print(f"\n  Step D: Verify final label distribution")
        verify_dataset(output_ds)

    print("\nAll done! V5 dataset ready for training.")


if __name__ == "__main__":
    main()
