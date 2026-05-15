"""
build_dementia_cr_hes_aug_v4.py
================================
Second-round self-training: V4 dataset = V3 labels + new pseudo-labeled dementia patients.

Takes V3 dataset and relabels top 2% non-dementia patients (identified by
V3 model inference) as dementia. Filters censored patients with <2y observation.

Includes overlap analysis against V2's 771 pseudo candidates to check for
confirmation bias.

Usage:
    PYTHONPATH=/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD \
    python build_dementia_cr_hes_aug_v4.py
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
V3_INFERENCE_CSV = "/Data0/swangek_data/991/CPRD/data/train_cif_dementia_v3.csv"
V2_INFERENCE_CSV = "/Data0/swangek_data/991/CPRD/data/train_cif_dementia_v2.csv"
V3_PSEUDO_CSV    = "/Data0/swangek_data/991/CPRD/data/pseudo_dementia_patients_v3.csv"
V4_PSEUDO_CSV    = "/Data0/swangek_data/991/CPRD/data/pseudo_dementia_patients_v4.csv"

# Selection criteria
TOP_PCT     = 0.02      # top 2% (V3 used 1%)
MIN_FOLLOWUP_YEARS = 2.0  # exclude censored patients with <2y observation

DATASETS = [
    {
        "input": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_aug_v3/",
        "output": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_aug_v4/",
    },
    {
        "input": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v3/",
        "output": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v4/",
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


def select_v4_candidates():
    """Select new pseudo-labeled patients for V4 from V3 inference output.

    Returns:
        v4_pseudo_df: DataFrame with columns [patient_id, label, event_time_years, cif_dementia_at_event]
        threshold: CIF threshold used
        v3_pseudo_pids: set of V3's 771 pseudo patient IDs (for overlap analysis)
    """
    df = pd.read_csv(V3_INFERENCE_CSV)
    print(f"Loaded V3 inference: {len(df)} patients")
    print(f"  Labels: {df['label'].value_counts().to_dict()}")

    # Pool: non-dementia patients in V3 train (excludes both real dementia and V3's 771 pseudo,
    # since those are now labeled 'dementia' in V3)
    pool = df[df['label'].isin(['death', 'censored'])].copy()
    print(f"  Candidate pool (death + censored in V3): {len(pool)}")

    # Compute threshold = top 2% CIF
    threshold = pool['cif_dementia_at_event'].quantile(1.0 - TOP_PCT)
    print(f"  Top {TOP_PCT*100:.0f}% threshold: CIF >= {threshold:.4f}")

    candidates = pool[pool['cif_dementia_at_event'] >= threshold].copy()
    print(f"  Candidates above threshold: {len(candidates)}")
    print(f"    From DEATH:    {(candidates['label']=='death').sum()}")
    print(f"    From censored: {(candidates['label']=='censored').sum()}")

    # Filter censored with <2y followup (event_time_years = (last_age - 72) for them)
    # event_time_years comes from V3 inference, not the same as V2 prediction
    # Actually in inference_train_cif.py, event_time_years = age_delta_scaled * SUPERVISED_TIME_SCALE
    # which is the time from prediction point (2nd-to-last event) to last event
    # NOT time from index. We need to be careful here.
    # For the V3->V4 case, we use the same logic as V2->V3: filter censored with short prediction-window observation
    short_censored = (candidates['label'] == 'censored') & (candidates['event_time_years'] < MIN_FOLLOWUP_YEARS)
    print(f"  Excluding {short_censored.sum()} censored patients with <{MIN_FOLLOWUP_YEARS}y prediction window")

    final = candidates[~short_censored].copy()
    print(f"\n  Final V4 pseudo candidates: {len(final)}")
    print(f"    From DEATH:    {(final['label']=='death').sum()}")
    print(f"    From censored: {(final['label']=='censored').sum()}")

    # Load V3 pseudo (the 771) for overlap analysis
    v3_pseudo_df = pd.read_csv(V3_PSEUDO_CSV)
    v3_pseudo_pids = set(v3_pseudo_df['patient_id'].astype(int).tolist())

    return final, threshold, v3_pseudo_pids


def overlap_analysis(v4_candidates, v3_pseudo_pids):
    """Compare V4 candidates against V2's CIF rankings.

    Key questions:
    1. Were V4 candidates ranked HIGH or LOW in V2 model?
       - If high → V3 is reinforcing V2's existing tendencies (mild confirmation bias)
       - If low/middle → V3 is finding genuinely new signal
    2. NOTE: V3's 771 pseudo are LABELED dementia in V3 train, so by construction
       they are NOT in V4's candidate pool. So direct ID-overlap = 0 by design.
    """
    print(f"\n{'='*70}")
    print("OVERLAP ANALYSIS: V4 candidates vs V2's predictions")
    print(f"{'='*70}")

    # Load V2 inference (V2 model on V2 train)
    v2_df = pd.read_csv(V2_INFERENCE_CSV)
    print(f"\nLoaded V2 inference: {len(v2_df)} patients")

    # Direct ID overlap with V3's 771 (should be 0 by construction)
    v4_pids = set(v4_candidates['patient_id'].astype(int).tolist())
    direct_overlap = v4_pids & v3_pseudo_pids
    print(f"\nDirect ID overlap: V4 candidates ({len(v4_pids)}) ∩ V3's 771 = {len(direct_overlap)}")
    print("  (Expected to be ~0 since V3's 771 are labeled dementia in V3 train, "
          "thus excluded from V4's candidate pool by construction)")

    # Compute V2 CIF rank for each patient in V2 (among non-dementia in V2)
    v2_pool = v2_df[v2_df['label'].isin(['death', 'censored'])].copy()
    v2_pool = v2_pool.sort_values('cif_dementia_at_event', ascending=False).reset_index(drop=True)
    v2_pool['v2_rank'] = np.arange(len(v2_pool)) + 1
    v2_pool['v2_pct_rank'] = v2_pool['v2_rank'] / len(v2_pool)
    pid_to_v2rank = dict(zip(v2_pool['patient_id'].astype(int), v2_pool['v2_pct_rank']))
    pid_to_v2cif = dict(zip(v2_pool['patient_id'].astype(int), v2_pool['cif_dementia_at_event']))

    # For each V4 candidate, look up V2 rank
    v4_candidates_copy = v4_candidates.copy()
    v4_candidates_copy['patient_id'] = v4_candidates_copy['patient_id'].astype(int)
    v4_candidates_copy['v2_pct_rank'] = v4_candidates_copy['patient_id'].map(pid_to_v2rank)
    v4_candidates_copy['v2_cif'] = v4_candidates_copy['patient_id'].map(pid_to_v2cif)

    # Distribution of V4 candidates across V2's ranking percentiles
    print(f"\nFor V4 candidates ({len(v4_candidates_copy)}), where did V2 rank them?")
    print("  (V2 rank percentile = 0.01 means top 1% in V2's ranking)")

    bins = [0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.01]
    bin_labels = ['top 0-1%', 'top 1-2%', 'top 2-5%', 'top 5-10%', 'top 10-20%', 'top 20-50%', 'bottom 50%']
    has_v2 = v4_candidates_copy['v2_pct_rank'].notna()
    no_v2 = (~has_v2).sum()
    if no_v2 > 0:
        print(f"  Not in V2 pool (was V2 dementia or absent): {no_v2}")

    sub = v4_candidates_copy[has_v2].copy()
    sub['v2_rank_bin'] = pd.cut(sub['v2_pct_rank'], bins=bins, labels=bin_labels, include_lowest=True)
    counts = sub['v2_rank_bin'].value_counts().reindex(bin_labels, fill_value=0)
    print(f"\n  V2 rank bin            | V4 candidates | % of V4 candidates")
    print(f"  {'-'*22} | {'-'*13} | {'-'*18}")
    for label in bin_labels:
        n = counts[label]
        pct = n / len(sub) * 100 if len(sub) > 0 else 0
        print(f"  {label:22s} | {n:13d} | {pct:6.1f}%")

    # Median / mean V2 rank of V4 candidates
    print(f"\n  V2 rank percentile of V4 candidates:")
    print(f"    median: {sub['v2_pct_rank'].median()*100:.2f}%")
    print(f"    mean:   {sub['v2_pct_rank'].mean()*100:.2f}%")
    print(f"    p25:    {sub['v2_pct_rank'].quantile(0.25)*100:.2f}%")
    print(f"    p75:    {sub['v2_pct_rank'].quantile(0.75)*100:.2f}%")

    # Set overlap if we'd used 2% in V2 (sanity check)
    v2_top2pct_pids = set(v2_pool[v2_pool['v2_pct_rank'] <= 0.02]['patient_id'].astype(int).tolist())
    v2_top1pct_pids = set(v2_pool[v2_pool['v2_pct_rank'] <= 0.01]['patient_id'].astype(int).tolist())
    overlap_2pct = v4_pids & v2_top2pct_pids
    overlap_1pct = v4_pids & v2_top1pct_pids
    print(f"\nSet overlap with hypothetical V2 thresholds:")
    print(f"  V4 ∩ V2-top-1%: {len(overlap_1pct)} / {len(v4_pids)} ({len(overlap_1pct)/len(v4_pids)*100:.1f}%)")
    print(f"    (note: V3's 771 actual pseudo were chosen from V2-top-1% with filter)")
    print(f"  V4 ∩ V2-top-2%: {len(overlap_2pct)} / {len(v4_pids)} ({len(overlap_2pct)/len(v4_pids)*100:.1f}%)")

    # Interpretation summary
    pct_in_v2_top5 = (sub['v2_pct_rank'] <= 0.05).mean() * 100
    pct_in_v2_top20 = (sub['v2_pct_rank'] <= 0.20).mean() * 100
    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print(f"  • {pct_in_v2_top5:.0f}% of V4 candidates were already in V2's top 5%")
    print(f"  • {pct_in_v2_top20:.0f}% of V4 candidates were already in V2's top 20%")
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

    return v4_candidates_copy


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
    print("  Building V4 Dataset (2nd-Round Self-Training: V3 + new pseudo labels)")
    print("=" * 70)

    # Step 1: Select V4 candidates
    print(f"\n[Step 1] Selecting V4 pseudo candidates from V3 inference (top {TOP_PCT*100:.0f}%)")
    v4_candidates, threshold, v3_pseudo_pids = select_v4_candidates()

    # Step 2: Overlap analysis
    print(f"\n[Step 2] Overlap analysis (V4 vs V2 rankings)")
    v4_candidates_with_v2rank = overlap_analysis(v4_candidates, v3_pseudo_pids)

    # Save V4 pseudo CSV
    v4_candidates_with_v2rank.to_csv(V4_PSEUDO_CSV, index=False)
    print(f"\n  Saved V4 pseudo candidates to: {V4_PSEUDO_CSV}")

    pseudo_pids = set(v4_candidates['patient_id'].astype(int).tolist())

    # Step 3-6: Build datasets
    for ds_info in DATASETS:
        input_ds = ds_info["input"]
        output_ds = ds_info["output"]
        ds_name = os.path.basename(output_ds.rstrip("/"))
        print(f"\n{'#'*70}")
        print(f"  Processing: {ds_name}")
        print(f"{'#'*70}")

        print(f"\n  Step A: Copy V3 -> V4")
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

    print("\nAll done! V4 dataset ready for training.")


if __name__ == "__main__":
    main()
