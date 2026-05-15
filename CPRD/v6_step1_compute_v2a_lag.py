"""
V6 Step 1: Compute V2-type-A empirical lag distribution.

V2-type-A patients = DEATH patients whose original (V1/hes_aug) label was DEATH,
                     but V2 relabeled them as dementia using the HES dementia date.

For each: lag = HES_dementia_date - last_pre_index_GP_event_date  (in years)

This empirical lag distribution will be sampled to assign realistic pseudo-event
times to V6 self-training candidates (V3-style top 1%), replacing the broken
"set to last_GP_visit_date" approach that V3/V4/V5 used.

Outputs:
  data/v2A_lag_distribution.npy  — array of lag values in years (only positive)
  figs/v2A_lag_distribution.png  — histogram
  v2A_lag_stats.json             — summary statistics + counts
"""
import os
import glob
import json
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V1_TRAIN_GLOB = '/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_aug/split=train/**/*.parquet'
V2_TRAIN_GLOB = '/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_aug_v2/split=train/**/*.parquet'
OUT_NPY = '/Data0/swangek_data/991/CPRD/data/v2A_lag_distribution.npy'
OUT_JSON = '/Data0/swangek_data/991/CPRD/v2A_lag_stats.json'
OUT_PNG = '/Data0/swangek_data/991/CPRD/figs/v2A_lag_distribution.png'
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

HES_LABEL_CODE = 'Eu02z'  # what V2 used to relabel
DEMENTIA_READ_CODES_SET = set([
    "F110.", "Eu00.", "Eu01.", "Eu02z", "Eu002", "E00..",
    "Eu023", "Eu00z", "Eu025", "Eu01z", "E001.", "F1100",
    "Eu001", "E004.", "Eu000", "Eu02.", "Eu013", "E000.",
    "Eu01y", "E001z", "F1101", "Eu020", "E004z", "E0021",
    "Eu02y", "Eu012", "Eu011", "E00z.", "E0040", "E003.",
    "E0020",
])


def build_pid_map(files, label):
    """Build {pid: (last_event, last_date_ns, second_to_last_event, second_to_last_date_ns)}."""
    m = {}
    for fp in tqdm(files, desc=f'load {label}', leave=False):
        df = pq.read_table(fp).to_pandas()
        for _, row in df.iterrows():
            pid = int(row['PATIENT_ID'])
            ev = list(row['EVENT'])
            dt = list(row['DATE'])
            if len(ev) < 2:
                continue
            m[pid] = (str(ev[-1]), dt[-1], str(ev[-2]), dt[-2])
    return m


def main():
    print("=" * 70)
    print("V6 STEP 1: Compute V2-type-A empirical lag distribution")
    print("=" * 70)

    print("\n[1/4] Loading V1 (hes_aug) train parquet...")
    v1_files = sorted(glob.glob(V1_TRAIN_GLOB, recursive=True))
    print(f"  Files: {len(v1_files)}")
    v1_map = build_pid_map(v1_files, 'V1')
    print(f"  V1 patients with ≥2 events: {len(v1_map)}")

    print("\n[2/4] Loading V2 (hes_aug_v2) train parquet...")
    v2_files = sorted(glob.glob(V2_TRAIN_GLOB, recursive=True))
    print(f"  Files: {len(v2_files)}")
    v2_map = build_pid_map(v2_files, 'V2')
    print(f"  V2 patients with ≥2 events: {len(v2_map)}")

    # Sanity check: V2 should have FEWER patients than V1 (V2 removed 487 prevalent)
    diff = len(v1_map) - len(v2_map)
    print(f"\n  V1 - V2 patient count = {diff}  (expected ~487 prevalent removed)")
    assert diff > 0 and diff < 600, f"Unexpected diff in patient counts: {diff}"

    print("\n[3/4] Identifying V2-type-A patients (V1 DEATH → V2 Eu02z with date change)...")
    type_A_lags_years = []   # positive lags only
    type_A_negative_lags = []  # for diagnostics
    type_A_all_lags = []       # for diagnostics (signed)
    type_B_count = 0           # V1 DEATH → V2 Eu02z but same date (death-cause dementia)
    death_unchanged = 0        # V1 DEATH and V2 DEATH (no relabel)
    pid_set_typeA = []

    for pid, v2_data in v2_map.items():
        if pid not in v1_map:
            continue
        v1_last_event, v1_last_date, _, _ = v1_map[pid]
        v2_last_event, v2_last_date, v2_2nd_last_event, v2_2nd_last_date = v2_data

        if v1_last_event != 'DEATH':
            continue  # not a DEATH-relabel candidate

        if v2_last_event == HES_LABEL_CODE:
            # Was relabeled in V2
            if v2_last_date < v1_last_date:
                # Type A: V2 used earlier HES date
                # Compute lag = HES_dementia_date - last_pre_index_event_date
                hes_date = np.datetime64(v2_last_date, 'D')
                last_gp_date = np.datetime64(v2_2nd_last_date, 'D')
                lag_days = (hes_date - last_gp_date).astype('timedelta64[D]').astype(int)
                lag_years = lag_days / 365.25
                type_A_all_lags.append(lag_years)
                if lag_years > 0:
                    type_A_lags_years.append(lag_years)
                else:
                    type_A_negative_lags.append(lag_years)
                pid_set_typeA.append(pid)
            elif v2_last_date == v1_last_date:
                # Type B: V2 used death date (death-cause dementia)
                type_B_count += 1
            else:
                # V2 date > V1 date — unexpected, log
                print(f"    UNEXPECTED: pid={pid}, V2_date ({v2_last_date}) > V1_date ({v1_last_date})")
        else:
            # V1=DEATH but V2 last event != Eu02z and != DEATH? Or V2=DEATH (unchanged)?
            if v2_last_event == 'DEATH':
                death_unchanged += 1

    print(f"  Type A candidates (V2 used HES date < death date): {len(pid_set_typeA)} total")
    print(f"    Positive lags (HES after last GP event):  {len(type_A_lags_years)}")
    print(f"    Negative lags (HES before last GP event): {len(type_A_negative_lags)}")
    print(f"  Type B (V2 used death date): {type_B_count}")
    print(f"  DEATH unchanged in V2: {death_unchanged}")

    # Sanity: expected ~1123 type A + 274 type B (from PROJECT_KNOWLEDGE Section 6.6)
    print(f"\n  Expected from docs: ~1123 type A, ~274 type B")
    print(f"  Observed: {len(pid_set_typeA)} type A, {type_B_count} type B")

    lags = np.array(type_A_lags_years, dtype=np.float64)
    print(f"\n[4/4] Lag distribution stats (positive only, n={len(lags)}):")
    pcts = np.percentile(lags, [5, 25, 50, 75, 95])
    stats = {
        'n_positive': int(len(lags)),
        'n_negative': int(len(type_A_negative_lags)),
        'n_typeA_total': int(len(pid_set_typeA)),
        'n_typeB': int(type_B_count),
        'mean_years': float(np.mean(lags)),
        'std_years': float(np.std(lags)),
        'median_years': float(np.median(lags)),
        'p5_years': float(pcts[0]),
        'p25_years': float(pcts[1]),
        'p75_years': float(pcts[3]),
        'p95_years': float(pcts[4]),
        'min_years': float(np.min(lags)),
        'max_years': float(np.max(lags)),
    }
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:20s} = {v:.4f}")
        else:
            print(f"  {k:20s} = {v}")

    print(f"\n  Saving positive lag array to {OUT_NPY}")
    np.save(OUT_NPY, lags)

    with open(OUT_JSON, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats to {OUT_JSON}")

    # Plot histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(lags, bins=50, edgecolor='black')
    axes[0].axvline(np.median(lags), color='red', linestyle='--', label=f'Median = {np.median(lags):.2f}y')
    axes[0].axvline(np.mean(lags), color='orange', linestyle='--', label=f'Mean = {np.mean(lags):.2f}y')
    axes[0].set_xlabel('Lag (years)')
    axes[0].set_ylabel('Number of V2-type-A patients')
    axes[0].set_title(f'V2-type-A lag: HES dementia date - last GP event date\n(positive lags only, n={len(lags)})')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Log-scale x for long tail
    axes[1].hist(lags, bins=50, edgecolor='black')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Lag (years, log scale)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Same distribution, log x-axis (shows tail)')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=100)
    print(f"  Saved histogram to {OUT_PNG}")

    # Final sanity checks
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)
    checks = [
        ('Type A count between 1000-1200', 1000 <= len(pid_set_typeA) <= 1200),
        ('Median lag between 0.5 and 5 years', 0.5 < stats['median_years'] < 5),
        ('Max lag < 25 years', stats['max_years'] < 25),
        ('Mean lag > median (right-skewed)', stats['mean_years'] > stats['median_years']),
        ('All saved lags positive', np.all(lags > 0)),
    ]
    for desc, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  [{status}] {desc}")
    if all(c[1] for c in checks):
        print("\nALL SANITY CHECKS PASSED ✓")
    else:
        print("\nSANITY FAILED — investigate before proceeding to Step 2")


if __name__ == '__main__':
    main()
