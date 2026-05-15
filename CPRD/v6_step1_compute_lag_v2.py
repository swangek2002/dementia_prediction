"""
V6 Step 1 (v2): Compute empirical lag distribution from V1-Type-C patients.

V1-Type-C = ALIVE CENSORED patients whose HES caught dementia after they lost GP contact.
            (V1/hes_aug relabeled their last event from censored → Eu02z with HES dementia date)

Why this is better than V2-Type-A:
- V2-Type-A patients are DEATH patients (terminal trajectory bias)
- V1-Type-C are alive censored — EXACTLY the profile of our V6 pseudo candidates
- Lag = HES dementia date - last actual GP event date (more relevant timing)

Identification:
- V0 (FineTune_Dementia_CR, no HES aug): patient's last event was NOT dementia and NOT DEATH
                                          (i.e., originally censored — last event is a GP visit / measurement)
- V1 (hes_aug): same patient now has last event = Eu02z with HES dementia date
- → This is V1-Type-C, an alive censored patient relabeled to dementia by V1

For each: lag = V1_last_date - V0_last_date  (years)
Filter lag > 0
Save distribution for V6 use.
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

V0_TRAIN_GLOB = '/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR/split=train/**/*.parquet'
V1_TRAIN_GLOB = '/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_aug/split=train/**/*.parquet'

OUT_NPY = '/Data0/swangek_data/991/CPRD/data/v1_typeC_lag_distribution.npy'
OUT_JSON = '/Data0/swangek_data/991/CPRD/v1_typeC_lag_stats.json'
OUT_PNG = '/Data0/swangek_data/991/CPRD/figs/v1_typeC_lag_distribution.png'
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

HES_LABEL_CODE = 'Eu02z'
DEMENTIA_READ_CODES_SET = set([
    "F110.", "Eu00.", "Eu01.", "Eu02z", "Eu002", "E00..",
    "Eu023", "Eu00z", "Eu025", "Eu01z", "E001.", "F1100",
    "Eu001", "E004.", "Eu000", "Eu02.", "Eu013", "E000.",
    "Eu01y", "E001z", "F1101", "Eu020", "E004z", "E0021",
    "Eu02y", "Eu012", "Eu011", "E00z.", "E0040", "E003.",
    "E0020",
])


def build_pid_map(files, label):
    """{pid: (last_event, last_date, n_events)}"""
    m = {}
    for fp in tqdm(files, desc=f'load {label}', leave=False):
        df = pq.read_table(fp).to_pandas()
        for _, row in df.iterrows():
            pid = int(row['PATIENT_ID'])
            ev = list(row['EVENT'])
            dt = list(row['DATE'])
            if len(ev) < 1:
                continue
            m[pid] = (str(ev[-1]), dt[-1], len(ev))
    return m


def main():
    print("=" * 70)
    print("V6 STEP 1 v2: V1-Type-C (alive censored→dementia via HES) empirical lag")
    print("=" * 70)

    print("\n[1/3] Loading V0 (original, no HES aug) train parquet...")
    v0_files = sorted(glob.glob(V0_TRAIN_GLOB, recursive=True))
    print(f"  V0 files: {len(v0_files)}")
    v0_map = build_pid_map(v0_files, 'V0')
    print(f"  V0 patients with ≥1 event: {len(v0_map)}")

    print("\n[2/3] Loading V1 (hes_aug) train parquet...")
    v1_files = sorted(glob.glob(V1_TRAIN_GLOB, recursive=True))
    print(f"  V1 files: {len(v1_files)}")
    v1_map = build_pid_map(v1_files, 'V1')
    print(f"  V1 patients with ≥1 event: {len(v1_map)}")

    # Sanity
    diff = abs(len(v0_map) - len(v1_map))
    print(f"\n  V0 vs V1 patient count diff: {diff} (expected ~0, V1 should match V0)")

    print("\n[3/3] Identifying V1-Type-C: V0 censored → V1 Eu02z with HES date...")
    type_C_lags_years = []
    type_C_negative = []
    type_C_total = 0
    pid_set = []

    for pid in v0_map.keys():
        if pid not in v1_map:
            continue
        v0_last_event, v0_last_date, _ = v0_map[pid]
        v1_last_event, v1_last_date, _ = v1_map[pid]

        # Criterion for Type C:
        # - V0 last event is NOT a dementia code (would mean originally diagnosed) AND not DEATH
        # - V1 last event = Eu02z (V1 relabeled it)
        # - V1 last date is later than V0 last date (HES date > GP last date)
        if v0_last_event in DEMENTIA_READ_CODES_SET:
            continue  # originally dementia, not relabel
        if v0_last_event == 'DEATH':
            continue  # original DEATH — V1 doesn't relabel these
        if v1_last_event != HES_LABEL_CODE:
            continue  # not relabeled by V1
        if v1_last_date <= v0_last_date:
            continue  # negative or zero lag — unusual

        # OK this is V1-Type-C
        type_C_total += 1
        v0_dt = np.datetime64(v0_last_date, 'D')
        v1_dt = np.datetime64(v1_last_date, 'D')
        lag_days = (v1_dt - v0_dt).astype('timedelta64[D]').astype(int)
        lag_years = lag_days / 365.25
        if lag_years > 0:
            type_C_lags_years.append(lag_years)
            pid_set.append(pid)
        else:
            type_C_negative.append(lag_years)

    print(f"\n  V1-Type-C identified: {type_C_total} (expected ~4000)")
    print(f"  Positive lags: {len(type_C_lags_years)}")
    print(f"  Negative or zero: {len(type_C_negative)}")

    lags = np.array(type_C_lags_years, dtype=np.float64)
    pcts = np.percentile(lags, [5, 25, 50, 75, 95])
    stats = {
        'source': 'V1-Type-C (alive censored → V1 relabeled to dementia via HES)',
        'n_typeC_total': type_C_total,
        'n_positive': int(len(lags)),
        'n_negative_or_zero': int(len(type_C_negative)),
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
    print("\n  Lag distribution stats (positive only):")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k:22s} = {v:.4f}")
        else:
            print(f"    {k:22s} = {v}")

    np.save(OUT_NPY, lags)
    print(f"\n  Saved positive lag array to {OUT_NPY}")
    with open(OUT_JSON, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats to {OUT_JSON}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(lags, bins=80, edgecolor='black')
    axes[0].axvline(np.median(lags), color='red', linestyle='--', label=f'Median = {np.median(lags):.2f}y')
    axes[0].axvline(np.mean(lags), color='orange', linestyle='--', label=f'Mean = {np.mean(lags):.2f}y')
    axes[0].axvline(5.0, color='gray', linestyle=':', label='5y horizon')
    axes[0].set_xlabel('Lag (years)')
    axes[0].set_ylabel('Number of V1-Type-C patients')
    axes[0].set_title(f'V1-Type-C lag (alive censored → HES dementia)\nn={len(lags)}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(0, min(20, lags.max() * 1.05))

    axes[1].hist(lags, bins=80, edgecolor='black', cumulative=True, density=True)
    axes[1].axhline(0.5, color='red', linestyle='--', label='50%')
    axes[1].axvline(5.0, color='gray', linestyle=':', label='5y horizon')
    axes[1].set_xlabel('Lag (years)')
    axes[1].set_ylabel('Cumulative fraction')
    axes[1].set_title('Cumulative distribution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, min(20, lags.max() * 1.05))

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=100)
    print(f"  Saved histogram to {OUT_PNG}")

    # Sanity checks
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)
    checks = [
        ('Type C count in [3000, 5000] (expected ~4000-4097)', 3000 <= type_C_total <= 5000),
        ('Median lag < V2-Type-A median (5.17y)', stats['median_years'] < 5.0),
        ('Median lag > 0.3y (reasonable for "lost GP contact then HES diagnosed")', stats['median_years'] > 0.3),
        ('Max lag < 25 years', stats['max_years'] < 25),
        ('All positive', np.all(lags > 0)),
        ('Mean > median (right-skewed expected)', stats['mean_years'] > stats['median_years']),
    ]
    for desc, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  [{status}] {desc}")

    return stats


if __name__ == '__main__':
    main()
