"""
V6 Step 1 (v3): Compute empirical lag from V1-Type-C using direct HES lookup match.

Method (cleanest):
1. Build hes_dementia_lookup = {pid: earliest_HES_dementia_date} from hesin*.csv
2. For each patient in V1 (hes_aug) train parquet:
   - If V1.last_event == 'Eu02z' AND pid in hes_dementia_lookup
     AND V1.last_date == hes_lookup[pid]  (V1 used HES date as event date)
     AND V1.second_to_last_event NOT in dementia codes (was originally censored)
   - Then this is V1-relabeled (Type C, alive censored)
   - Compute lag = V1.last_date - V1.second_to_last_date  (in years)

Filter lag > 0
Save distribution.
"""
import os
import glob
import json
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V1_TRAIN_GLOB = '/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_aug/split=train/**/*.parquet'
HESIN_CSV = '/Data0/swangek_data/991/CPRD/data/hesin.csv'
HESIN_DIAG_CSV = '/Data0/swangek_data/991/CPRD/data/hesin_diag.csv'

OUT_NPY = '/Data0/swangek_data/991/CPRD/data/v1_typeC_lag_distribution.npy'
OUT_JSON = '/Data0/swangek_data/991/CPRD/v1_typeC_lag_stats.json'
OUT_PNG = '/Data0/swangek_data/991/CPRD/figs/v1_typeC_lag_distribution.png'
OUT_LOOKUP_PICKLE = '/Data0/swangek_data/991/CPRD/data/hes_dementia_lookup.pickle'
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
HES_DEMENTIA_PREFIXES = ('F00', 'F01', 'F02', 'G30')
HES_DEMENTIA_EXACT = {'F03'}


def is_dementia_icd10(code):
    if not code or not isinstance(code, str):
        return False
    code = code.strip()
    if code in HES_DEMENTIA_EXACT:
        return True
    return any(code.startswith(p) for p in HES_DEMENTIA_PREFIXES)


def build_hes_dementia_lookup():
    """Build {patient_id: earliest_HES_dementia_date} from hesin + hesin_diag csv."""
    if os.path.exists(OUT_LOOKUP_PICKLE):
        print(f"  Cached lookup found at {OUT_LOOKUP_PICKLE}, loading...")
        with open(OUT_LOOKUP_PICKLE, 'rb') as f:
            return pickle.load(f)

    print("  Reading hesin.csv (admissions)...")
    hesin = pd.read_csv(HESIN_CSV, usecols=['dnx_hesin_id', 'eid', 'admidate'],
                       dtype={'eid': 'Int64'})
    print(f"    {len(hesin)} admissions")
    hesin['admidate'] = pd.to_datetime(hesin['admidate'], errors='coerce')
    hesin = hesin.dropna(subset=['admidate'])

    print("  Reading hesin_diag.csv (diagnoses)...")
    diag = pd.read_csv(HESIN_DIAG_CSV, usecols=['dnx_hesin_id', 'diag_icd10'])
    print(f"    {len(diag)} diagnoses total")

    # Filter to dementia ICD-10
    diag['is_dem'] = diag['diag_icd10'].apply(is_dementia_icd10)
    diag_dem = diag[diag['is_dem']].copy()
    print(f"    {len(diag_dem)} dementia diagnoses (F00*, F01*, F02*, F03, G30*)")

    # Join with hesin to get admidate
    merged = diag_dem.merge(hesin, on='dnx_hesin_id', how='inner')
    print(f"    {len(merged)} matched to admissions")

    # For each patient, take earliest dementia admidate
    earliest = merged.groupby('eid')['admidate'].min()
    print(f"    {len(earliest)} unique patients with HES dementia")

    lookup = {int(pid): dt.to_pydatetime() for pid, dt in earliest.items() if pd.notnull(dt)}
    with open(OUT_LOOKUP_PICKLE, 'wb') as f:
        pickle.dump(lookup, f)
    print(f"    Cached to {OUT_LOOKUP_PICKLE}")
    return lookup


def main():
    print("=" * 70)
    print("V6 STEP 1 v3: V1-Type-C lag via HES lookup")
    print("=" * 70)

    print("\n[1/3] Building HES dementia lookup...")
    hes_lookup = build_hes_dementia_lookup()
    print(f"  Patients with HES dementia: {len(hes_lookup)}")

    print("\n[2/3] Loading V1 (hes_aug) train parquet, identifying V1-Type-C...")
    v1_files = sorted(glob.glob(V1_TRAIN_GLOB, recursive=True))
    print(f"  V1 files: {len(v1_files)}")

    type_C_count = 0
    lags = []
    eu02z_patients_total = 0
    eu02z_with_hes_match = 0
    eu02z_date_mismatch = 0
    second_to_last_dementia = 0
    second_to_last_too_close = 0

    for fp in tqdm(v1_files, desc='V1', leave=False):
        df = pq.read_table(fp).to_pandas()
        for _, row in df.iterrows():
            pid = int(row['PATIENT_ID'])
            ev = list(row['EVENT'])
            dt = list(row['DATE'])
            if len(ev) < 2:
                continue
            last_event = str(ev[-1])
            if last_event != HES_LABEL_CODE:
                continue
            eu02z_patients_total += 1

            # Check HES lookup
            if pid not in hes_lookup:
                continue
            eu02z_with_hes_match += 1

            # Check date matches HES lookup date (V1 set last_date = HES date)
            v1_last_date = pd.Timestamp(dt[-1]).normalize().to_pydatetime()
            hes_date = hes_lookup[pid]
            if hasattr(hes_date, 'date'):
                hes_date_norm = pd.Timestamp(hes_date).normalize().to_pydatetime()
            else:
                hes_date_norm = hes_date
            if abs((v1_last_date - hes_date_norm).days) > 1:  # allow 1 day tolerance
                eu02z_date_mismatch += 1
                continue

            # Check second-to-last is NOT a dementia code (Type C means originally censored)
            second_to_last = str(ev[-2])
            if second_to_last in DEMENTIA_READ_CODES_SET or second_to_last == 'DEATH':
                second_to_last_dementia += 1
                continue

            # Compute lag = HES date - second_to_last (last actual GP event)
            v1_2nd_last_date = pd.Timestamp(dt[-2]).normalize().to_pydatetime()
            lag_days = (v1_last_date - v1_2nd_last_date).days
            lag_years = lag_days / 365.25
            if lag_years <= 0:
                continue
            if lag_years < 0.003:  # ~1 day, suspicious (might be same-day cluster)
                second_to_last_too_close += 1
                continue

            lags.append(lag_years)
            type_C_count += 1

    print(f"\n  V1 patients with last_event=Eu02z: {eu02z_patients_total}")
    print(f"  ... and in HES dementia lookup:      {eu02z_with_hes_match}")
    print(f"  ... and date matches HES date:        {eu02z_with_hes_match - eu02z_date_mismatch}")
    print(f"  ... and 2nd-last is censored event:   {eu02z_with_hes_match - eu02z_date_mismatch - second_to_last_dementia}")
    print(f"  ... and 2nd-last date < HES date:     {type_C_count}")
    print(f"  Excluded: date mismatch={eu02z_date_mismatch}, 2nd-last dementia/death={second_to_last_dementia}, too-close 2nd-last={second_to_last_too_close}")

    lags = np.array(lags, dtype=np.float64)
    print(f"\n[3/3] Lag stats (n={len(lags)} V1-Type-C patients):")
    pcts = np.percentile(lags, [5, 25, 50, 75, 95])
    stats = {
        'source': 'V1-Type-C (alive censored → V1 relabeled via HES, identified via HES lookup match)',
        'n_typeC': int(type_C_count),
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
            print(f"  {k:22s} = {v:.4f}")
        else:
            print(f"  {k:22s} = {v}")

    np.save(OUT_NPY, lags)
    with open(OUT_JSON, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Saved to {OUT_NPY} and {OUT_JSON}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(lags, bins=80, edgecolor='black')
    axes[0].axvline(np.median(lags), color='red', linestyle='--', label=f'Median = {np.median(lags):.2f}y')
    axes[0].axvline(np.mean(lags), color='orange', linestyle='--', label=f'Mean = {np.mean(lags):.2f}y')
    axes[0].axvline(5.0, color='gray', linestyle=':', label='5y model horizon')
    axes[0].set_xlabel('Lag (years): HES dementia date - last GP event date')
    axes[0].set_ylabel('Number of V1-Type-C patients')
    axes[0].set_title(f'V1-Type-C empirical lag (n={len(lags)})')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(lags, bins=80, edgecolor='black', cumulative=True, density=True)
    axes[1].axhline(0.5, color='red', linestyle='--', label='50%')
    axes[1].axvline(5.0, color='gray', linestyle=':', label='5y horizon')
    axes[1].set_xlabel('Lag (years)')
    axes[1].set_ylabel('Cumulative fraction')
    axes[1].set_title('Cumulative distribution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=100)
    print(f"  Saved histogram to {OUT_PNG}")

    # Sanity
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)
    checks = [
        ('Type C count in [2500, 5500] (V1 doc says ~4097 across train/val/test, train ~90%)', 2500 <= type_C_count <= 5500),
        ('Median lag in [0.5, 5] years', 0.5 < stats['median_years'] < 5.0),
        ('Max lag < 15 years', stats['max_years'] < 15),
        ('All positive', np.all(lags > 0)),
        ('Mean > median (right-skewed expected)', stats['mean_years'] > stats['median_years']),
    ]
    passed = 0
    for desc, p in checks:
        s = "✓" if p else "✗"
        print(f"  [{s}] {desc}")
        if p: passed += 1
    print(f"\n  {passed}/{len(checks)} checks passed.")
    return stats


if __name__ == '__main__':
    main()
