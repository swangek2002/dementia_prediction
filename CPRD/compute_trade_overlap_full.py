"""
Same overlap analysis as compute_trade_overlap.py, but on the FULL V2 cohort
(train + val + test = ~133K patients), not just the test set.

Loads PIDs from V2 parquet files for all 3 splits.
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import glob
from pathlib import Path

DATA_DIR = Path('/Data0/swangek_data/991/CPRD/data')
V2_DS = DATA_DIR / 'FoundationalModel' / 'FineTune_Dementia_CR_hes_static_v2'
HESIN_DIAG_CSV = DATA_DIR / 'hesin_diag.csv'

# 1. Gather all PIDs across train/val/test
print("Loading V2 cohort PIDs (all 3 splits)...")
cohort_pids = set()
split_pids = {}
for split in ['train', 'val', 'test']:
    pq_files = sorted(glob.glob(str(V2_DS / f'split={split}' / '**' / '*.parquet'), recursive=True))
    pids = set()
    for fp in pq_files:
        t = pq.read_table(fp, columns=['PATIENT_ID'])
        pids.update(t['PATIENT_ID'].to_pylist())
    split_pids[split] = pids
    cohort_pids.update(pids)
    print(f"  {split}: {len(pids):,} patients")
print(f"  Total cohort: {len(cohort_pids):,}")

# 2. Load HES diag
print(f"\nReading {HESIN_DIAG_CSV} (chunked)...")
diag = pd.read_csv(HESIN_DIAG_CSV, usecols=['eid', 'diag_icd10'])
print(f"  Total HES rows: {len(diag):,}, unique patients: {diag['eid'].nunique():,}")

diag['icd10_clean'] = diag['diag_icd10'].fillna('').astype(str).str.upper().str.replace('.', '', regex=False)
diag = diag[diag['icd10_clean'] != '']
print(f"  After dropping empty: {len(diag):,} rows")

# Filter HES to only cohort patients (huge speedup)
diag_cohort = diag[diag['eid'].isin(cohort_pids)].copy()
print(f"  Cohort-only HES rows: {len(diag_cohort):,}, "
      f"unique cohort pts in HES: {diag_cohort['eid'].nunique():,}")

# 3. Bucket counts at cohort level
BUCKETS = [
    ('F00 AD (we use)',                lambda c: c.startswith('F00')),
    ('F01 Vascular (we use)',          lambda c: c.startswith('F01')),
    ('F02 Other (we use)',             lambda c: c.startswith('F02')),
    ('F03 Unspecified (we use)',       lambda c: c.startswith('F03')),
    ('G30 AD neuro (we use)',          lambda c: c.startswith('G30')),
    ('G31.0 Pick/FTD (missed)',        lambda c: c.startswith('G310')),
    ('G31.8 Lewy etc. (missed)',       lambda c: c.startswith('G318')),
    ('G31.1 Senile degen NEC (missed)', lambda c: c == 'G311'),
    ('G31.9 Degen NS unspec (missed)', lambda c: c == 'G319'),
    ('F04 Amnestic (missed)',          lambda c: c.startswith('F04')),
    ('G23.1 PSP (missed)',             lambda c: c == 'G231'),
]

N = len(cohort_pids)
print(f"\nFull-cohort patient counts (N={N:,})\n")
print(f"  {'Bucket':<35} {'#pts in cohort':>15} {'% of cohort':>14}")
print(f"  {'-'*35} {'-'*15} {'-'*14}")

bucket_results = {}
for name, pred in BUCKETS:
    m = diag_cohort['icd10_clean'].apply(pred)
    pids = set(diag_cohort.loc[m, 'eid'].astype(int).tolist())
    bucket_results[name] = pids
    print(f"  {name:<35} {len(pids):>15,} {len(pids)/N*100:>13.3f}%")

# 4. Our V2 HES set = F00 ∪ F01 ∪ F02 ∪ F03 ∪ G30
our_v2_hes = set()
for name in ['F00 AD (we use)', 'F01 Vascular (we use)', 'F02 Other (we use)',
             'F03 Unspecified (we use)', 'G30 AD neuro (we use)']:
    our_v2_hes |= bucket_results[name]

# 5. TRADE's full umbrella (WHO mapping)
#    AD/ADRD: F01, F02, F03, F04, G231, G30, G310, G318, G319
#    MCI:     G311, G318 (G31.84/85 fold into G318)
trade_full = set()
for name in ['F00 AD (we use)', 'F01 Vascular (we use)', 'F02 Other (we use)',
             'F03 Unspecified (we use)', 'F04 Amnestic (missed)',
             'G23.1 PSP (missed)', 'G30 AD neuro (we use)',
             'G31.0 Pick/FTD (missed)', 'G31.8 Lewy etc. (missed)',
             'G31.9 Degen NS unspec (missed)',
             'G31.1 Senile degen NEC (missed)']:
    trade_full |= bucket_results[name]

# 6. Strict TRADE umbrella WITHOUT debatable codes (G31.9, F04, G23.1)
trade_strict = set()
for name in ['F00 AD (we use)', 'F01 Vascular (we use)', 'F02 Other (we use)',
             'F03 Unspecified (we use)', 'G30 AD neuro (we use)',
             'G31.0 Pick/FTD (missed)', 'G31.8 Lewy etc. (missed)',
             'G31.1 Senile degen NEC (missed)']:
    trade_strict |= bucket_results[name]

print()
print("="*80)
print("OVERLAP ANALYSIS (full V2 cohort)")
print("="*80)

def show_overlap(label_a, set_a, label_b, set_b):
    inter = set_a & set_b
    a_only = set_a - set_b
    b_only = set_b - set_a
    print(f"  {label_a}: {len(set_a):,} pts ({len(set_a)/N*100:.3f}%)")
    print(f"  {label_b}: {len(set_b):,} pts ({len(set_b)/N*100:.3f}%)")
    print(f"  Intersection:       {len(inter):,}")
    print(f"  Only in {label_a}: {len(a_only):,}")
    print(f"  Only in {label_b}: {len(b_only):,}")
    if len(set_a) > 0:
        print(f"  {label_a} coverage by {label_b}: {len(inter)/len(set_a)*100:.2f}%")
    if len(set_b) > 0:
        print(f"  {label_b} coverage by {label_a}: {len(inter)/len(set_b)*100:.2f}%")

print()
print("[A] Us (V2 HES) vs TRADE-full (with G31.9 etc.):")
show_overlap('Us       ', our_v2_hes, 'TRADE-full', trade_full)
print()
print("[B] Us (V2 HES) vs TRADE-strict (NO G31.9/F04/G23.1):")
show_overlap('Us         ', our_v2_hes, 'TRADE-strict', trade_strict)

# 7. Hypothetical: us + FTD/Lewy补码
us_plus_ftd_lewy = our_v2_hes | bucket_results['G31.0 Pick/FTD (missed)'] | bucket_results['G31.8 Lewy etc. (missed)']
print()
print("="*80)
print("Hypothetical: what if we ADD FTD + Lewy body to our HES list?")
print("="*80)
print(f"  Our current V2 HES patients in cohort:  {len(our_v2_hes):,} ({len(our_v2_hes)/N*100:.3f}%)")
print(f"  +G31.0 (Pick/FTD):                       +{len(bucket_results['G31.0 Pick/FTD (missed)'] - our_v2_hes):,}")
print(f"  +G31.8 (Lewy etc.):                      +{len(bucket_results['G31.8 Lewy etc. (missed)'] - our_v2_hes):,}")
print(f"  Combined new patients:                   +{len(us_plus_ftd_lewy - our_v2_hes):,}")
print(f"  After adding FTD+Lewy:                    {len(us_plus_ftd_lewy):,} ({len(us_plus_ftd_lewy)/N*100:.3f}%)")

# 8. Hypothetical: us + ALL missed
us_plus_all = our_v2_hes.copy()
for name in ['G31.0 Pick/FTD (missed)', 'G31.8 Lewy etc. (missed)',
             'G31.1 Senile degen NEC (missed)', 'G31.9 Degen NS unspec (missed)',
             'F04 Amnestic (missed)', 'G23.1 PSP (missed)']:
    us_plus_all |= bucket_results[name]
print(f"\n  After adding ALL missed:                  {len(us_plus_all):,} ({len(us_plus_all)/N*100:.3f}%)")
print(f"    of which new (not in current):         +{len(us_plus_all - our_v2_hes):,}")

print()
print("="*80)
print("Bottom line (full cohort scale)")
print("="*80)
print(f"  Cohort size:                              {N:,}")
print(f"  Current V2 HES dementia patients:         {len(our_v2_hes):,}  ({len(our_v2_hes)/N*100:.2f}%)")
print(f"  TRADE-full (incl debatable):              {len(trade_full):,}  ({len(trade_full)/N*100:.2f}%)")
print(f"  TRADE-strict (no debatable):              {len(trade_strict):,}  ({len(trade_strict)/N*100:.2f}%)")
print(f"  Gap us vs TRADE-full:                     -{len(trade_full - our_v2_hes):,} pts")
print(f"  Gap us vs TRADE-strict:                   -{len(trade_strict - our_v2_hes):,} pts")
