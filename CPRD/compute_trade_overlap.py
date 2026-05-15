"""
Compute (1) how many extra dementia patients we'd capture by adding G31.0/G31.8
(FTD + Lewy body) HES ICD-10 codes to our V2 label, and
(2) the overlap between our V2 label and TRADE's AD/ADRD/MCI label.

Strategy:
  HES side only (ICD-10), because Read V2 GP mapping to G31.* is fuzzy and
  HES is the side we can compare directly with TRADE.

Approach:
  - Read hesin_diag.csv, filter to dementia-related ICD-10
  - Bucket each patient into: AD/F00, F01, F02, F03, G30, G31.0 (Pick's/FTD),
    G31.8 (Lewy body), G31.84/G31.85 (not in WHO), F04 (amnestic), G23.1 (PSP),
    G31.1, G31.9
  - Join with V2 test set (8241 patients) to see overlap
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path('/Data0/swangek_data/991/CPRD/data')
HESIN_DIAG_CSV = DATA_DIR / 'hesin_diag.csv'

# Our V2 HES dementia code prefixes
OUR_HES_PREFIXES = ('F00', 'F01', 'F02', 'G30')
OUR_HES_EXACT = {'F03'}

# Buckets we want to inspect
BUCKETS = {
    'AD/F00 (we use)':        lambda c: c.startswith('F00'),
    'F01 (Vascular, we use)': lambda c: c.startswith('F01'),
    'F02 (Other, we use)':    lambda c: c.startswith('F02'),
    'F03 (Unspecified, we use)': lambda c: c.startswith('F03'),
    'G30 (AD, we use)':       lambda c: c.startswith('G30'),
    # Below = NOT in our V2 HES list
    'G31.0 (Pick/FTD, missed)':  lambda c: c == 'G310' or c.startswith('G310'),
    'G31.8 (Lewy etc., missed)': lambda c: c == 'G318' or c.startswith('G318'),
    'G31.1 (Senile degen NEC, missed)': lambda c: c == 'G311',
    'G31.9 (Degen NS unspec, missed)':  lambda c: c == 'G319',
    'F04 (Amnestic, missed)': lambda c: c.startswith('F04'),
    'G23.1 (PSP, missed)':    lambda c: c == 'G231',
}

# Load V2 test PIDs
v2 = np.load(DATA_DIR / 'test_cif_v2_full.npz', allow_pickle=True)
v2_test_pids = set(int(p) for p in v2['patient_ids'])
v2_labels = v2['labels']
v2_pid_array = np.array([int(p) for p in v2['patient_ids']])
v2_dementia_pids = set(v2_pid_array[v2_labels == 'dementia'].tolist())
print(f"V2 test set: {len(v2_test_pids)} patients, "
      f"{len(v2_dementia_pids)} dementia events")

# Load full HES diagnosis CSV
print(f"\nReading {HESIN_DIAG_CSV} ...")
# Columns: dnx_hesin_diag_id, dnx_hesin_id, eid, ins_index, arr_index, level,
#          diag_icd9, diag_icd9_nb, diag_icd10, diag_icd10_nb
diag = pd.read_csv(HESIN_DIAG_CSV, usecols=['eid', 'diag_icd10'])
print(f"  Total HES diagnosis rows: {len(diag):,}")
print(f"  Unique patients: {diag['eid'].nunique():,}")

# Normalize ICD-10: strip dots, uppercase
diag['icd10_clean'] = diag['diag_icd10'].fillna('').astype(str).str.upper().str.replace('.', '', regex=False)
diag = diag[diag['icd10_clean'] != '']

# Bucket each row
print(f"\nBucket counts (patient-level unique counts):")
print(f"  {'Bucket':<45} {'#diag rows':>12} {'#unique pts':>12} {'#in V2 test':>12} {'#V2 test new positives':>22}")
print(f"  {'-'*45} {'-'*12} {'-'*12} {'-'*12} {'-'*22}")

bucket_results = {}
for name, pred in BUCKETS.items():
    mask = diag['icd10_clean'].apply(pred)
    rows = mask.sum()
    pts = diag.loc[mask, 'eid'].nunique()
    # Patients in V2 test set with this code
    pids_in_v2 = set(diag.loc[mask, 'eid'].unique().astype(int).tolist()) & v2_test_pids
    # Patients in V2 test set who are NOT already labeled dementia
    new_pos = pids_in_v2 - v2_dementia_pids
    bucket_results[name] = {
        'rows': int(rows),
        'pts': int(pts),
        'pids_in_v2': pids_in_v2,
        'new_pos_in_v2': new_pos,
    }
    print(f"  {name:<45} {rows:>12,} {pts:>12,} {len(pids_in_v2):>12,} {len(new_pos):>22,}")

# Cumulative new positives from "missed" categories
missed_keys = [
    'G31.0 (Pick/FTD, missed)',
    'G31.8 (Lewy etc., missed)',
    'G31.1 (Senile degen NEC, missed)',
    'G31.9 (Degen NS unspec, missed)',
    'F04 (Amnestic, missed)',
    'G23.1 (PSP, missed)',
]
new_pos_union = set()
for k in missed_keys:
    new_pos_union |= bucket_results[k]['new_pos_in_v2']

print()
print("="*80)
print("Q1: How many EXTRA V2-test dementia positives if we add missed codes?")
print("="*80)
print(f"  Current V2 test dementia: {len(v2_dementia_pids)}")
print(f"  Union of all 'missed' code patients in V2 test (excl. already dementia): {len(new_pos_union)}")
print(f"  By specific subset (strict FTD+Lewy only):")
ftd_lewy_union = bucket_results['G31.0 (Pick/FTD, missed)']['new_pos_in_v2'] | \
                  bucket_results['G31.8 (Lewy etc., missed)']['new_pos_in_v2']
print(f"    G31.0 + G31.8 only: {len(ftd_lewy_union)}")
print(f"  Hypothetical new dementia count if we added FTD+Lewy: {len(v2_dementia_pids) + len(ftd_lewy_union)}")
print(f"  Hypothetical new dementia count if we added ALL missed: {len(v2_dementia_pids) + len(new_pos_union)}")

print()
print("="*80)
print("Q2: Overlap with TRADE's AD/ADRD/MCI label scope")
print("="*80)

# TRADE's AD/ADRD codes = F01* + F02* + F03* + F04* + G23.1 + G30* + G31.01 + G31.09 + G31.83 + G31.9
# TRADE's MCI codes = G31.1 + G31.84 + G31.85
# We map WHO equivalents:
#   G31.01 -> G310 (Pick's, falls into G31.0 in WHO)
#   G31.09 -> falls into G310/G318 in WHO
#   G31.83 -> falls into G318 in WHO (Lewy body)
#   G31.84 -> NO equivalent in WHO (UK doesn't have G31.84 specifically)
#   G31.85 -> falls into G318 (Corticobasal degeneration)

trade_adrd_codes_who = ['F01', 'F02', 'F03', 'F04', 'G231', 'G30', 'G310', 'G318', 'G319']
trade_mci_codes_who  = ['G311', 'G318']   # G31.84/85 fold into G31.8; G31.1 is its own code

def patient_has_any(prefix_list, exact_list=None):
    """Return set of eid having any matching ICD-10 code."""
    exact_list = exact_list or []
    m = diag['icd10_clean'].apply(
        lambda c: any(c.startswith(p) for p in prefix_list) or c in exact_list
    )
    return set(diag.loc[m, 'eid'].astype(int).tolist())

# Patients with any TRADE AD/ADRD code (in HES)
trade_adrd_pts = patient_has_any(['F01', 'F02', 'F03', 'F04', 'G231', 'G30', 'G310', 'G318', 'G319'])
# Patients with any TRADE MCI code (WHO mapping, partial)
trade_mci_pts = patient_has_any(['G311', 'G318'])
trade_all_pts = trade_adrd_pts | trade_mci_pts

# Patients with any OF OUR V2 HES code
our_v2_hes_pts = patient_has_any(['F00', 'F01', 'F02', 'G30'], ['F03'])

# All restricted to V2 test set
our_v2_hes_in_test = our_v2_hes_pts & v2_test_pids
trade_all_in_test = trade_all_pts & v2_test_pids
trade_adrd_in_test = trade_adrd_pts & v2_test_pids
trade_mci_in_test = trade_mci_pts & v2_test_pids

intersection = our_v2_hes_in_test & trade_all_in_test
us_only = our_v2_hes_in_test - trade_all_in_test
trade_only = trade_all_in_test - our_v2_hes_in_test

print(f"  (V2 test set N={len(v2_test_pids)})")
print(f"  Our V2 HES dementia patients (test):       {len(our_v2_hes_in_test):>6}")
print(f"  TRADE AD/ADRD patients (test, WHO map):    {len(trade_adrd_in_test):>6}")
print(f"  TRADE MCI patients (test, WHO map):        {len(trade_mci_in_test):>6}")
print(f"  TRADE all (AD/ADRD/MCI) patients (test):   {len(trade_all_in_test):>6}")
print()
print(f"  Intersection (us ∩ TRADE):                 {len(intersection):>6}")
print(f"  Only ours (not in TRADE):                  {len(us_only):>6}")
print(f"  Only TRADE (not in ours):                  {len(trade_only):>6}")
if len(our_v2_hes_in_test) > 0:
    print(f"  Coverage of us within TRADE: {len(intersection)/len(our_v2_hes_in_test)*100:.1f}%")
if len(trade_all_in_test) > 0:
    print(f"  Coverage of TRADE within us: {len(intersection)/len(trade_all_in_test)*100:.1f}%")

# Also: TRADE has medications. We didn't include those.
# But here we only compare HES side.
print()
print("CAVEAT: This is HES-only comparison. Our V2 label is GP+HES union, so the")
print("        actual V2 dementia count uses GP Read codes too (Eu0../E00../F110.).")
print("        Many V2 dementia patients are GP-only triggered and would not appear")
print("        in HES at all -- but TRADE doesn't have GP data anyway, so for an")
print("        ICD-10-only apples-to-apples comparison the HES view is the right one.")
print()
print("        Also: TRADE additionally uses dementia medications (Donepezil etc.)")
print("        as a label trigger. Not modeled here. Would only INCREASE TRADE's set.")
