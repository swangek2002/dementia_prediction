"""
Finest-grained breakdown of the ICD-10 codes that TRADE includes but our V2 doesn't.

For each "missed" code category (G31.x, F04.x, G23.1), list every distinct ICD-10
sub-code appearing in our cohort's HES data, with patient counts.
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import glob
from pathlib import Path

DATA_DIR = Path('/Data0/swangek_data/991/CPRD/data')
V2_DS = DATA_DIR / 'FoundationalModel' / 'FineTune_Dementia_CR_hes_static_v2'
HESIN_DIAG_CSV = DATA_DIR / 'hesin_diag.csv'

# Load cohort PIDs
print("Loading cohort PIDs...")
cohort_pids = set()
for split in ['train', 'val', 'test']:
    pq_files = sorted(glob.glob(str(V2_DS / f'split={split}' / '**' / '*.parquet'), recursive=True))
    for fp in pq_files:
        t = pq.read_table(fp, columns=['PATIENT_ID'])
        cohort_pids.update(t['PATIENT_ID'].to_pylist())
print(f"  Cohort: {len(cohort_pids):,} patients")

print(f"\nReading HES diag...")
diag = pd.read_csv(HESIN_DIAG_CSV, usecols=['eid', 'diag_icd10'])
diag['icd10_clean'] = diag['diag_icd10'].fillna('').astype(str).str.upper().str.replace('.', '', regex=False)
diag = diag[diag['icd10_clean'] != '']
diag = diag[diag['eid'].isin(cohort_pids)].copy()
print(f"  Cohort HES rows: {len(diag):,}")

# ---- Categories we want to break down ----
# 'missed' from TRADE perspective + show what we ALREADY use for reference
CATEGORIES = {
    "OUR ICD-10 CODES (already used in V2 label)": {
        'F00 (Dementia in Alzheimer\'s)': lambda c: c.startswith('F00'),
        'F01 (Vascular dementia)':        lambda c: c.startswith('F01'),
        'F02 (Dementia in other diseases)': lambda c: c.startswith('F02'),
        'F03 (Unspecified dementia)':     lambda c: c.startswith('F03'),
        'G30 (Alzheimer\'s disease)':     lambda c: c.startswith('G30'),
    },
    "MISSED: AD/ADRD codes TRADE has, we don't": {
        'F04 (Amnestic disorder)':              lambda c: c.startswith('F04'),
        'G23.1 (Progressive supranuclear palsy)': lambda c: c == 'G231',
        'G31.0 (Pick\'s / FTD)':                lambda c: c.startswith('G310'),
        'G31.8 (Lewy body / CBD / other)':      lambda c: c.startswith('G318'),
        'G31.9 (Degenerative NS unspec)':       lambda c: c == 'G319',
    },
    "MISSED: MCI codes TRADE has, we don't": {
        'G31.1 (Senile degen NEC)': lambda c: c == 'G311',
    },
}

# Code -> description (best-effort, WHO ICD-10 / ICD-10-CM)
DESC = {
    'F00':   'Dementia in Alzheimer\'s disease (umbrella)',
    'F000':  'Dementia in Alzheimer\'s disease, early onset (F00.0)',
    'F001':  'Dementia in Alzheimer\'s disease, late onset (F00.1)',
    'F002':  'Dementia in Alzheimer\'s disease, atypical/mixed type (F00.2)',
    'F009':  'Dementia in Alzheimer\'s disease, unspecified (F00.9)',
    'F010':  'Vascular dementia of acute onset (F01.0)',
    'F011':  'Multi-infarct dementia (F01.1)',
    'F012':  'Subcortical vascular dementia (F01.2)',
    'F013':  'Mixed cortical/subcortical vasc. dementia (F01.3)',
    'F018':  'Other vascular dementia (F01.8)',
    'F019':  'Vascular dementia, unspecified (F01.9)',
    'F020':  'Dementia in Pick\'s disease (F02.0)',
    'F021':  'Dementia in Creutzfeldt-Jakob disease (F02.1)',
    'F022':  'Dementia in Huntington\'s disease (F02.2)',
    'F023':  'Dementia in Parkinson\'s disease (F02.3)',
    'F024':  'Dementia in HIV disease (F02.4)',
    'F028':  'Dementia in other specified diseases (F02.8)',
    'F03':   'Unspecified dementia (F03)',
    'F030':  'Unspecified dementia, no behavioural disturbance (F03.90)',
    'F0390': 'Unspecified dementia w/o behavioural disturbance (F03.90)',
    'F0391': 'Unspecified dementia with behavioural disturbance (F03.91)',
    'F04':   'Organic amnesic syndrome (F04)',
    'F040':  '— (F04.0)',
    'G231':  'Progressive supranuclear palsy [Steele-Richardson-Olszewski] (G23.1)',
    'G300':  'Alzheimer\'s disease with early onset (G30.0)',
    'G301':  'Alzheimer\'s disease with late onset (G30.1)',
    'G308':  'Other Alzheimer\'s disease (G30.8)',
    'G309':  'Alzheimer\'s disease, unspecified (G30.9)',
    'G310':  'Circumscribed brain atrophy / Pick\'s disease (G31.0)',
    'G3101': 'Pick\'s disease (G31.01 — ICD-10-CM US)',
    'G3109': 'Other frontotemporal dementia (G31.09 — ICD-10-CM US)',
    'G311':  'Senile degeneration of brain, NEC (G31.1)',
    'G318':  'Other specified degenerative diseases of nervous system (G31.8)',
    'G3183': 'Dementia with Lewy bodies (G31.83 — ICD-10-CM US)',
    'G3184': 'Mild cognitive impairment, so stated (G31.84 — ICD-10-CM US)',
    'G3185': 'Corticobasal degeneration (G31.85 — ICD-10-CM US)',
    'G319':  'Degenerative disease of nervous system, unspecified (G31.9)',
}

def description(code):
    return DESC.get(code, '(no description)')

# Run breakdown
N = len(cohort_pids)
print(f"\nN cohort = {N:,}")

for cat_title, buckets in CATEGORIES.items():
    print()
    print("="*88)
    print(cat_title)
    print("="*88)
    for label, pred in buckets.items():
        m = diag['icd10_clean'].apply(pred)
        sub = diag[m]
        if len(sub) == 0:
            print(f"\n  {label}: NO PATIENTS IN COHORT")
            continue
        print(f"\n  {label}")
        # Show every distinct full code under this bucket, with patient count
        code_pts = sub.groupby('icd10_clean')['eid'].nunique().sort_values(ascending=False)
        code_rows = sub.groupby('icd10_clean').size()
        print(f"    {'ICD code':<10} {'#patients':>10} {'%cohort':>9} {'#diag rows':>11}  Description")
        print(f"    {'-'*10} {'-'*10} {'-'*9} {'-'*11}  {'-'*60}")
        total_pts_unique = sub['eid'].nunique()
        for code, npts in code_pts.items():
            nrows = int(code_rows.loc[code])
            d = description(code)
            print(f"    {code:<10} {npts:>10,} {npts/N*100:>8.3f}% {nrows:>11,}  {d}")
        print(f"    {'TOTAL':<10} {total_pts_unique:>10,} {total_pts_unique/N*100:>8.3f}%  (unique patients)")

print()
print("="*88)
print("Notes")
print("="*88)
print("""
- ICD-10 in HES (UK) uses WHO ICD-10 (4-char), e.g. F00.0, F03 (no subdivisions),
  G31.0, G31.8. There is NO G31.83/G31.84/G31.85 in WHO ICD-10 — those are
  ICD-10-CM (US) extensions used by TRADE.
- G31.0 in WHO = Pick's disease + Circumscribed brain atrophy (TRADE's G31.01 + G31.09 fold here)
- G31.8 in WHO = "Other specified degenerative diseases" — includes Lewy body
  (TRADE's G31.83), Corticobasal degeneration (TRADE's G31.85), and others.
  We cannot disambiguate Lewy body from non-dementia G31.8 codes within WHO.
- The patient counts above show how many UNIQUE patients in our 133,322 cohort
  have at least one occurrence of each code in their HES history.
""")
