"""
Count patients per GP Read code (the 41 dementia label codes) across the V2 cohort.
Reads from V2 parquet files directly.
"""
import pyarrow.parquet as pq
import glob
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path('/Data0/swangek_data/991/CPRD/data')
V2_DS = DATA_DIR / 'FoundationalModel' / 'FineTune_Dementia_CR_hes_static_v2'

DEMENTIA_READ_CODES = [
    "F110.", "Eu00.", "Eu01.", "Eu02z", "Eu002", "E00..", "Eu023", "Eu00z",
    "Eu025", "Eu01z", "E001.", "F1100", "Eu001", "E004.", "Eu000", "Eu02.",
    "Eu013", "E000.", "Eu01y", "E001z", "F1101", "Eu020", "E004z", "E0021",
    "Eu02y", "Eu012", "Eu011", "E00z.", "E0040", "E003.", "E0020",
    "Eu057", "Eu04.", "Eu053", "Eu04z", "Eu0z.", "Eu060", "Eu054", "Eu05y",
    "Eu052", "Eu062",
]
CODE_DESC = {
    "F110.":"Alzheimer's disease (neurology)", "F1100":"Alzheimer's, early onset",
    "F1101":"Alzheimer's, late onset",
    "Eu00.":"[X]Dementia in AD (umbrella)", "Eu000":"Dementia in AD, early onset (Type 2)",
    "Eu001":"Dementia in AD, late onset (Type 1)", "Eu002":"Dementia in AD, atypical/mixed",
    "Eu00z":"Dementia in AD, unspecified",
    "Eu01.":"Vascular dementia (umbrella)", "Eu011":"Multi-infarct dementia",
    "Eu012":"Subcortical vascular dementia", "Eu013":"Mixed cortical/subcortical vasc",
    "Eu01y":"Other vascular dementia", "Eu01z":"Vascular dementia, unspecified",
    "Eu02.":"Dementia in other diseases (umbrella)", "Eu020":"Dementia in Pick's disease (FTD)",
    "Eu023":"Dementia in Parkinson's", "Eu025":"Dementia in other specified diseases",
    "Eu02y":"Dementia in other specified", "Eu02z":"Unspecified dementia",
    "E00..":"Senile/presenile organic psychotic conditions (umbrella)",
    "E000.":"Uncomplicated senile dementia", "E001.":"Presenile dementia",
    "E001z":"Presenile dementia NOS", "E0020":"Senile dem w/ depressive/paranoid features",
    "E0021":"Senile dem w/ paranoia", "E003.":"Senile dem w/ delirium",
    "E004.":"Arteriosclerotic dementia", "E004z":"Arteriosclerotic dem NOS",
    "E0040":"Uncomplicated arteriosclerotic dem", "E00z.":"Senile/presenile psychoses NOS",
    "Eu057":"Mild cognitive disorder (closest to MCI)",
    "Eu053":"Organic mood (affective) disorders",
    "Eu054":"Organic anxiety disorder", "Eu052":"Organic delusional disorder",
    "Eu05y":"Other organic mental disorders",
    "Eu04.":"Delirium (F05) — umbrella", "Eu04z":"Delirium, unspecified",
    "Eu060":"Organic personality disorder", "Eu062":"Postconcussional syndrome",
    "Eu0z.":"Unspecified organic mental disorder",
}

code_set = set(DEMENTIA_READ_CODES)
code_pids = defaultdict(set)
all_pids = set()

for split in ['train', 'val', 'test']:
    files = sorted(glob.glob(str(V2_DS / f'split={split}' / '**' / '*.parquet'), recursive=True))
    print(f"[{split}] scanning {len(files)} parquet files...")
    for fp in files:
        t = pq.read_table(fp, columns=['PATIENT_ID', 'EVENT'])
        pids = t['PATIENT_ID'].to_pylist()
        events = t['EVENT'].to_pylist()  # list of lists
        for pid, ev in zip(pids, events):
            all_pids.add(pid)
            if ev is None:
                continue
            seen_codes = set(ev) & code_set
            for c in seen_codes:
                code_pids[c].add(pid)

N = len(all_pids)
print(f"\nCohort N = {N:,}")
print()
print(f"{'Read Code':<8} {'#patients':>10} {'%cohort':>9}  Description")
print(f"{'-'*8} {'-'*10} {'-'*9}  {'-'*60}")
items = [(c, len(code_pids.get(c, set()))) for c in DEMENTIA_READ_CODES]
items.sort(key=lambda x: -x[1])
union_pids = set()
for c, n in items:
    pct = n/N*100
    desc = CODE_DESC.get(c, '(no desc)')
    print(f"{c:<8} {n:>10,} {pct:>8.3f}%  {desc}")
    union_pids |= code_pids.get(c, set())
print()
print(f"Union (any GP code in cohort): {len(union_pids):,} patients ({len(union_pids)/N*100:.3f}%)")
