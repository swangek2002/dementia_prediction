"""
verify_hes_fusion_dataset.py
============================
Statistics for the HES-fused dataset:
  - per-split totals + k=1/k=2/k=0 counts
  - average sequence length per split
  - event-vocab leakage check (no ICD-10 looking codes in EVENT)
"""
import os
import re

import numpy as np
import pyarrow.parquet as pq

DEMENTIA_CODES = set([
    'F110.', 'Eu00.', 'Eu01.', 'Eu02z', 'Eu002', 'E00..', 'Eu023', 'Eu00z', 'Eu025',
    'Eu01z', 'E001.', 'F1100', 'Eu001', 'E004.', 'Eu000', 'Eu02.', 'Eu013', 'E000.',
    'Eu01y', 'E001z', 'F1101', 'Eu020', 'E004z', 'E0021', 'Eu02y', 'Eu012', 'Eu011',
    'E00z.', 'E0040', 'E003.', 'E0020',
])
DEATH_CODES = set(['DEATH'])

HES_BASE = '/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_fusion/'

# A code is "ICD-10 shaped" iff it matches this AND is not a known Read v2.
# We use this only as a leakage canary; the canonical Read v2 codes contain
# a dot or are 5 chars alphanum, so this regex catches typical untranslated
# ICD-10 like 'F03', 'I10', 'G309'.
ICD10_LIKE = re.compile(r"^[A-Z][0-9]{2}[A-Z0-9]{0,4}$")
# Real Read v2 codes that match the regex but are legit (e.g. 'F110', 'E001')
# would also have a trailing dot in our vocab — but a few like 'F1100' don't.
# So we just *report* counts of suspicious codes, not assert.


def scan_split(path):
    total = 0
    k1 = k2 = k0 = 0
    seq_lens = []
    suspicious = {}
    for root, _, files in os.walk(path):
        for fn in files:
            if not fn.endswith('.parquet'):
                continue
            df = pq.read_table(os.path.join(root, fn)).to_pandas()
            for _, row in df.iterrows():
                total += 1
                events = row['EVENT']
                seq_lens.append(len(events))
                if len(events) == 0:
                    k0 += 1
                    continue
                last = events[-1]
                if last in DEMENTIA_CODES:
                    k1 += 1
                elif last in DEATH_CODES:
                    k2 += 1
                else:
                    k0 += 1
                for ev in events:
                    if (ev not in DEMENTIA_CODES and ev not in DEATH_CODES
                            and ICD10_LIKE.match(str(ev))):
                        suspicious[ev] = suspicious.get(ev, 0) + 1
    return total, k1, k2, k0, seq_lens, suspicious


def main():
    print("=" * 78)
    print("  HES-Fusion Dataset Statistics  (index_age=72, no SAW)")
    print("=" * 78)
    print(f"{'Split':<8} {'Total':>8} {'k=1':>8} {'k=2':>8} {'k=0':>8} "
          f"{'k1%':>8} {'mean_len':>10} {'med_len':>8}")
    print("-" * 78)
    grand = [0, 0, 0, 0]
    all_lens = []
    all_susp = {}
    for split in ['train', 'val', 'test']:
        p = f'{HES_BASE}split={split}/'
        if not os.path.isdir(p):
            print(f"{split:<8} <missing>")
            continue
        total, k1, k2, k0, seq_lens, susp = scan_split(p)
        rate = 100 * k1 / max(1, total)
        mean_len = float(np.mean(seq_lens)) if seq_lens else 0.0
        med_len = float(np.median(seq_lens)) if seq_lens else 0.0
        print(f"{split:<8} {total:>8} {k1:>8} {k2:>8} {k0:>8} "
              f"{rate:>7.2f}% {mean_len:>10.1f} {med_len:>8.0f}")
        grand[0] += total; grand[1] += k1; grand[2] += k2; grand[3] += k0
        all_lens.extend(seq_lens)
        for k, v in susp.items():
            all_susp[k] = all_susp.get(k, 0) + v
    print("-" * 78)
    print(f"{'TOTAL':<8} {grand[0]:>8} {grand[1]:>8} {grand[2]:>8} {grand[3]:>8} "
          f"{100*grand[1]/max(1,grand[0]):>7.2f}% "
          f"{(np.mean(all_lens) if all_lens else 0):>10.1f} "
          f"{(np.median(all_lens) if all_lens else 0):>8.0f}")

    print()
    print("Reference comparison:")
    print("  GP-only baseline (idx65): test k=1 ~60")
    print("  HES label augmented:      test k=1 ~301")
    print("  This run (HES fusion):    target test k=1 >= 301")

    print()
    print(f"Possible ICD-10 leakage in EVENT column: {len(all_susp)} distinct codes")
    if all_susp:
        top = sorted(all_susp.items(), key=lambda kv: -kv[1])[:20]
        for k, v in top:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
