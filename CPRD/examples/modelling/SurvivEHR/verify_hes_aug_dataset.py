"""
verify_hes_aug_dataset.py
=========================
Verify HES-augmented dataset statistics before training.
Compares label distributions across train/val/test splits.
"""
import os
import pyarrow.parquet as pq

DEMENTIA_CODES = set([
    'F110.', 'Eu00.', 'Eu01.', 'Eu02z', 'Eu002', 'E00..', 'Eu023', 'Eu00z', 'Eu025',
    'Eu01z', 'E001.', 'F1100', 'Eu001', 'E004.', 'Eu000', 'Eu02.', 'Eu013', 'E000.',
    'Eu01y', 'E001z', 'F1101', 'Eu020', 'E004z', 'E0021', 'Eu02y', 'Eu012', 'Eu011',
    'E00z.', 'E0040', 'E003.', 'E0020',
])
DEATH_CODES = set(['DEATH'])

HES_BASE = '/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_aug/'


def count_labels(dataset_path):
    total, k1, k2, k0 = 0, 0, 0, 0
    for root, dirs, files in os.walk(dataset_path):
        for fn in files:
            if not fn.endswith('.parquet'):
                continue
            df = pq.read_table(os.path.join(root, fn)).to_pandas()
            for _, row in df.iterrows():
                total += 1
                events = row['EVENT']
                if len(events) > 0:
                    last = events[-1]
                    if last in DEMENTIA_CODES:
                        k1 += 1
                    elif last in DEATH_CODES:
                        k2 += 1
                    else:
                        k0 += 1
    return total, k1, k2, k0


print("=" * 70)
print("  HES-Augmented Dataset Statistics (index_age=72, no SAW)")
print("=" * 70)
print(f"{'Split':<8} {'Total':>8} {'Dementia(k=1)':>14} {'Death(k=2)':>11} {'Censored(k=0)':>14} {'Event Rate':>11}")
print("-" * 70)

grand_total, grand_k1, grand_k2, grand_k0 = 0, 0, 0, 0
for split in ['train', 'val', 'test']:
    path = f'{HES_BASE}split={split}/'
    total, k1, k2, k0 = count_labels(path)
    rate = 100 * k1 / max(1, total)
    print(f"{split:<8} {total:>8} {k1:>14} {k2:>11} {k0:>14} {rate:>10.2f}%")
    grand_total += total
    grand_k1 += k1
    grand_k2 += k2
    grand_k0 += k0

print("-" * 70)
grand_rate = 100 * grand_k1 / max(1, grand_total)
print(f"{'TOTAL':<8} {grand_total:>8} {grand_k1:>14} {grand_k2:>11} {grand_k0:>14} {grand_rate:>10.2f}%")
print("=" * 70)
