"""
Compute AUROC and PPV at top 1/5/10% for baseline models:
  - hes_aug (V1 labels, idx72)
  - idx68 5-fold CV (5 folds, different idx age = 68)

For each model, evaluate at 5y horizon with exclude_early_censored policy.

Two label modes:
  - Native: each model's NPZ labels (V1 for hes_aug; V1 for idx68 CV)
  - V2-overlay (hes_aug only): override V1 with V2 labels via PID match,
    restrict to canonical 8241 cohort. This makes hes_aug fairly comparable to V3.

idx68 CV cohort is DIFFERENT from V3 cohort (different idx age) — not overlaid.
"""
import numpy as np
import pyarrow.parquet as pq
import glob
from sklearn.metrics import roc_auc_score
from pathlib import Path

DATA_DIR = Path('/Data0/swangek_data/991/CPRD/data')
HORIZON_TOT = 25.0
HORIZON_Y = 5.0
H_NORM = HORIZON_Y / HORIZON_TOT  # 0.2

# Canonical V2 clean cohort = V2 test minus 16 leaky
with open(DATA_DIR / 'leaky_patients_test.txt') as f:
    leaky_test = {int(line.strip()) for line in f if line.strip()}

# Build V2 PID -> (label, event_t_scaled) map from V2 NPZ (truth source)
v2 = np.load(DATA_DIR / 'test_cif_v2_full.npz', allow_pickle=True)
v2_pids = np.array([int(p) for p in v2['patient_ids']])
v2_label_map = dict(zip(v2_pids.tolist(), v2['labels'].tolist()))
v2_event_map = dict(zip(v2_pids.tolist(), v2['event_time_scaled'].tolist()))
canonical_pids = set(v2_pids.tolist()) - leaky_test
print(f"V2 truth source: {len(v2_pids)} patients, canonical (minus leaky 16) = {len(canonical_pids)}")


def metrics_at_5y(pids_int, labels, event_t, risk_5y, name):
    """Compute AUROC + PPV at top 1/5/10% at 5y horizon with exclude_early_censored."""
    is_dem = (labels == 'dementia')
    is_cens = (labels == 'censored')
    pos_mask = is_dem & (event_t <= H_NORM)
    cens_before = is_cens & (event_t < H_NORM)
    keep = ~cens_before
    y_true = pos_mask[keep].astype(int)
    y_score = risk_5y[keep]
    n_tot = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n_tot - n_pos
    if n_pos == 0 or n_neg == 0:
        print(f"  [skip] {name}: no positives or no negatives in eval set")
        return None
    auroc = roc_auc_score(y_true, y_score)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    ppv = {}
    for pct in [1, 5, 10]:
        k = max(1, int(np.round(n_tot * pct / 100)))
        ppv[pct] = y_sorted[:k].sum() / k
    return {
        'N_total': n_tot, 'N_pos': n_pos, 'N_neg': n_neg,
        'prevalence': n_pos / n_tot,
        'AUROC': auroc,
        'PPV1': ppv[1], 'PPV5': ppv[5], 'PPV10': ppv[10],
    }


def load_and_compute(npz_path, use_v2_overlay=False, restrict_to_canonical=False, name=''):
    d = np.load(npz_path, allow_pickle=True)
    pids = np.array([int(p) for p in d['patient_ids']])
    labels = d['labels']
    event_t = d['event_time_scaled']
    cif_dem = d['cif_dementia']
    t_eval = d['t_eval']
    h_idx = int(np.argmin(np.abs(t_eval - H_NORM)))
    risk_5y = cif_dem[:, h_idx]
    actual_t = t_eval[h_idx] * HORIZON_TOT
    print(f"\n  Loaded {Path(npz_path).name}: N={len(pids)}, t_eval[{h_idx}]={actual_t:.4f}y")

    if use_v2_overlay:
        # Restrict to canonical PIDs, override label/event_t with V2 source
        keep_mask = np.array([pid in canonical_pids for pid in pids])
        pids = pids[keep_mask]
        risk_5y = risk_5y[keep_mask]
        labels = np.array([v2_label_map[pid] for pid in pids])
        event_t = np.array([v2_event_map[pid] for pid in pids])
        print(f"    [V2-overlay] kept {len(pids)} canonical patients with V2 labels")
    elif restrict_to_canonical:
        keep_mask = np.array([pid in canonical_pids for pid in pids])
        pids = pids[keep_mask]
        labels = labels[keep_mask]
        event_t = event_t[keep_mask]
        risk_5y = risk_5y[keep_mask]
        print(f"    [canonical-restrict] kept {len(pids)} patients")

    import collections
    print(f"    label counts: {dict(collections.Counter(labels.tolist()))}")
    return metrics_at_5y(pids, labels, event_t, risk_5y, name)


def print_row(name, r):
    if r is None:
        print(f"  {name:<45} (no result)")
        return
    print(f"  {name:<45} {r['N_total']:>6}  {r['N_pos']:>5}  "
          f"{r['prevalence']*100:>6.2f}%  {r['AUROC']:.4f}  "
          f"{r['PPV1']*100:>6.2f}%  {r['PPV5']*100:>6.2f}%  {r['PPV10']*100:>6.2f}%")

print()
print("="*100)
print("BASELINE MODELS @ 5y horizon, exclude_early_censored policy")
print("="*100)
print(f"  {'Model':<45} {'N_tot':>6}  {'N_pos':>5}  {'Prev':>7}  {'AUROC':>6}  "
      f"{'PPV1%':>7}  {'PPV5%':>7}  {'PPV10%':>7}")
print(f"  {'-'*45} {'-'*6}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}")

# === V3 (reference) ===
r = load_and_compute(DATA_DIR / 'test_cif_v3_full.npz', restrict_to_canonical=True,
                     name='V3 (canonical)')
print_row("V3 (top1% SST, ref) -- canonical V2 cohort", r)

# === hes_aug (V1 labels, native) ===
r_native = load_and_compute(DATA_DIR / 'test_cif_hes_aug_full.npz', name='hes_aug native')
print_row("hes_aug (NATIVE V1 labels, idx72)", r_native)

# === hes_aug (V2 overlay) ===
r_overlay = load_and_compute(DATA_DIR / 'test_cif_hes_aug_full.npz', use_v2_overlay=True,
                              name='hes_aug V2-overlay')
print_row("hes_aug (V2-LABEL OVERLAY, canonical 8241)", r_overlay)

# === idx68 5-fold CV ===
print()
print("idx68 5-fold CV (different idx age = 68, NOT directly comparable to V3):")
fold_results = []
for k in range(5):
    p = DATA_DIR / f'test_cif_idx68_cv_fold{k}_full.npz'
    r = load_and_compute(p, name=f'idx68 fold{k}')
    fold_results.append(r)
    print_row(f"idx68 fold{k} (NATIVE V1 labels, idx68)", r)

# Aggregate across 5 folds
valid = [r for r in fold_results if r is not None]
if valid:
    print()
    avg = {k: np.mean([r[k] for r in valid]) for k in ['AUROC','PPV1','PPV5','PPV10','prevalence']}
    std = {k: np.std([r[k] for r in valid], ddof=1) for k in ['AUROC','PPV1','PPV5','PPV10']}
    print(f"  5-fold AVERAGE (mean ± std):")
    print(f"    AUROC      = {avg['AUROC']:.4f} ± {std['AUROC']:.4f}")
    print(f"    PPV@top 1% = {avg['PPV1']*100:.2f}% ± {std['PPV1']*100:.2f}%")
    print(f"    PPV@top 5% = {avg['PPV5']*100:.2f}% ± {std['PPV5']*100:.2f}%")
    print(f"    PPV@top10% = {avg['PPV10']*100:.2f}% ± {std['PPV10']*100:.2f}%")
    print(f"    Mean prevalence: {avg['prevalence']*100:.2f}%")

print()
print("="*100)
print("Side-by-side summary (5y horizon, exclude_early_censored)")
print("="*100)
print(f"  {'Model':<45} {'AUROC':>7}  {'Prev':>7}  {'PPV1':>7}  {'PPV5':>7}  {'PPV10':>7}  {'lift@1%':>8}")
print(f"  {'-'*45} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}")

# helper to print lift
def fmt_lift(p, prev):
    return f"{p/prev:.1f}x" if prev > 0 else 'NaN'

def pr_row(name, r):
    if r is None: return
    print(f"  {name:<45} {r['AUROC']:>7.4f}  {r['prevalence']*100:>6.2f}%  "
          f"{r['PPV1']*100:>6.2f}%  {r['PPV5']*100:>6.2f}%  {r['PPV10']*100:>6.2f}%  "
          f"{fmt_lift(r['PPV1'], r['prevalence']):>8}")

# Re-load V3 for summary
v3 = load_and_compute(DATA_DIR / 'test_cif_v3_full.npz', restrict_to_canonical=True,
                      name='V3')
pr_row("V3 (idx72, V2 labels)", v3)
pr_row("hes_aug (idx72, V1 labels, native)", r_native)
pr_row("hes_aug (idx72, V2 overlay)", r_overlay)
if valid:
    fold_mean = {
        'AUROC': avg['AUROC'], 'prevalence': avg['prevalence'],
        'PPV1': avg['PPV1'], 'PPV5': avg['PPV5'], 'PPV10': avg['PPV10']
    }
    pr_row("idx68 5-fold CV (idx68, V1 labels, avg)", fold_mean)
