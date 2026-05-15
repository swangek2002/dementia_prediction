"""
Independent verification of AUROC and PPV computation in compute_trade_comparison.py.

Strategy: compute the same numbers using three INDEPENDENT methods and check
they all agree exactly.

Methods:
  1. Original code path (sklearn roc_auc_score + manual PPV)
  2. Manual AUROC via Mann-Whitney U statistic on score pairs
  3. Manual AUROC via tied-pair-aware loop (slow but ground-truth)

Also sanity checks:
  - t_eval index for 5y horizon
  - Cohort filter integrity (8241)
  - Positive class count (121)
  - Per-bucket score statistics (positives should have HIGHER risk)
  - PPV at top X% reproducibility
  - Manual computation of one PPV value by hand
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path

DATA_DIR = Path('/Data0/swangek_data/991/CPRD/data')

# Load V3 NPZ
print("Loading V3 NPZ...")
d = np.load(DATA_DIR / 'test_cif_v3_full.npz', allow_pickle=True)
pids = d['patient_ids']
labels = d['labels']
event_t = d['event_time_scaled']
cif_dem = d['cif_dementia']
t_eval = d['t_eval']
print(f"  Before filter: {len(pids)} patients")
print(f"  t_eval shape: {t_eval.shape}, range [{t_eval[0]:.4f}, {t_eval[-1]:.4f}]")
print(f"  cif_dementia shape: {cif_dem.shape}")

# Filter leaky 16
with open(DATA_DIR / 'leaky_patients_test.txt') as f:
    leaky = {int(line.strip()) for line in f if line.strip()}
pid_int = np.array([int(p) for p in pids])
keep = ~np.isin(pid_int, list(leaky))
pids = pids[keep]; labels = labels[keep]; event_t = event_t[keep]; cif_dem = cif_dem[keep]
N = len(pids)
print(f"  After filter: {N} patients (expected 8241)")
assert N == 8241, "cohort size mismatch"
print(f"  Label counts: dementia={int((labels=='dementia').sum())}, "
      f"death={int((labels=='death').sum())}, censored={int((labels=='censored').sum())}")

# ---- Horizon mapping ----
HORIZON_TOT = 25.0
HORIZON_Y = 5.0
h_norm = HORIZON_Y / HORIZON_TOT
h_idx = int(np.argmin(np.abs(t_eval - h_norm)))
actual_t = t_eval[h_idx] * HORIZON_TOT
print(f"\nHorizon mapping check:")
print(f"  Target: {HORIZON_Y}y in [0,25] -> normalized {h_norm:.4f}")
print(f"  Picked t_eval[{h_idx}] = {t_eval[h_idx]:.6f} (actual {actual_t:.4f}y)")
print(f"  Neighbors: t_eval[{h_idx-1}]={t_eval[h_idx-1]:.6f}, t_eval[{h_idx+1}]={t_eval[h_idx+1]:.6f}")

# ---- Build labels (exclude_early_censored policy) ----
risk = cif_dem[:, h_idx]
is_dem = labels == 'dementia'
is_cens = labels == 'censored'
pos_mask = is_dem & (event_t <= h_norm)
cens_before = is_cens & (event_t < h_norm)
keep_for_eval = ~cens_before

y_true = pos_mask[keep_for_eval].astype(int)
y_score = risk[keep_for_eval]
n_total = len(y_true)
n_pos = int(y_true.sum())
n_neg = n_total - n_pos

print(f"\nEvaluation set (exclude_early_censored):")
print(f"  N total: {n_total}")
print(f"  N positive (dementia within 5y): {n_pos}")
print(f"  N negative: {n_neg}")
print(f"  Excluded (early-censored): {(~keep_for_eval).sum()}")

# Sanity: score distribution
print(f"\nScore distribution (CIF_dementia @ 5y):")
print(f"  Positives: mean={y_score[y_true==1].mean():.4f}, "
      f"median={np.median(y_score[y_true==1]):.4f}, "
      f"min={y_score[y_true==1].min():.4f}, max={y_score[y_true==1].max():.4f}")
print(f"  Negatives: mean={y_score[y_true==0].mean():.4f}, "
      f"median={np.median(y_score[y_true==0]):.4f}, "
      f"min={y_score[y_true==0].min():.4f}, max={y_score[y_true==0].max():.4f}")

# === METHOD 1: sklearn ===
auroc_sklearn = roc_auc_score(y_true, y_score)
print(f"\n[Method 1] sklearn roc_auc_score: {auroc_sklearn:.6f}")

# === METHOD 2: Mann-Whitney U / numpy ranks ===
# AUROC = (U1) / (n_pos * n_neg) where U1 = sum(ranks of positives) - n_pos*(n_pos+1)/2
from scipy.stats import rankdata
ranks = rankdata(y_score)  # average-rank for ties
U_pos = ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2
auroc_mw = U_pos / (n_pos * n_neg)
print(f"[Method 2] Mann-Whitney U / ranks:  {auroc_mw:.6f}")

# === METHOD 3: Direct pair counting (slow but exact) ===
# For every (pos, neg) pair: score_pos > score_neg => +1, score_pos == score_neg => +0.5
pos_scores = y_score[y_true == 1]
neg_scores = y_score[y_true == 0]
# To make it tractable for n_pos=121, n_neg=4396 = ~530k pairs, use broadcasting:
greater = (pos_scores[:, None] > neg_scores[None, :]).sum()
equal   = (pos_scores[:, None] == neg_scores[None, :]).sum()
auroc_pair = (greater + 0.5 * equal) / (n_pos * n_neg)
print(f"[Method 3] Direct pair counting:    {auroc_pair:.6f}")

# Cross-check
print(f"\n  Methods agree (sklearn vs MW): diff={abs(auroc_sklearn - auroc_mw):.2e}")
print(f"  Methods agree (sklearn vs pair): diff={abs(auroc_sklearn - auroc_pair):.2e}")
assert abs(auroc_sklearn - auroc_mw) < 1e-9, "AUROC mismatch sklearn vs MW!"
assert abs(auroc_sklearn - auroc_pair) < 1e-9, "AUROC mismatch sklearn vs pair!"
print(f"  [OK] All three AUROC methods agree to 1e-9")

# === PPV @ top K% — verify two ways ===
print(f"\nPPV verification:")
order = np.argsort(-y_score)
y_sorted = y_true[order]
sc_sorted = y_score[order]

# Sanity check: top 10 highest-risk patients
print(f"\n  Top 10 highest-risk patients:")
print(f"  {'rank':>5} {'risk':>8} {'true':>5}")
for i in range(10):
    print(f"  {i+1:>5} {sc_sorted[i]:>8.4f} {int(y_sorted[i]):>5}")

print(f"\n  PPV @ top X% — three independent calcs:")
print(f"  {'X%':>4} {'k':>5} {'#pos in top-k':>14} {'PPV (mine)':>11} {'PPV (alt)':>11} {'PPV (sort)':>11}")
for pct in [1, 5, 10]:
    k = max(1, int(np.round(n_total * pct / 100)))
    # Method A: same as compute_trade_comparison
    ppv_a = y_sorted[:k].sum() / k
    # Method B: threshold-based (everyone with score >= threshold)
    threshold = sc_sorted[k-1]
    in_topk = y_score >= threshold
    # Note: ties at threshold could give >k patients
    if in_topk.sum() == k:
        ppv_b = y_true[in_topk].sum() / in_topk.sum()
    else:
        ppv_b = float('nan')  # tie at boundary
    # Method C: sort-then-take
    sorted_idx = np.argsort(-y_score, kind='stable')
    ppv_c = y_true[sorted_idx[:k]].sum() / k
    n_pos_in_topk = int(y_sorted[:k].sum())
    print(f"  {pct:>3}% {k:>5} {n_pos_in_topk:>14} "
          f"{ppv_a*100:>10.3f}% {('-' if np.isnan(ppv_b) else f'{ppv_b*100:>10.3f}%')} "
          f"{ppv_c*100:>10.3f}%")
    assert abs(ppv_a - ppv_c) < 1e-9, "PPV mismatch between sort methods!"

# === Manual hand-check of one PPV ===
print(f"\nHand-check of PPV @ top 1%:")
k = max(1, int(np.round(n_total * 1 / 100)))
print(f"  N total in eval = {n_total}")
print(f"  k = round({n_total} * 0.01) = {k}")
print(f"  Top-{k} patients by predicted risk:")
print(f"  Of these, how many are true positives (label=='dementia' AND event_t <= 0.2)?")
top_k_count_pos = int(y_sorted[:k].sum())
print(f"  Count: {top_k_count_pos}")
print(f"  PPV = {top_k_count_pos} / {k} = {top_k_count_pos/k:.4f} = {top_k_count_pos/k*100:.2f}%")

# === Double-check positive labeling logic ===
print(f"\nPositive-class definition cross-check:")
print(f"  Total label == 'dementia': {int(is_dem.sum())}")
print(f"  Of which event_t <= {h_norm:.3f} ({HORIZON_Y}y): {int(pos_mask.sum())}")
print(f"  Of which event_t >  {h_norm:.3f}: {int((is_dem & (event_t > h_norm)).sum())}")
# Sanity: pos_mask + (dem after 5y) should equal total dementia
assert int(pos_mask.sum()) + int((is_dem & (event_t > h_norm)).sum()) == int(is_dem.sum())
print(f"  Sum = {int(pos_mask.sum()) + int((is_dem & (event_t > h_norm)).sum())} (= total dementia ✓)")

# === Edge case: would Method 2 (include early censored as neg) blow up? ===
print(f"\nFor reference, alternative policy include_early_censored_as_neg:")
auroc_inc = roc_auc_score(pos_mask.astype(int), risk)
print(f"  N={N}, pos={int(pos_mask.sum())}, neg={N-int(pos_mask.sum())}")
print(f"  AUROC = {auroc_inc:.4f} (low because 'unknown' early-censored counted as neg)")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print(f"V3 @ 5y horizon (exclude_early_censored policy):")
print(f"  AUROC = {auroc_sklearn:.4f}  (verified by 3 independent methods)")
print(f"  PPV @ top 1%  = {y_sorted[:max(1, int(np.round(n_total*0.01)))].sum()/max(1, int(np.round(n_total*0.01)))*100:.2f}%")
print(f"  PPV @ top 5%  = {y_sorted[:max(1, int(np.round(n_total*0.05)))].sum()/max(1, int(np.round(n_total*0.05)))*100:.2f}%")
print(f"  PPV @ top 10% = {y_sorted[:max(1, int(np.round(n_total*0.10)))].sum()/max(1, int(np.round(n_total*0.10)))*100:.2f}%")
