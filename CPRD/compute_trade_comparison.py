"""
Compute 5y AUROC and PPV at top 1%/5%/10% for direct comparison with TRADE paper.

TRADE (NYU EHR foundation model for AD/ADRD/MCI):
  - 1y AUROC: 0.772
  - 5y AUROC: 0.735
  - 5y PPV top 1%:  39.2%
  - 5y PPV top 5%:  27.8%
  - 5y PPV top 10%: 22.4%
  - eRADAR baseline 5y AUROC: 0.688

Our setup:
  - CIF curves over 25-year horizon (1000 time points)
  - Query CIF at t_eval ~= 0.2 (= 5y / 25y) to get per-patient 5y risk
  - Outcome: dementia only (TRADE uses AD/ADRD/MCI umbrella)
  - Cohort: UK Biobank linked CPRD+HES, idx age 72

Binary label policy at 5y:
  - Positive: label=='dementia' AND event_time_scaled <= 0.2 (5y)
  - Negative: censored with event_time_scaled >= 0.2, OR death/dementia after 5y
  - Excluded: censored before 5y (status uncertain) -- two policies tested
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path

DATA_DIR = Path('/Data0/swangek_data/991/CPRD/data')

# Canonical clean cohort PIDs = V5 test set (8241 patients, post GP-prevalent removal).
# V3 NPZ has 8257; we will filter to the 8241 by removing the 16 leaky test PIDs.
LEAKY_TEST_FILE = DATA_DIR / 'leaky_patients_test.txt'

with open(LEAKY_TEST_FILE) as f:
    leaky_test_pids = {int(line.strip()) for line in f if line.strip()}
print(f"Leaky test PIDs: {len(leaky_test_pids)}")


def load_and_filter(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    pids = d['patient_ids']
    labels = d['labels']
    event_t = d['event_time_scaled']
    cif_dem = d['cif_dementia']
    cif_dth = d['cif_death']
    t_eval = d['t_eval']
    n_before = len(pids)

    # Filter out leaky
    pid_int = np.array([int(p) for p in pids])
    keep_mask = ~np.isin(pid_int, list(leaky_test_pids))
    pids = pids[keep_mask]
    labels = labels[keep_mask]
    event_t = event_t[keep_mask]
    cif_dem = cif_dem[keep_mask]
    cif_dth = cif_dth[keep_mask]
    print(f"  {npz_path.name}: {n_before} -> {keep_mask.sum()} after filtering leaky")
    return pids, labels, event_t, cif_dem, cif_dth, t_eval


def metrics_at_horizon(pids, labels, event_t, cif_dem, t_eval, horizon_years,
                       horizon_total_years=25.0, policy='exclude_early_censored'):
    """Compute AUROC and PPV at top 1/5/10% for a given horizon.

    policy:
      'exclude_early_censored': drop censored patients with event_t < horizon (TRADE-style: their status unknown)
      'include_early_censored_as_neg': treat early-censored as negative (more lenient denominator)
    """
    h_norm = horizon_years / horizon_total_years
    h_idx = int(np.argmin(np.abs(t_eval - h_norm)))
    actual_t_at_idx = t_eval[h_idx] * horizon_total_years
    risk = cif_dem[:, h_idx]

    # Build label
    # Positive: dementia & event_t <= h_norm
    # Negative_definite: event_t > h_norm (whether dementia, death, or censored beyond)
    # Negative_competing: died before horizon, no dementia -> in TRADE this would also be "no AD by 5y" = negative
    # Censored_early: censored & event_t < h_norm  (uncertain)
    is_dem = (labels == 'dementia')
    is_death = (labels == 'death')
    is_cens = (labels == 'censored')

    pos_mask = is_dem & (event_t <= h_norm)
    # Definite negatives: any event after horizon
    neg_after_horizon = (event_t > h_norm)
    # Competing death before horizon: subject did NOT have AD by horizon, count as negative
    neg_death_before = is_death & (event_t <= h_norm)

    if policy == 'exclude_early_censored':
        cens_before = is_cens & (event_t < h_norm)
        keep = ~cens_before  # drop early-censored
        y_true = pos_mask[keep].astype(int)
        y_score = risk[keep]
        excl = (~keep).sum()
    elif policy == 'include_early_censored_as_neg':
        keep = np.ones_like(labels, dtype=bool)
        y_true = pos_mask.astype(int)
        y_score = risk
        excl = 0
    else:
        raise ValueError(policy)

    auroc = roc_auc_score(y_true, y_score)

    # Sort by score descending for PPV
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n = len(y_sorted)
    ppv = {}
    for pct in [1, 5, 10]:
        k = max(1, int(np.round(n * pct / 100)))
        ppv[pct] = y_sorted[:k].sum() / k

    return {
        'horizon_y': horizon_years,
        'h_idx': h_idx,
        'actual_t_y': actual_t_at_idx,
        'n_total': len(labels),
        'n_kept': len(y_true),
        'n_excl_censored_early': excl,
        'n_pos': int(y_true.sum()),
        'n_neg': int(len(y_true) - y_true.sum()),
        'auroc': auroc,
        'ppv_top1': ppv[1],
        'ppv_top5': ppv[5],
        'ppv_top10': ppv[10],
        'policy': policy,
    }


def print_block(model_name, npz_path):
    print('='*78)
    print(f'{model_name}')
    print('='*78)
    pids, labels, event_t, cif_dem, cif_dth, t_eval = load_and_filter(npz_path)
    print(f"  cohort: {len(pids)} patients | dementia={int((labels=='dementia').sum())} "
          f"death={int((labels=='death').sum())} censored={int((labels=='censored').sum())}")
    print()

    for horizon in [1, 5]:
        for policy in ['exclude_early_censored', 'include_early_censored_as_neg']:
            r = metrics_at_horizon(pids, labels, event_t, cif_dem, t_eval, horizon,
                                   horizon_total_years=25.0, policy=policy)
            print(f"  Horizon = {horizon}y (t_eval[{r['h_idx']}] = {r['actual_t_y']:.4f}y) "
                  f"| policy={policy}")
            print(f"    kept N={r['n_kept']}  pos={r['n_pos']}  neg={r['n_neg']}  "
                  f"(excluded early-censored={r['n_excl_censored_early']})")
            print(f"    AUROC = {r['auroc']:.4f}")
            print(f"    PPV @ top  1% = {r['ppv_top1']*100:.2f}%")
            print(f"    PPV @ top  5% = {r['ppv_top5']*100:.2f}%")
            print(f"    PPV @ top 10% = {r['ppv_top10']*100:.2f}%")
            print()
    print()


print()
print('TRADE paper benchmark (NYU AD/ADRD/MCI, US EHR foundation model)')
print('  1y AUROC: 0.772')
print('  5y AUROC: 0.735')
print('  5y PPV @ top 1%/5%/10%: 39.2% / 27.8% / 22.4%')
print('  eRADAR baseline 5y AUROC: 0.688')
print()
print('Note: TRADE outcome = AD/ADRD/MCI umbrella (~7.2% event rate)')
print('Ours: dementia only (~5% event rate)')
print()

for name, fname in [
    ('V3 (top1% SST, V2 labels, GP+22dim HES static)', 'test_cif_v3_full.npz'),
    ('V5 (top5% SST, V2 labels, GP+22dim HES static)', 'test_cif_v5_full.npz'),
    ('V2 (base, no SST, V2 labels)',                    'test_cif_v2_full.npz'),
    ('V2 ablation (no HES static)',                      'test_cif_v2_ablation_full.npz'),
    ('V4 (top2% SST)',                                   'test_cif_v4_full.npz'),
]:
    p = DATA_DIR / fname
    if p.exists():
        print_block(name, p)
    else:
        print(f"[skip] {p} not found")
