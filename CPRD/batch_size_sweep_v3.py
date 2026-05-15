"""
Per-batch vs cohort C_td comparison on V3 (current cohort-level peak).

Goal: empirically show the mechanism by which per-batch averaging fails:
- Many batches have 0 dementia cases (excluded)
- Small batches → noisy C_td
- Cross-batch informative pairs lost
- Aggregating per-batch C_td gives WRONG estimate of cohort C_td

We sweep batch_size from 16 → 8241 (full cohort) and report:
- Mean per-batch C_td (with diagnostics)
- Number of batches with 0/1/2+ dementia cases
- Number of batches that failed C_td calculation

Expected: as batch_size increases, per-batch averaged C_td converges to cohort 0.8506.
"""
import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
import json

NPZ = '/Data0/swangek_data/991/CPRD/data/test_cif_v3_full.npz'
LEAKY_TXT = '/Data0/swangek_data/991/CPRD/data/leaky_patients_test.txt'

d = np.load(NPZ, allow_pickle=True)
pids = d['patient_ids']
labels = d['labels']
event_time = d['event_time_scaled']
cif = d['cif_dementia']
t_eval = d['t_eval']

LEAKY = set()
with open(LEAKY_TXT) as f:
    for l in f:
        if l.strip():
            LEAKY.add(int(l.strip()))

keep = ~np.isin(pids, list(LEAKY))
labels = labels[keep]
event_time = event_time[keep]
cif = cif[keep]
N = len(labels)
evt_dem = (labels == 'dementia').astype(int)
print(f"Cohort size: {N}")
print(f"Dementia events: {evt_dem.sum()}")
print(f"Censored: {(labels=='censored').sum()}")
print(f"Death: {(labels=='death').sum()}")
print()


def per_batch_sweep(batch_size, seed=None):
    """Compute per-batch averaged C_td at given batch size.

    If seed provided, shuffles patient order to mimic training behavior."""
    indices = np.arange(N)
    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    n_batches = (N + batch_size - 1) // batch_size
    batch_ctds = []
    n_zero_dem = 0
    n_one_dem = 0
    n_twoplus_dem = 0
    n_failed = 0
    n_used = 0
    n_dem_in_used = []
    for b in range(n_batches):
        bi = indices[b * batch_size : (b + 1) * batch_size]
        b_cif = cif[bi]
        b_et = event_time[bi]
        b_evt = evt_dem[bi]
        n_d = b_evt.sum()
        if n_d == 0:
            n_zero_dem += 1
            continue
        elif n_d == 1:
            n_one_dem += 1
        else:
            n_twoplus_dem += 1
        try:
            surv = pd.DataFrame((1 - b_cif).T, index=t_eval)
            ev = EvalSurv(surv, b_et.astype(float), b_evt.astype(int), censor_surv='km')
            c = ev.concordance_td('antolini')
            if c is not None and not np.isnan(c):
                batch_ctds.append(c)
                n_used += 1
                n_dem_in_used.append(int(n_d))
            else:
                n_failed += 1
        except Exception as e:
            n_failed += 1
    return {
        'batch_size': batch_size,
        'n_batches': n_batches,
        'n_zero_dem': n_zero_dem,
        'n_one_dem': n_one_dem,
        'n_twoplus_dem': n_twoplus_dem,
        'n_failed': n_failed,
        'n_used': n_used,
        'frac_zero_dem': n_zero_dem / n_batches,
        'frac_used': n_used / n_batches,
        'mean_dem_per_batch': float(np.mean(n_dem_in_used)) if n_dem_in_used else 0.0,
        'avg_ctd': float(np.mean(batch_ctds)) if batch_ctds else float('nan'),
        'std_ctd': float(np.std(batch_ctds)) if batch_ctds else float('nan'),
        'min_ctd': float(np.min(batch_ctds)) if batch_ctds else float('nan'),
        'max_ctd': float(np.max(batch_ctds)) if batch_ctds else float('nan'),
        'median_ctd': float(np.median(batch_ctds)) if batch_ctds else float('nan'),
    }


sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4120, 8241]
print(f"{'batch_sz':>10} {'n_batches':>10} {'%zero_dem':>10} {'%used':>8} {'mean_dem/batch':>15} {'avg C_td':>10} {'std':>8} {'[min, max]':>22}")
print("-" * 110)
results = []
for bs in sizes:
    r = per_batch_sweep(bs, seed=None)  # sequential ordering as in original eval
    results.append(r)
    print(f"{bs:>10} {r['n_batches']:>10} {r['frac_zero_dem']*100:>9.1f}% {r['frac_used']*100:>7.1f}% "
          f"{r['mean_dem_per_batch']:>15.2f} {r['avg_ctd']:>10.4f} {r['std_ctd']:>8.4f} "
          f"[{r['min_ctd']:.3f}, {r['max_ctd']:.3f}]")

print("\nAlso shuffled (seed=1337) for comparison vs original eval (which used shuffle=False, but original training also used random_split):")
print(f"{'batch_sz':>10} {'avg C_td (shuffled)':>20}")
for bs in [16, 32, 64]:
    rs = []
    for seed in [1, 2, 3, 1337, 42]:
        r = per_batch_sweep(bs, seed=seed)
        rs.append(r['avg_ctd'])
    print(f"{bs:>10} {np.mean(rs):>20.4f} (5 seeds: mean={np.mean(rs):.4f}, std={np.std(rs):.4f})")

# Save results JSON
with open('/Data0/swangek_data/991/CPRD/batch_sweep_v3.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to /Data0/swangek_data/991/CPRD/batch_sweep_v3.json")
