"""
compute_metrics_aligned.py
==========================
Step 3: Time-aligned AUROC@5y for V3 model.

The CIF@5y in v1 is "5 years from prediction point", but the AUROC label is
"5 years from index date (age 72)". This mismatch causes severely degraded
AUROC@5y. This script computes a per-patient time-aligned CIF using the
fine-grained grid saved by inference_test_metrics_v2.py.

For each patient:
  delta_i = years from index date (age 72) to prediction point (2nd-to-last event)
          = event_time_from_index_years - event_time_scaled * 5.0
  tau_i   = 5.0 - delta_i   (years from prediction point until "5y from index")
  cif_aligned = CIF at tau_i years (interpolated from fine grid)

Patients with delta_i < 0 or delta_i > 5 are excluded from valid cohort.

Usage:
    cd /Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR
    PYTHONPATH=/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD \
    /Data0/swangek_data/conda_envs/survivehr/bin/python compute_metrics_aligned.py
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index

INPUT_CSV = "/Data0/swangek_data/991/CPRD/data/test_cif_v3_aligned.csv"
INPUT_NPZ = "/Data0/swangek_data/991/CPRD/data/test_cif_v3_fine.npz"


def auroc_at_5y(df_valid, mask, score_col, risk):
    """Compute AUROC for the 5y dementia/death cases vs ≥5y controls."""
    if risk == "dementia":
        cases = (df_valid['label'] == 'dementia') & (df_valid['event_time_from_index_years'] <= 5.0)
    else:
        cases = (df_valid['label'] == 'death') & (df_valid['event_time_from_index_years'] <= 5.0)
    controls = df_valid['event_time_from_index_years'] > 5.0
    m = (cases | controls) & mask
    n_cases = int(cases[m].sum())
    n_controls = int(controls[m].sum())
    if n_cases == 0 or n_controls == 0:
        return np.nan, n_cases, n_controls
    y_true = cases[m].astype(int).values
    y_score = df_valid.loc[m, score_col].values
    if np.isnan(y_score).any():
        return np.nan, n_cases, n_controls
    return roc_auc_score(y_true, y_score), n_cases, n_controls


def main():
    df = pd.read_csv(INPUT_CSV)
    npz = np.load(INPUT_NPZ)
    t_grid_years = npz['t_grid_years']
    cif_dem_grid = npz['cif_dementia']
    cif_death_grid = npz['cif_death']
    pids_npz = npz['patient_ids']

    print(f"Loaded {len(df)} patients from CSV")
    print(f"NPZ shapes: cif_dem={cif_dem_grid.shape}, t_grid={t_grid_years.shape}")
    print(f"  t_grid years: {t_grid_years.round(2).tolist()}")

    # Verify alignment between CSV and NPZ
    if not (pids_npz == df['patient_id'].values).all():
        print("WARNING: NPZ patient order != CSV order, merging by patient_id")
        # Reorder NPZ rows to match CSV
        pid_to_npz_idx = {int(p): i for i, p in enumerate(pids_npz)}
        new_order = np.array([pid_to_npz_idx[int(p)] for p in df['patient_id'].values])
        cif_dem_grid = cif_dem_grid[new_order]
        cif_death_grid = cif_death_grid[new_order]
    else:
        print("CSV and NPZ patient order: aligned.")

    # delta_i = years from index to prediction point
    df['delta_i'] = df['event_time_from_index_years'] - df['event_time_scaled'] * 5.0
    df['tau_i_years'] = 5.0 - df['delta_i']

    print(f"\ndelta_i distribution: median={df['delta_i'].median():.2f}, "
          f"p25={df['delta_i'].quantile(0.25):.2f}, p75={df['delta_i'].quantile(0.75):.2f}, "
          f"min={df['delta_i'].min():.2f}, max={df['delta_i'].max():.2f}")

    mask_valid = (df['delta_i'] >= 0) & (df['delta_i'] <= 5.0)
    print(f"\nValid cohort (delta_i in [0, 5]): {mask_valid.sum()}/{len(df)}")
    print(f"  Excluded delta_i<0:  {(df['delta_i'] < 0).sum()}")
    print(f"  Excluded delta_i>5:  {(df['delta_i'] > 5).sum()}")

    # Per-patient interpolation
    print("\nInterpolating time-aligned CIF for each valid patient...")
    cif_dem_aligned = np.full(len(df), np.nan)
    cif_death_aligned = np.full(len(df), np.nan)

    valid_indices = df.index[mask_valid].values
    for idx in valid_indices:
        t_target = float(df.loc[idx, 'tau_i_years'])
        f_dem = interp1d(t_grid_years, cif_dem_grid[idx], kind='linear',
                         bounds_error=False,
                         fill_value=(float(cif_dem_grid[idx, 0]), float(cif_dem_grid[idx, -1])))
        f_death = interp1d(t_grid_years, cif_death_grid[idx], kind='linear',
                           bounds_error=False,
                           fill_value=(float(cif_death_grid[idx, 0]), float(cif_death_grid[idx, -1])))
        cif_dem_aligned[idx] = float(f_dem(t_target))
        cif_death_aligned[idx] = float(f_death(t_target))

    df['cif_dementia_aligned_5y'] = cif_dem_aligned
    df['cif_death_aligned_5y'] = cif_death_aligned

    df_valid = df[mask_valid].copy()
    print(f"\nTime-aligned CIF stats (valid cohort):")
    print(f"  cif_dem_aligned: mean={df_valid['cif_dementia_aligned_5y'].mean():.4f}, "
          f"median={df_valid['cif_dementia_aligned_5y'].median():.4f}, "
          f"p95={df_valid['cif_dementia_aligned_5y'].quantile(0.95):.4f}")
    print(f"  cif_death_aligned: mean={df_valid['cif_death_aligned_5y'].mean():.4f}, "
          f"median={df_valid['cif_death_aligned_5y'].median():.4f}, "
          f"p95={df_valid['cif_death_aligned_5y'].quantile(0.95):.4f}")

    # AUROC@5y dementia (3 risk scores)
    print(f"\n{'='*70}")
    print("=== Time-Aligned AUROC@5y Comparison (Step 3) ===")
    print(f"{'='*70}")

    cases_d = (df_valid['label'] == 'dementia') & (df_valid['event_time_from_index_years'] <= 5.0)
    controls_d = df_valid['event_time_from_index_years'] > 5.0
    print(f"\nAUROC@5y Dementia (cases={int(cases_d.sum())}, controls={int(controls_d.sum())}):")

    full_mask = pd.Series([True] * len(df_valid), index=df_valid.index)
    for name, col in [
        ("CIF@5y (saturated, original)", "cif_dementia_5y"),
        ("pi_dementia (asymptotic)",     "pi_dementia"),
        ("CIF time-aligned (the fix)",   "cif_dementia_aligned_5y"),
    ]:
        auroc, nc, nctl = auroc_at_5y(df_valid, full_mask, col, "dementia")
        print(f"  {name:42s} = {auroc:.4f}")

    cases_dth = (df_valid['label'] == 'death') & (df_valid['event_time_from_index_years'] <= 5.0)
    controls_dth = df_valid['event_time_from_index_years'] > 5.0
    print(f"\nAUROC@5y Death (cases={int(cases_dth.sum())}, controls={int(controls_dth.sum())}):")
    for name, col in [
        ("CIF@5y (saturated, original)", "cif_death_5y"),
        ("pi_death (asymptotic)",        "pi_death"),
        ("CIF time-aligned (the fix)",   "cif_death_aligned_5y"),
    ]:
        auroc, nc, nctl = auroc_at_5y(df_valid, full_mask, col, "death")
        print(f"  {name:42s} = {auroc:.4f}")

    # Top-K precision @5y dementia (using time-aligned CIF)
    print("\nTop-K Precision @5y Dementia (using time-aligned CIF):")
    df_sorted = df_valid.sort_values('cif_dementia_aligned_5y', ascending=False)
    for pct in [0.01, 0.05, 0.10]:
        k = int(len(df_sorted) * pct)
        top_k = df_sorted.head(k)
        tp = ((top_k['label'] == 'dementia') & (top_k['event_time_from_index_years'] <= 5.0)).sum()
        suffix = "  (cf. DemRisk=37%, NYU EHR-BERT=39.2%)" if pct == 0.01 else ""
        print(f"  Top {pct*100:>2.0f}%: {tp/k*100:5.1f}% ({tp}/{k}){suffix}")

    # Same on full cohort with original CIF@5y for comparison
    print("\nTop-K Precision @5y Dementia (using CIF@5y, full cohort, sanity check vs Step 2):")
    df_sorted0 = df.sort_values('cif_dementia_5y', ascending=False)
    for pct in [0.01, 0.05, 0.10]:
        k = int(len(df_sorted0) * pct)
        top_k = df_sorted0.head(k)
        tp = ((top_k['label'] == 'dementia') & (top_k['event_time_from_index_years'] <= 5.0)).sum()
        print(f"  Top {pct*100:>2.0f}%: {tp/k*100:5.1f}% ({tp}/{k})")

    # Harrell's C using pi_dementia (time-independent risk score)
    print(f"\n{'='*70}")
    print("Harrell's C (cause-specific) — note: lifelines expects higher score = longer survival")
    print(f"{'='*70}")
    mask_h_dem = df['label'].isin(['dementia', 'censored'])
    dh = df[mask_h_dem]
    c_pi_dem = concordance_index(
        event_times=dh['event_time_from_index_years'].values,
        predicted_scores=-dh['pi_dementia'].values,
        event_observed=(dh['label'] == 'dementia').values,
    )
    print(f"  Dementia (pi_dementia, full cohort, n={len(dh)}, events={int((dh['label']=='dementia').sum())}): "
          f"{c_pi_dem:.4f}  (cf. Yuan=0.749)")

    # Also with aligned CIF (only valid cohort)
    dh_v = df_valid[df_valid['label'].isin(['dementia', 'censored'])]
    c_aligned_dem = concordance_index(
        event_times=dh_v['event_time_from_index_years'].values,
        predicted_scores=-dh_v['cif_dementia_aligned_5y'].values,
        event_observed=(dh_v['label'] == 'dementia').values,
    )
    print(f"  Dementia (CIF aligned, valid cohort, n={len(dh_v)}, events={int((dh_v['label']=='dementia').sum())}): "
          f"{c_aligned_dem:.4f}")

    mask_h_dth = df['label'].isin(['death', 'censored'])
    dhd = df[mask_h_dth]
    c_pi_dth = concordance_index(
        event_times=dhd['event_time_from_index_years'].values,
        predicted_scores=-dhd['pi_death'].values,
        event_observed=(dhd['label'] == 'death').values,
    )
    print(f"  Death    (pi_death,    full cohort, n={len(dhd)}, events={int((dhd['label']=='death').sum())}): "
          f"{c_pi_dth:.4f}")

    dhd_v = df_valid[df_valid['label'].isin(['death', 'censored'])]
    c_aligned_dth = concordance_index(
        event_times=dhd_v['event_time_from_index_years'].values,
        predicted_scores=-dhd_v['cif_death_aligned_5y'].values,
        event_observed=(dhd_v['label'] == 'death').values,
    )
    print(f"  Death    (CIF aligned, valid cohort, n={len(dhd_v)}, events={int((dhd_v['label']=='death').sum())}): "
          f"{c_aligned_dth:.4f}")

    # Calibration @5y with aligned CIF
    print(f"\n{'='*70}")
    print("Calibration @5y (dementia, time-aligned CIF, valid cohort with known 5y outcome):")
    print(f"{'='*70}")
    mask_cal = (df_valid['event_time_from_index_years'] > 5.0) | \
               ((df_valid['label'] == 'dementia') & (df_valid['event_time_from_index_years'] <= 5.0))
    df_cal = df_valid[mask_cal].copy()
    df_cal['actual'] = ((df_cal['label'] == 'dementia') &
                        (df_cal['event_time_from_index_years'] <= 5.0)).astype(int)
    df_cal['pred_bin'] = pd.qcut(df_cal['cif_dementia_aligned_5y'], q=10, duplicates='drop')
    cal = df_cal.groupby('pred_bin', observed=True).agg(
        mean_predicted=('cif_dementia_aligned_5y', 'mean'),
        mean_actual=('actual', 'mean'),
        count=('actual', 'count'),
    ).reset_index()
    print(cal[['mean_predicted', 'mean_actual', 'count']].to_string(index=False))

    # Reference: existing metrics
    print(f"\n{'='*70}")
    print("Already-reported metrics (from V3 eval):")
    print("  C_td dementia: 0.7685   |  C_td death:    0.9518")
    print("  IBS dementia:  0.1740   |  IBS death:     0.1009")
    print("  INBLL dementia: 0.5101  |  INBLL death:   0.3319")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
