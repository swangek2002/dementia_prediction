"""
compute_additional_metrics.py
=============================
Compute AUROC@Ny, Top-K precision, Harrell's C, and calibration data
from the test set CIF CSV produced by inference_test_metrics.py.

No GPU needed — pure pandas/numpy/sklearn.

Usage:
    cd /Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR
    PYTHONPATH=/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD \
    /Data0/swangek_data/conda_envs/survivehr/bin/python compute_additional_metrics.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index

INPUT_CSV = "/Data0/swangek_data/991/CPRD/data/test_cif_v3.csv"


def compute_auroc_at_year(df, year, risk="dementia", score_col=None):
    """Compute cumulative/dynamic AUROC at a given time horizon.
    Cases: event of interest before year (from index date).
    Controls: event-free at year (still alive, no dementia at year from index).
    Exclude: competing events or censored before year.

    score_col: override the risk score column. If None, uses cif_{risk}_{year}y.
    """
    t_star = float(year)
    if score_col is None:
        score_col = f"cif_{risk}_{year}y"

    if risk == "dementia":
        cases = (df['label'] == 'dementia') & (df['event_time_from_index_years'] <= t_star)
    else:
        cases = (df['label'] == 'death') & (df['event_time_from_index_years'] <= t_star)

    controls = df['event_time_from_index_years'] > t_star  # still event-free at t_star

    mask = cases | controls
    n_cases = cases[mask].sum()
    n_controls = controls[mask].sum()

    if n_cases == 0 or n_controls == 0:
        return np.nan, 0, 0

    y_true = cases[mask].astype(int).values
    y_score = df.loc[mask, score_col].values

    auroc = roc_auc_score(y_true, y_score)
    return auroc, int(n_cases), int(n_controls)


def compute_topk_precision(df, year=5, percentiles=[0.01, 0.05, 0.10]):
    """Top-K precision: among top K% by predicted risk, how many actually got dementia within year."""
    cif_col = f"cif_dementia_{year}y"
    df_sorted = df.sort_values(cif_col, ascending=False)

    results = {}
    for pct in percentiles:
        k = int(len(df_sorted) * pct)
        top_k = df_sorted.head(k)
        true_positives = ((top_k['label'] == 'dementia') &
                          (top_k['event_time_from_index_years'] <= float(year))).sum()
        precision = true_positives / k if k > 0 else 0
        results[pct] = (precision, int(true_positives), k)

    return results


def compute_harrells_c(df, risk="dementia", score_col=None):
    """Cause-specific Harrell's C-index.

    Note: lifelines concordance_index expects higher predicted score = longer
    survival time (less risk). Since CIF is a risk score (higher = more risk),
    we negate it so that higher -> shorter event time.
    """
    if risk == "dementia":
        mask = df['label'].isin(['dementia', 'censored'])
        df_sub = df[mask]
        event_observed = (df_sub['label'] == 'dementia').values
        if score_col is None:
            score_col = 'cif_dementia_5y'
        predicted = df_sub[score_col].values
    else:
        mask = df['label'].isin(['death', 'censored'])
        df_sub = df[mask]
        event_observed = (df_sub['label'] == 'death').values
        if score_col is None:
            score_col = 'cif_death_5y'
        predicted = df_sub[score_col].values

    c_index = concordance_index(
        event_times=df_sub['event_time_from_index_years'].values,
        predicted_scores=-predicted,  # negate: higher CIF = higher risk = shorter time
        event_observed=event_observed,
    )
    return c_index, len(df_sub), int(event_observed.sum())


def compute_calibration(df, year=5, n_bins=10):
    """Calibration: predicted vs observed event rate in quantile bins."""
    # Include: patients with known 5y outcome
    # (event before 5y OR follow-up > 5y)
    mask_cal = (df['event_time_from_index_years'] > float(year)) | \
               ((df['label'] == 'dementia') & (df['event_time_from_index_years'] <= float(year)))

    df_cal = df[mask_cal].copy()
    df_cal['actual'] = ((df_cal['label'] == 'dementia') &
                        (df_cal['event_time_from_index_years'] <= float(year))).astype(int)

    cif_col = f"cif_dementia_{year}y"
    df_cal['pred_bin'] = pd.qcut(df_cal[cif_col], q=n_bins, duplicates='drop')

    calibration = df_cal.groupby('pred_bin', observed=True).agg(
        mean_predicted=(cif_col, 'mean'),
        mean_actual=('actual', 'mean'),
        count=('actual', 'count')
    ).reset_index()

    return calibration


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} patients from {INPUT_CSV}")
    print(f"  Labels: {df['label'].value_counts().to_dict()}")

    # Compute prediction_age (age at model's prediction point = 2nd-to-last event)
    df['prediction_age'] = (72.0 + df['event_time_from_index_years']) - df['event_time_scaled'] * 5.0

    print(f"\n{'='*60}")
    print("=== V3 Model Additional Evaluation Metrics ===")
    print(f"{'='*60}")

    # --- CIF Saturation Diagnostic ---
    print("\nCIF Saturation Diagnostic (CIF_dementia median by label):")
    for year in [1, 2, 3, 5]:
        col = f'cif_dementia_{year}y'
        vals = {l: df[df['label']==l][col].median() for l in ['dementia','death','censored']}
        print(f"  @{year}y: dem={vals['dementia']:.4f}, death={vals['death']:.4f}, cens={vals['censored']:.4f}")
    print("  Note: CIF saturates near 1.0 at 5y, reducing discrimination.")
    print("  The model's time scale (5y) causes the CDF component to approach 1.0.")

    # --- AUROC using CIF at model horizon ---
    print("\nAUROC (risk score = CIF at model time horizon):")
    for year in [1, 2, 3, 5]:
        auroc, nc, nctl = compute_auroc_at_year(df, year, "dementia")
        suffix = ""
        if year == 5:
            suffix = "  (cf. UKBDRS=0.80, Wang=0.810, Botz=0.776)"
        print(f"  Dementia @{year}y: {auroc:.4f}  (cases={nc}, controls={nctl}){suffix}")

    auroc_death, nc_d, nctl_d = compute_auroc_at_year(df, 5, "death")
    print(f"  Death @5y:    {auroc_death:.4f}  (cases={nc_d}, controls={nctl_d})  (cf. Gu=0.866)")

    # --- AUROC@5y using earlier CIF as risk score ---
    # Since CIF@5y is saturated, earlier time points may provide better ranking
    print("\nAUROC@5y with alternative risk scores:")
    for score_year in [1, 2, 3]:
        score_col = f'cif_dementia_{score_year}y'
        auroc_alt, nc, nctl = compute_auroc_at_year(df, 5, "dementia", score_col=score_col)
        print(f"  Using CIF@{score_year}y: {auroc_alt:.4f}  (cases={nc}, controls={nctl})")

    # Death AUROC@5y with alternative scores
    for score_year in [1, 2, 3]:
        score_col = f'cif_death_{score_year}y'
        auroc_alt, nc, nctl = compute_auroc_at_year(df, 5, "death", score_col=score_col)
        print(f"  Death using CIF@{score_year}y: {auroc_alt:.4f}  (cases={nc}, controls={nctl})")

    # --- Top-K Precision ---
    print("\nTop-K Precision @5y (using CIF_dementia_5y):")
    topk = compute_topk_precision(df, year=5)
    for pct in [0.01, 0.05, 0.10]:
        prec, tp, k = topk[pct]
        suffix = ""
        if pct == 0.01:
            suffix = "  (cf. DemRisk=37%, NYU EHR-BERT=39.2%)"
        print(f"  Top {pct*100:.0f}%:  {prec*100:.1f}% ({tp}/{k}){suffix}")

    # Top-K using CIF@2y as risk score (less saturated)
    print("\nTop-K Precision @5y (using CIF_dementia_2y as risk score):")
    df_sorted_2y = df.sort_values('cif_dementia_2y', ascending=False)
    for pct in [0.01, 0.05, 0.10]:
        k = int(len(df_sorted_2y) * pct)
        top_k = df_sorted_2y.head(k)
        tp = ((top_k['label'] == 'dementia') & (top_k['event_time_from_index_years'] <= 5.0)).sum()
        precision = tp / k if k > 0 else 0
        suffix = ""
        if pct == 0.01:
            suffix = "  (cf. DemRisk=37%, NYU EHR-BERT=39.2%)"
        print(f"  Top {pct*100:.0f}%:  {precision*100:.1f}% ({tp}/{k}){suffix}")

    # --- Harrell's C ---
    print("\nHarrell's C (cause-specific):")
    for score_year in [1, 2, 3, 5]:
        score_col_dem = f'cif_dementia_{score_year}y'
        c_dem, n_dem, e_dem = compute_harrells_c(df, "dementia", score_col=score_col_dem)
        suffix = "  (cf. Yuan=0.749)" if score_year == 5 else ""
        print(f"  Dementia (CIF@{score_year}y): {c_dem:.4f}  (n={n_dem}, events={e_dem}){suffix}")

    for score_year in [1, 2, 3, 5]:
        score_col_dth = f'cif_death_{score_year}y'
        c_death, n_death, e_death = compute_harrells_c(df, "death", score_col=score_col_dth)
        print(f"  Death    (CIF@{score_year}y): {c_death:.4f}  (n={n_death}, events={e_death})")

    # --- Calibration ---
    print("\nCalibration @5y (dementia, CIF_dementia_5y):")
    cal = compute_calibration(df, year=5)
    print(cal[['mean_predicted', 'mean_actual', 'count']].to_string(index=False))

    # --- Already-reported metrics ---
    print(f"\n{'='*60}")
    print("Already-reported metrics (from eval):")
    print("  C_td dementia: 0.7685")
    print("  C_td death:    0.9518")
    print("  IBS dementia:  0.1740")
    print("  IBS death:     0.1009")
    print("  INBLL dementia: 0.5101")
    print("  INBLL death:   0.3319")

    # --- Key observations ---
    print(f"\n{'='*60}")
    print("KEY OBSERVATIONS:")
    print("  1. CIF values saturate near 1.0 at 5y (model time t=1.0),")
    print("     because the cause-specific CDFs F_k(t) approach 1.0 at the")
    print("     end of the model's time range (supervised_time_scale=5.0).")
    print("  2. This saturation degrades AUROC@5y and Top-K precision@5y.")
    print("  3. Earlier time horizons (1-2y) show much better discrimination.")
    print("  4. C_td (0.7685) evaluates at each patient's actual event time,")
    print("     where CIF values have not yet saturated, giving a fairer picture.")
    print("  5. The model is a DYNAMIC predictor (conditioned on full history")
    print("     up to last observation). Literature baselines (UKBDRS etc.)")
    print("     predict from a single assessment. Direct comparison requires caveats.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
