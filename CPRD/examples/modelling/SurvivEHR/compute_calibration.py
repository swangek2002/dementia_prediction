"""
compute_calibration.py
======================
Compute calibration plot + calibration slope using APPROACH A (model native timeframe).

Approach A: For each timepoint t, define
  - predicted CIF: from inference CSV (cif_dementia_Xy column)
  - observed label: y_i(t) = 1[(T_i - p_i) <= t]   i.e., event distance from prediction point <= t
    where T_i - p_i is event_time_scaled * SUPERVISED_TIME_SCALE
    (event_time_scaled is the model's scaled time delta from prediction point to outcome)

This evaluates the model's probability calibration in its NATIVE timeframe.
This is the clinically correct paradigm for dynamic prediction — see
PROJECT_KNOWLEDGE.md Section 11.6 and PROGRESS_REPORT.md Section 7.3.

Usage:
    python compute_calibration.py --input <inference_csv> --output <output_dir> [--label <label>]

Inference CSV must contain columns:
  - patient_id, label, event_time_scaled
  - cif_dementia_1y, cif_dementia_2y, cif_dementia_3y, cif_dementia_5y
  - cif_death_1y,    cif_death_2y,    cif_death_3y,    cif_death_5y
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

SUPERVISED_TIME_SCALE = 5.0
N_BINS = 10
TIMEPOINTS = [1, 2, 3, 5]
RISKS = ['dementia', 'death']


def compute_calibration_at_t(df, risk, t_years):
    """Compute calibration data for a single (risk, t) combination.

    Approach A:
      - predicted: cif_{risk}_{t}y column
      - observed:  y_i = 1[event happened (as risk) within t years from prediction point]
                       = 1[label == risk AND event_time_scaled * SUPERVISED_TIME_SCALE <= t]
      - For censored / wrong-cause events that happened AFTER t: they're "negatives"
      - For censored events BEFORE t: they're "unknown" but conservatively treated as "negative"
        (a more sophisticated treatment would use IPCW; this is a simplification)

    Returns:
        bins_df: per-bin summary (mean_pred, observed_rate, count)
        slope: calibration slope from logistic regression (on logit scale)
        intercept: calibration intercept
        score_summary: dict of (n_cases, n_controls, n_unknown, mean_pred_all, mean_obs_all)
    """
    pred_col = f'cif_{risk}_{t_years}y'
    if pred_col not in df.columns:
        raise ValueError(f"Missing column {pred_col} in input CSV")

    # Time from prediction point to outcome (real years)
    event_dist_yr = df['event_time_scaled'] * SUPERVISED_TIME_SCALE

    # Define event status under Approach A
    is_target_event = (df['label'] == risk) & (event_dist_yr <= t_years)
    is_other_event_before_t = (df['label'] != risk) & (df['label'] != 'censored') & (event_dist_yr <= t_years)
    is_censored_before_t = (df['label'] == 'censored') & (event_dist_yr <= t_years)

    # For Approach A:
    # - cases = is_target_event (e.g., dementia within t years from prediction point)
    # - controls = all patients with event_dist_yr > t_years (definitely event-free at t for the target risk)
    # - tricky group = censored before t (their outcome at t is unknown)
    # - For simplicity here, exclude unknowns (censored before t for any risk) from calibration:
    valid = (event_dist_yr > t_years) | is_target_event | is_other_event_before_t
    # Other-cause events before t are treated as "did not have target event by t" (negatives)
    # Censored before t (unknown) are excluded
    # event-free at t and target event before t are kept

    df_v = df[valid].copy()
    y = ((df_v['label'] == risk) & (event_dist_yr[valid] <= t_years)).astype(int).values
    yhat = df_v[pred_col].values

    n_cases = int(y.sum())
    n_controls = int((1 - y).sum())
    n_excluded = int((~valid).sum())

    if n_cases < 5 or n_controls < 5:
        print(f"  [skip] {risk}@{t_years}y: too few cases ({n_cases}) or controls ({n_controls})")
        return None, None, None, None

    # Bin by predicted probability (quantile bins)
    try:
        bins = pd.qcut(yhat, q=N_BINS, duplicates='drop', labels=False)
    except Exception:
        bins = np.zeros_like(yhat, dtype=int)

    df_v['_pred'] = yhat
    df_v['_y'] = y
    df_v['_bin'] = bins

    bin_summary = df_v.groupby('_bin').agg(
        mean_pred=('_pred', 'mean'),
        observed_rate=('_y', 'mean'),
        count=('_y', 'count'),
    ).reset_index()

    # Calibration slope via logistic regression
    # logit(P(y=1)) = alpha + beta * logit(yhat)
    eps = 1e-6
    yhat_clip = np.clip(yhat, eps, 1 - eps)
    logit_yhat = np.log(yhat_clip / (1 - yhat_clip))
    try:
        # Sklearn LogisticRegression solves: logit(p) = alpha + beta * x
        lr = LogisticRegression(C=1e12, fit_intercept=True, max_iter=1000)
        lr.fit(logit_yhat.reshape(-1, 1), y)
        slope = float(lr.coef_[0, 0])
        intercept = float(lr.intercept_[0])
    except Exception as e:
        print(f"  [warn] {risk}@{t_years}y: logistic regression failed: {e}")
        slope, intercept = float('nan'), float('nan')

    score_summary = {
        'n_cases': n_cases,
        'n_controls': n_controls,
        'n_excluded': n_excluded,
        'mean_pred_all': float(yhat.mean()),
        'observed_rate_all': float(y.mean()),
    }
    return bin_summary, slope, intercept, score_summary


def plot_calibration(bins_df_dict, output_path, model_label):
    """Plot multi-timepoint calibration on a single figure (one subplot per risk)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax_idx, risk in enumerate(RISKS):
        ax = axes[ax_idx]
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect (y=x)')

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for ci, t in enumerate(TIMEPOINTS):
            key = f'{risk}_{t}y'
            bins_df = bins_df_dict.get(key)
            if bins_df is None or len(bins_df) == 0:
                continue
            ax.plot(
                bins_df['mean_pred'].values,
                bins_df['observed_rate'].values,
                marker='o',
                color=colors[ci],
                label=f'@{t}y',
                linewidth=1.5,
            )
        ax.set_xlabel('Mean predicted probability (model native timeframe)')
        ax.set_ylabel('Observed rate')
        ax.set_title(f'{risk.capitalize()} calibration — {model_label}')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to test inference CSV')
    parser.add_argument('--output-dir', required=True, help='Directory to write plots / CSV')
    parser.add_argument('--label', default='model', help='Label for plot title (e.g., V4, V2_ablation)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} test patients from {args.input}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

    print(f"\n{'='*70}")
    print(f"Calibration analysis (Approach A: model native timeframe)")
    print(f"  Model: {args.label}")
    print(f"  Output dir: {args.output_dir}")
    print(f"{'='*70}")

    bins_df_dict = {}
    slope_table = []

    for risk in RISKS:
        for t in TIMEPOINTS:
            print(f"\n— {risk}@{t}y —")
            bins_df, slope, intercept, score_summary = compute_calibration_at_t(df, risk, t)
            if bins_df is None:
                continue
            bins_df_dict[f'{risk}_{t}y'] = bins_df
            print(f"  n_cases={score_summary['n_cases']}, n_controls={score_summary['n_controls']}, excluded={score_summary['n_excluded']}")
            print(f"  mean predicted={score_summary['mean_pred_all']:.4f}, mean observed={score_summary['observed_rate_all']:.4f}")
            print(f"  Calibration slope: {slope:.4f}  (1.0 = perfect)")
            print(f"  Calibration intercept: {intercept:.4f}  (0.0 = perfect on logit scale)")
            slope_table.append({
                'risk': risk,
                't_years': t,
                'n_cases': score_summary['n_cases'],
                'n_controls': score_summary['n_controls'],
                'n_excluded': score_summary['n_excluded'],
                'mean_pred': score_summary['mean_pred_all'],
                'observed_rate': score_summary['observed_rate_all'],
                'calibration_slope': slope,
                'calibration_intercept': intercept,
            })

    # Save outputs
    plot_path = os.path.join(args.output_dir, f'calibration_{args.label}.png')
    plot_calibration(bins_df_dict, plot_path, args.label)

    slope_df = pd.DataFrame(slope_table)
    slope_csv = os.path.join(args.output_dir, f'calibration_slopes_{args.label}.csv')
    slope_df.to_csv(slope_csv, index=False)
    print(f"\n  Saved slopes to {slope_csv}")

    # Per-bin details
    bins_csv = os.path.join(args.output_dir, f'calibration_bins_{args.label}.csv')
    bin_rows = []
    for key, bdf in bins_df_dict.items():
        risk, t = key.rsplit('_', 1)
        for _, row in bdf.iterrows():
            bin_rows.append({
                'risk': risk,
                't': t,
                'bin': row['_bin'],
                'mean_pred': row['mean_pred'],
                'observed_rate': row['observed_rate'],
                'count': row['count'],
            })
    if bin_rows:
        pd.DataFrame(bin_rows).to_csv(bins_csv, index=False)
        print(f"  Saved bin details to {bins_csv}")

    print("\n=== Summary table ===")
    if not slope_df.empty:
        print(slope_df[['risk', 't_years', 'n_cases', 'n_controls', 'calibration_slope', 'calibration_intercept']].to_string(index=False))


if __name__ == '__main__':
    main()
