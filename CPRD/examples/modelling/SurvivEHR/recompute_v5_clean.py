"""
recompute_v5_clean.py
======================
Re-evaluate V5 metrics on test set with the 16 prevalent-leaky patients excluded.

These are patients with pre-index GP-coded dementia in their input sequence
(missed by V2's HES-only prevalent removal).

Outputs cleaned metrics + delta vs. original.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

INPUT_CSV = "/Data0/swangek_data/991/CPRD/data/test_cif_v5.csv"
LEAKY_TXT = "/Data0/swangek_data/991/CPRD/data/leaky_patients_test.txt"
OUTPUT_CSV = "/Data0/swangek_data/991/CPRD/data/test_cif_v5_clean.csv"
SUPERVISED_TIME_SCALE = 5.0

df = pd.read_csv(INPUT_CSV)
leaky = set(int(x.strip()) for x in open(LEAKY_TXT) if x.strip())
print(f"Loaded {len(df)} test patients; {len(leaky)} leaky to exclude")

mask_leaky = df["patient_id"].isin(leaky)
print(f"  Found {mask_leaky.sum()}/{len(leaky)} leaky in CSV")
print(f"  Leaky label distribution: {df.loc[mask_leaky, 'label'].value_counts().to_dict()}")

df_clean = df.loc[~mask_leaky].reset_index(drop=True)
print(f"  Clean cohort: {len(df_clean)}")
df_clean.to_csv(OUTPUT_CSV, index=False)
print(f"  Saved cleaned CSV to {OUTPUT_CSV}")


# ---- Metrics on cleaned cohort vs original ----
def compute_metrics(dfx, label):
    print(f"\n{'='*60}")
    print(f"Metrics: {label}  (n={len(dfx)})")
    print(f"{'='*60}")
    print(f"  Label distribution: {dfx['label'].value_counts().to_dict()}")

    event_dist_yr = dfx["event_time_scaled"] * SUPERVISED_TIME_SCALE

    for risk in ["dementia", "death"]:
        for t in [1, 2, 3, 5]:
            pred_col = f"cif_{risk}_{t}y"
            is_target = (dfx["label"] == risk) & (event_dist_yr <= t)
            is_other = (dfx["label"] != risk) & (dfx["label"] != "censored") & (event_dist_yr <= t)
            valid = (event_dist_yr > t) | is_target | is_other
            d = dfx.loc[valid].copy()
            y = ((d["label"] == risk) & (event_dist_yr.loc[valid] <= t)).astype(int).values
            yhat = d[pred_col].values
            if y.sum() < 5 or (1 - y).sum() < 5:
                continue
            auc = roc_auc_score(y, yhat)
            # Calibration slope
            eps = 1e-6
            yc = np.clip(yhat, eps, 1 - eps)
            lz = np.log(yc / (1 - yc))
            try:
                lr = LogisticRegression(C=1e12, fit_intercept=True, max_iter=1000)
                lr.fit(lz.reshape(-1, 1), y)
                slope = float(lr.coef_[0, 0])
                intercept = float(lr.intercept_[0])
            except Exception:
                slope = intercept = float("nan")
            print(f"  {risk:9s}@{t}y  AUROC={auc:.4f}  slope={slope:.3f}  intercept={intercept:.3f}  "
                  f"(cases={int(y.sum())}, controls={int((1-y).sum())})")

    # Top-K precision @5y for dementia (Approach A — within model time)
    print(f"\n  Top-K precision dementia @5y (model native time):")
    df_sorted = dfx.sort_values("cif_dementia_5y", ascending=False)
    for pct in [0.01, 0.05, 0.10]:
        k = int(len(df_sorted) * pct)
        top_k = df_sorted.head(k)
        edist = top_k["event_time_scaled"] * SUPERVISED_TIME_SCALE
        tp = ((top_k["label"] == "dementia") & (edist <= 5.0)).sum()
        print(f"    Top {pct*100:.0f}% (k={k}): {tp/k*100:.1f}% ({tp}/{k})")


compute_metrics(df, "ORIGINAL (with leaky)")
compute_metrics(df_clean, "CLEANED (leaky removed)")

# ---- Direct comparison: per-patient look at the 16 leaky ----
print(f"\n{'='*60}")
print("Per-patient CIF stats for the 16 leaky test patients:")
print(f"{'='*60}")
leaky_rows = df.loc[mask_leaky, ["patient_id", "label", "event_time_scaled",
                                  "cif_dementia_5y", "cif_death_5y"]]
print(leaky_rows.to_string(index=False))
print(f"\n  Mean CIF_dementia@5y of leaky: {leaky_rows['cif_dementia_5y'].mean():.4f}")
print(f"  Mean CIF_dementia@5y of full:  {df['cif_dementia_5y'].mean():.4f}")
print(f"  Mean CIF_dementia@5y of clean: {df_clean['cif_dementia_5y'].mean():.4f}")
