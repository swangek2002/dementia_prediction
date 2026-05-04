"""
build_hes_summary_features.py
==============================
Extract per-patient HES summary statistics for use as static covariates.

Produces 22 features per patient (using only HES records BEFORE index date
to avoid temporal leakage):
  --- Original 8 ---
  0. HES_TOTAL_ADMISSIONS        - log-scaled total hospital admissions
  1. HES_TOTAL_UNIQUE_DIAG       - log-scaled unique ICD-10 diagnosis count
  2. HES_HAS_STROKE              - binary flag (I60-I69)
  3. HES_HAS_MI                  - binary flag (I21-I22)
  4. HES_HAS_HEART_FAILURE       - binary flag (I50)
  5. HES_HAS_DIABETES            - binary flag (E10-E14)
  6. HES_HAS_DELIRIUM            - binary flag (F05)
  7. HES_HAS_TBI                 - binary flag (S06)
  --- New 11 comorbidities ---
  8.  HES_HAS_HYPERTENSION       - binary flag (I10-I15)
  9.  HES_HAS_ATRIAL_FIBRILLATION - binary flag (I48)
  10. HES_HAS_CKD                - binary flag (N18)
  11. HES_HAS_DEPRESSION         - binary flag (F32, F33)
  12. HES_HAS_PARKINSON          - binary flag (G20)
  13. HES_HAS_EPILEPSY           - binary flag (G40, G41)
  14. HES_HAS_OBESITY             - binary flag (E66)
  15. HES_HAS_HYPERLIPIDEMIA     - binary flag (E78)
  16. HES_HAS_COPD               - binary flag (J44)
  17. HES_HAS_ALCOHOL             - binary flag (F10)
  18. HES_HAS_SLEEP_DISORDER     - binary flag (G47)
  --- New 3 continuous ---
  19. HES_MEAN_STAY_DAYS         - log-scaled mean length of stay
  20. HES_EMERGENCY_RATIO        - emergency admission ratio [0,1]
  21. HES_YEARS_SINCE_LAST_ADMISSION - scaled years since last admission before index

Dementia codes (F00-F03, G30) are EXCLUDED to avoid label leakage.
Only HES records with admidate BEFORE each patient's index date are used.

Output: data/hes_summary_features.pickle -> {patient_id: np.array([22 floats])}
"""

import math
import pickle
import sqlite3
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

HESIN_CSV = "/Data0/swangek_data/991/CPRD/data/hesin.csv"
HESIN_DIAG_CSV = "/Data0/swangek_data/991/CPRD/data/hesin_diag.csv"
OUTPUT_PATH = "/Data0/swangek_data/991/CPRD/data/hes_summary_features.pickle"
PATH_TO_DB = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"

INDEX_ON_AGE = 72

# ============================================================
# Feature definitions (22 features total)
# ============================================================
FEATURE_NAMES = [
    # --- Original 8 ---
    "HES_TOTAL_ADMISSIONS",         # 0: continuous, log-scaled
    "HES_TOTAL_UNIQUE_DIAG",        # 1: continuous, log-scaled
    "HES_HAS_STROKE",               # 2: binary
    "HES_HAS_MI",                   # 3: binary
    "HES_HAS_HEART_FAILURE",        # 4: binary
    "HES_HAS_DIABETES",             # 5: binary
    "HES_HAS_DELIRIUM",             # 6: binary
    "HES_HAS_TBI",                  # 7: binary
    # --- New 11 comorbidities ---
    "HES_HAS_HYPERTENSION",         # 8: binary
    "HES_HAS_ATRIAL_FIBRILLATION",  # 9: binary
    "HES_HAS_CKD",                  # 10: binary
    "HES_HAS_DEPRESSION",           # 11: binary
    "HES_HAS_PARKINSON",            # 12: binary
    "HES_HAS_EPILEPSY",             # 13: binary
    "HES_HAS_OBESITY",              # 14: binary
    "HES_HAS_HYPERLIPIDEMIA",       # 15: binary
    "HES_HAS_COPD",                 # 16: binary
    "HES_HAS_ALCOHOL",              # 17: binary
    "HES_HAS_SLEEP_DISORDER",       # 18: binary
    # --- New 3 continuous ---
    "HES_MEAN_STAY_DAYS",           # 19: continuous, log-scaled
    "HES_EMERGENCY_RATIO",          # 20: continuous, [0,1]
    "HES_YEARS_SINCE_LAST_ADMISSION",  # 21: continuous, scaled
]
NUM_FEATURES = len(FEATURE_NAMES)  # 22

# Comorbidity ICD-10 prefix definitions (original + new)
COMORBIDITY_PREFIXES = {
    # Original
    "stroke":               ["I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69"],
    "mi":                   ["I21", "I22"],
    "heart_failure":        ["I50"],
    "diabetes":             ["E10", "E11", "E12", "E13", "E14"],
    "delirium":             ["F05"],
    "tbi":                  ["S06"],
    # New
    "hypertension":         ["I10", "I11", "I12", "I13", "I14", "I15"],
    "atrial_fibrillation":  ["I48"],
    "ckd":                  ["N18"],
    "depression":           ["F32", "F33"],
    "parkinson":            ["G20"],
    "epilepsy":             ["G40", "G41"],
    "obesity":              ["E66"],
    "hyperlipidemia":       ["E78"],
    "copd":                 ["J44"],
    "alcohol":              ["F10"],
    "sleep_disorder":       ["G47"],
}

# Comorbidity name -> feature index mapping (binary features start at index 2)
COMORBIDITY_INDEX = {
    "stroke": 2, "mi": 3, "heart_failure": 4, "diabetes": 5,
    "delirium": 6, "tbi": 7,
    "hypertension": 8, "atrial_fibrillation": 9, "ckd": 10,
    "depression": 11, "parkinson": 12, "epilepsy": 13,
    "obesity": 14, "hyperlipidemia": 15, "copd": 16,
    "alcohol": 17, "sleep_disorder": 18,
}

# Dementia codes to EXCLUDE (used for labels, not features)
DEMENTIA_PREFIXES = ["F00", "F01", "F02", "G30"]
DEMENTIA_EXACT = {"F03"}


def _is_dementia(code: str) -> bool:
    for p in DEMENTIA_PREFIXES:
        if code.startswith(p):
            return True
    return code in DEMENTIA_EXACT


def load_year_of_birth_lookup():
    """Load patient year-of-birth from GP database to compute index dates."""
    conn = sqlite3.connect(PATH_TO_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT PATIENT_ID, YEAR_OF_BIRTH FROM static_table")
    rows = cursor.fetchall()
    conn.close()
    yob_lookup = {}
    for pid, yob_str in rows:
        yob_lookup[int(pid)] = pd.Timestamp(yob_str)
    return yob_lookup


def main():
    print("=" * 60)
    print("  Building HES Summary Features v2 (22 dims)")
    print("  WITH temporal filtering (only pre-index-date records)")
    print("=" * 60)

    # ============================================================
    # Step 0: Load year-of-birth to compute index dates
    # ============================================================
    print("\nStep 0: Loading year-of-birth lookup...")
    yob_lookup = load_year_of_birth_lookup()
    print(f"  {len(yob_lookup)} patients with year-of-birth")

    # Compute index dates: {eid_str: index_date_timestamp}
    index_dates = {}
    for pid, yob in yob_lookup.items():
        index_dates[str(pid)] = yob + pd.DateOffset(years=INDEX_ON_AGE)

    # ============================================================
    # Step 1: Read hesin.csv for admission info
    # ============================================================
    print("\nStep 1: Reading hesin.csv for admission info...")
    hesin = pd.read_csv(
        HESIN_CSV,
        usecols=["dnx_hesin_id", "eid", "admidate", "disdate", "admimeth"],
        dtype={"dnx_hesin_id": str, "eid": str, "admimeth": str},
    )
    print(f"  Total admission records: {len(hesin)}")

    # Parse dates
    hesin["admidate_dt"] = pd.to_datetime(hesin["admidate"], errors="coerce")
    hesin["disdate_dt"] = pd.to_datetime(hesin["disdate"], errors="coerce")

    # --- Temporal filtering: only keep records BEFORE index date ---
    hesin["index_date"] = hesin["eid"].map(index_dates)
    n_before = len(hesin)
    hesin = hesin.dropna(subset=["index_date"])  # patients without yob are dropped
    hesin = hesin[hesin["admidate_dt"] < hesin["index_date"]]
    n_after = len(hesin)
    print(f"  After temporal filtering (admidate < index date): {n_after} / {n_before} records kept ({n_after/n_before*100:.1f}%)")

    # 1a. Admission counts per patient (feature 0)
    admission_counts = hesin.groupby("eid")["dnx_hesin_id"].nunique()
    print(f"  {len(admission_counts)} patients with pre-index HES records")

    # 1b. Mean length of stay (feature 19: HES_MEAN_STAY_DAYS)
    hesin["stay_days"] = (hesin["disdate_dt"] - hesin["admidate_dt"]).dt.days
    hesin.loc[hesin["stay_days"] < 0, "stay_days"] = np.nan  # clean invalid
    mean_stay = hesin.dropna(subset=["stay_days"]).groupby("eid")["stay_days"].mean()

    # 1c. Emergency admission ratio (feature 20: HES_EMERGENCY_RATIO)
    # admimeth: 21-28 = emergency admissions
    hesin["is_emergency"] = hesin["admimeth"].apply(
        lambda x: 1 if str(x)[:2] in ["21", "22", "23", "24", "25", "26", "27", "28"] else 0
    )
    emergency_counts = hesin.groupby("eid")["is_emergency"].sum()

    # 1d. Years since last admission before index date (feature 21)
    # Now relative to index date (not study end)
    last_admission = hesin.dropna(subset=["admidate_dt"]).groupby("eid")["admidate_dt"].max()

    # ============================================================
    # Step 2: Read hesin_diag.csv and filter by admission
    # ============================================================
    print("\nStep 2: Processing diagnoses from hesin_diag.csv ...")
    diag = pd.read_csv(
        HESIN_DIAG_CSV,
        usecols=["eid", "dnx_hesin_id", "diag_icd10"],
        dtype={"eid": str, "dnx_hesin_id": str, "diag_icd10": str},
    )
    diag = diag.dropna(subset=["diag_icd10"])
    diag["diag_icd10"] = diag["diag_icd10"].str.strip()
    print(f"  Total diagnosis records: {len(diag)}")

    # --- Temporal filtering: only keep diagnoses from pre-index admissions ---
    # Build set of (eid, dnx_hesin_id) that are pre-index
    pre_index_admissions = set(zip(hesin["eid"], hesin["dnx_hesin_id"]))
    diag_keys = set(zip(diag["eid"], diag["dnx_hesin_id"]))
    diag_filtered = diag[
        diag.apply(lambda r: (r["eid"], r["dnx_hesin_id"]) in pre_index_admissions, axis=1)
    ]
    print(f"  After temporal filtering: {len(diag_filtered)} / {len(diag)} diagnosis records kept ({len(diag_filtered)/len(diag)*100:.1f}%)")

    # Per-patient: unique diagnoses (excluding dementia) and comorbidity flags
    patient_unique_diag = defaultdict(set)
    patient_comorbidities = defaultdict(lambda: {k: False for k in COMORBIDITY_PREFIXES})

    for eid, icd10 in zip(diag_filtered["eid"], diag_filtered["diag_icd10"]):
        if _is_dementia(icd10):
            continue
        patient_unique_diag[eid].add(icd10)
        for comorbidity, prefixes in COMORBIDITY_PREFIXES.items():
            if any(icd10.startswith(p) for p in prefixes):
                patient_comorbidities[eid][comorbidity] = True

    # ============================================================
    # Step 3: Build 22-dim feature vectors
    # ============================================================
    print("\nStep 3: Building 22-dim feature vectors ...")
    all_patients = set(admission_counts.index) | set(patient_unique_diag.keys())

    features = {}
    for eid in all_patients:
        pid = int(eid)
        feat = np.zeros(NUM_FEATURES, dtype=np.float32)

        # --- Original continuous features ---
        n_admissions = admission_counts.get(eid, 0)
        n_unique_diag = len(patient_unique_diag.get(eid, set()))
        feat[0] = min(math.log1p(n_admissions) / math.log(51), 1.0)
        feat[1] = min(math.log1p(n_unique_diag) / math.log(101), 1.0)

        # --- All comorbidities (binary, index 2-18) ---
        comorb = patient_comorbidities.get(eid, {})
        for comorb_name, feat_idx in COMORBIDITY_INDEX.items():
            feat[feat_idx] = float(comorb.get(comorb_name, False))

        # --- New continuous features ---
        # Feature 19: HES_MEAN_STAY_DAYS
        ms = mean_stay.get(eid, 0.0)
        feat[19] = min(math.log1p(ms) / math.log(31), 1.0)

        # Feature 20: HES_EMERGENCY_RATIO
        total_adm = admission_counts.get(eid, 0)
        emerg_count = emergency_counts.get(eid, 0)
        feat[20] = float(emerg_count / total_adm) if total_adm > 0 else 0.0

        # Feature 21: HES_YEARS_SINCE_LAST_ADMISSION (relative to index date)
        last_adm = last_admission.get(eid, None)
        idx_date = index_dates.get(eid, None)
        if last_adm is not None and pd.notna(last_adm) and idx_date is not None:
            years_since = (idx_date - last_adm).days / 365.25
            feat[21] = min(max(years_since, 0.0) / 20.0, 1.0)
        else:
            feat[21] = 1.0  # no admission record -> maximum distance

        features[pid] = feat

    # ============================================================
    # Step 4: Print statistics
    # ============================================================
    print(f"\n  Total patients: {len(features)}")
    all_feats = np.stack(list(features.values()))
    for i, name in enumerate(FEATURE_NAMES):
        col = all_feats[:, i]
        nonzero_frac = (col > 0).mean()
        print(f"  {name:40s}  mean={col.mean():.4f}  std={col.std():.4f}  "
              f"nonzero={nonzero_frac:.3f}")

    # ============================================================
    # Step 5: Save
    # ============================================================
    print(f"\nSaving to {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"features": features, "feature_names": FEATURE_NAMES}, f)
    print("Done.")


if __name__ == "__main__":
    main()
