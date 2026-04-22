"""
build_hes_summary_features.py
==============================
Extract per-patient HES summary statistics for use as static covariates.

Produces 8 features per patient:
  1. HES_TOTAL_ADMISSIONS    - log-scaled total hospital admissions
  2. HES_TOTAL_UNIQUE_DIAG   - log-scaled unique ICD-10 diagnosis count
  3. HES_HAS_STROKE          - binary flag (I60-I69)
  4. HES_HAS_MI              - binary flag (I21-I22)
  5. HES_HAS_HEART_FAILURE   - binary flag (I50)
  6. HES_HAS_DIABETES        - binary flag (E10-E14)
  7. HES_HAS_DELIRIUM        - binary flag (F05)
  8. HES_HAS_TBI             - binary flag (S06)

Dementia codes (F00-F03, G30) are EXCLUDED to avoid label leakage,
since hes_aug already uses them for label augmentation.

Output: data/hes_summary_features.pickle -> {patient_id: np.array([8 floats])}
"""

import math
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

HESIN_CSV = "/Data0/swangek_data/991/CPRD/data/hesin.csv"
HESIN_DIAG_CSV = "/Data0/swangek_data/991/CPRD/data/hesin_diag.csv"
OUTPUT_PATH = "/Data0/swangek_data/991/CPRD/data/hes_summary_features.pickle"

# Feature column names (must start with HES_ for dataloader detection)
FEATURE_NAMES = [
    "HES_TOTAL_ADMISSIONS",
    "HES_TOTAL_UNIQUE_DIAG",
    "HES_HAS_STROKE",
    "HES_HAS_MI",
    "HES_HAS_HEART_FAILURE",
    "HES_HAS_DIABETES",
    "HES_HAS_DELIRIUM",
    "HES_HAS_TBI",
]
NUM_FEATURES = len(FEATURE_NAMES)

# Comorbidity ICD-10 prefix definitions
COMORBIDITY_PREFIXES = {
    "stroke":        ["I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69"],
    "mi":            ["I21", "I22"],
    "heart_failure": ["I50"],
    "diabetes":      ["E10", "E11", "E12", "E13", "E14"],
    "delirium":      ["F05"],
    "tbi":           ["S06"],
}

# Dementia codes to EXCLUDE (used for labels, not features)
DEMENTIA_PREFIXES = ["F00", "F01", "F02", "G30"]
DEMENTIA_EXACT = {"F03"}


def _is_dementia(code: str) -> bool:
    for p in DEMENTIA_PREFIXES:
        if code.startswith(p):
            return True
    return code in DEMENTIA_EXACT


def main():
    print("=" * 60)
    print("  Building HES Summary Features")
    print("=" * 60)

    # Step 1: Count admissions per patient from hesin.csv
    print("\nStep 1: Counting admissions per patient from hesin.csv ...")
    hesin = pd.read_csv(
        HESIN_CSV,
        usecols=["dnx_hesin_id", "eid"],
        dtype={"dnx_hesin_id": str, "eid": str},
    )
    # Count unique admissions (unique dnx_hesin_id) per patient
    admission_counts = hesin.groupby("eid")["dnx_hesin_id"].nunique()
    print(f"  {len(admission_counts)} patients with HES records")
    print(f"  Admission count: mean={admission_counts.mean():.1f}, "
          f"median={admission_counts.median():.0f}, max={admission_counts.max()}")

    # Step 2: Process diagnoses from hesin_diag.csv
    print("\nStep 2: Processing diagnoses from hesin_diag.csv ...")
    diag = pd.read_csv(
        HESIN_DIAG_CSV,
        usecols=["eid", "diag_icd10"],
        dtype={"eid": str, "diag_icd10": str},
    )
    diag = diag.dropna(subset=["diag_icd10"])
    diag["diag_icd10"] = diag["diag_icd10"].str.strip()
    print(f"  {len(diag)} diagnosis records with valid ICD-10 codes")

    # Per-patient: unique diagnoses (excluding dementia) and comorbidity flags
    patient_unique_diag = defaultdict(set)
    patient_comorbidities = defaultdict(lambda: {k: False for k in COMORBIDITY_PREFIXES})

    for eid, icd10 in zip(diag["eid"], diag["diag_icd10"]):
        if _is_dementia(icd10):
            continue
        patient_unique_diag[eid].add(icd10)
        for comorbidity, prefixes in COMORBIDITY_PREFIXES.items():
            if any(icd10.startswith(p) for p in prefixes):
                patient_comorbidities[eid][comorbidity] = True

    # Step 3: Build feature vectors
    print("\nStep 3: Building feature vectors ...")
    all_patients = set(admission_counts.index) | set(patient_unique_diag.keys())

    features = {}
    for eid in all_patients:
        pid = int(eid)

        # Continuous features (log-scaled, capped at 1.0)
        n_admissions = admission_counts.get(eid, 0)
        n_unique_diag = len(patient_unique_diag.get(eid, set()))

        feat = np.zeros(NUM_FEATURES, dtype=np.float32)
        feat[0] = min(math.log1p(n_admissions) / math.log(51), 1.0)   # HES_TOTAL_ADMISSIONS
        feat[1] = min(math.log1p(n_unique_diag) / math.log(101), 1.0) # HES_TOTAL_UNIQUE_DIAG

        # Binary comorbidity flags
        comorb = patient_comorbidities.get(eid, {})
        feat[2] = float(comorb.get("stroke", False))
        feat[3] = float(comorb.get("mi", False))
        feat[4] = float(comorb.get("heart_failure", False))
        feat[5] = float(comorb.get("diabetes", False))
        feat[6] = float(comorb.get("delirium", False))
        feat[7] = float(comorb.get("tbi", False))

        features[pid] = feat

    # Step 4: Print statistics
    print(f"\n  Total patients: {len(features)}")
    all_feats = np.stack(list(features.values()))
    for i, name in enumerate(FEATURE_NAMES):
        col = all_feats[:, i]
        nonzero_frac = (col > 0).mean()
        print(f"  {name:30s}  mean={col.mean():.4f}  std={col.std():.4f}  "
              f"nonzero={nonzero_frac:.3f}")

    # Step 5: Save
    print(f"\nSaving to {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"features": features, "feature_names": FEATURE_NAMES}, f)
    print("Done.")


if __name__ == "__main__":
    main()
