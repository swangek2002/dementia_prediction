"""
hes_dementia_lookup.py
======================
Builds a lookup table of HES (Hospital Episode Statistics) dementia diagnoses.
Returns a dict: {patient_id (int): earliest_hes_dementia_date (datetime)}.

HES uses ICD-10 codes for dementia:
  - F00* (Dementia in Alzheimer's disease)
  - F01* (Vascular dementia)
  - F02* (Dementia in other diseases)
  - F03  (Unspecified dementia)
  - G30* (Alzheimer's disease)
"""

import pandas as pd
from pathlib import Path

HESIN_CSV = "/Data0/swangek_data/991/CPRD/data/hesin.csv"
HESIN_DIAG_CSV = "/Data0/swangek_data/991/CPRD/data/hesin_diag.csv"

HES_DEMENTIA_ICD10_PREFIXES = ["F00", "F01", "F02", "G30"]
HES_DEMENTIA_ICD10_EXACT = ["F03"]


def _is_dementia_icd10(code: str) -> bool:
    """Check if an ICD-10 code is a dementia diagnosis."""
    if pd.isna(code):
        return False
    code = str(code).strip()
    for prefix in HES_DEMENTIA_ICD10_PREFIXES:
        if code.startswith(prefix):
            return True
    return code in HES_DEMENTIA_ICD10_EXACT


def build_hes_dementia_lookup(
    hesin_csv: str = HESIN_CSV,
    hesin_diag_csv: str = HESIN_DIAG_CSV,
) -> dict:
    """
    Build a lookup dict: {patient_id (int): earliest_hes_dementia_date (pd.Timestamp)}.

    Steps:
      1. Read hesin.csv -> build dnx_hesin_id -> admidate mapping
      2. Read hesin_diag.csv -> filter for dementia ICD-10 codes
      3. Join on dnx_hesin_id to get admission dates
      4. Group by eid, take earliest date
    """
    print("Loading HES data...")

    # Step 1: Load admission dates from hesin.csv
    hesin = pd.read_csv(
        hesin_csv,
        usecols=["dnx_hesin_id", "eid", "admidate"],
        dtype={"dnx_hesin_id": str, "eid": str},
    )
    hesin["admidate"] = pd.to_datetime(hesin["admidate"], errors="coerce")
    hesin = hesin.dropna(subset=["admidate"])
    print(f"  hesin.csv: {len(hesin)} records with valid admidate")

    # Step 2: Load diagnosis codes, filter for dementia
    diag = pd.read_csv(
        hesin_diag_csv,
        usecols=["dnx_hesin_id", "eid", "diag_icd10"],
        dtype={"dnx_hesin_id": str, "eid": str, "diag_icd10": str},
    )
    dementia_mask = diag["diag_icd10"].apply(_is_dementia_icd10)
    diag_dementia = diag[dementia_mask].copy()
    print(f"  hesin_diag.csv: {len(diag_dementia)} dementia diagnosis records "
          f"(from {len(diag)} total)")

    # Step 3: Join to get admission dates for dementia diagnoses
    merged = diag_dementia.merge(
        hesin[["dnx_hesin_id", "admidate"]],
        on="dnx_hesin_id",
        how="inner",
    )
    print(f"  After join: {len(merged)} records with dates")

    # Step 4: Group by patient, take earliest dementia date
    merged["eid_int"] = merged["eid"].astype(int)
    earliest = merged.groupby("eid_int")["admidate"].min()
    lookup = earliest.to_dict()
    print(f"  Unique patients with HES dementia: {len(lookup)}")

    return lookup


if __name__ == "__main__":
    lookup = build_hes_dementia_lookup()
    print(f"\nTotal patients with HES dementia diagnosis: {len(lookup)}")
    # Show a few examples
    for pid, date in list(lookup.items())[:5]:
        print(f"  Patient {pid}: earliest HES dementia = {date}")
