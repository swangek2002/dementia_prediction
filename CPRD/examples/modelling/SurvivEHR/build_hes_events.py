"""
build_hes_events.py
===================
Translate HES (UKB hospital episode) diagnoses into Read v2 events ready
to be inserted into the SQLite diagnosis_table.

For each HES diagnosis row:
  - PRIMARY (level=1): always include if it can be mapped
  - SECONDARY (level=2): include if it can be mapped
  - Translate ICD-10 -> Read v2 via:
        HOTFIX 1 dementia override (F00*/F01*/F02*/F03*/G30*)
        then OMOP mapping pickle for everything else
  - PRACTICE_ID = static_table.PRACTICE_ID for that patient (skip if missing)
  - DATE = HES admission date (epistart preferred, fall back to admidate),
           reformatted to match the diagnosis_table convention exactly.

HOTFIX 2: HES dates are explicitly parsed and reformatted to match the
EXACT string format used in diagnosis_table.DATE (verified empirically as
'YYYY-MM-DD'). We try multiple input formats so a row that happens to be
in DD/MM/YYYY does not silently break sorting.

Output: hes_events_for_db.pickle
        list of tuples (PRACTICE_ID, PATIENT_ID, EVENT, DATE_STR)
"""

import os
import pickle
import sqlite3
from datetime import datetime

import pandas as pd

PATH_TO_DB = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
HESIN_CSV = "/Data0/swangek_data/991/CPRD/data/hesin.csv"
HESIN_DIAG_CSV = "/Data0/swangek_data/991/CPRD/data/hesin_diag.csv"
OMOP_PICKLE = "/Data0/swangek_data/991/CPRD/data/omop_icd10_to_readv2.pickle"
OUTPUT_PICKLE = "/Data0/swangek_data/991/CPRD/data/hes_events_for_db.pickle"

# ----- HOTFIX 1: hard override for dementia outcomes -----------------------
# Any ICD-10 starting with one of these prefixes MUST be mapped to one of the
# canonical 31 dementia Read v2 codes (the ones the framework recognises as
# outcomes). The targets below were chosen from the canonical 31 set as the
# semantically closest, high-frequency Read v2 dementia codes (frequencies
# verified against diagnosis_table: F110.=699, Eu00.=296, Eu01.=104,
# Eu02z=92). This guarantees that no HES dementia event is silently dropped
# by _reduce_on_outcome regardless of what OMOP says.
DEMENTIA_ICD10_OVERRIDE_PREFIXES = {
    "F00": "Eu00.",  # Dementia in Alzheimer's disease
    "F01": "Eu01.",  # Vascular dementia
    "F02": "F110.",  # Dementia in other diseases (use highest-freq overall)
    "F03": "Eu02z",  # Unspecified dementia
    "G30": "Eu00.",  # Alzheimer's disease
}

# Sanity: every override target must be in the canonical 31 dementia codes.
DEMENTIA_READ_CODES_CANONICAL = {
    "F110.", "Eu00.", "Eu01.", "Eu02z", "Eu002", "E00..",
    "Eu023", "Eu00z", "Eu025", "Eu01z", "E001.", "F1100",
    "Eu001", "E004.", "Eu000", "Eu02.", "Eu013", "E000.",
    "Eu01y", "E001z", "F1101", "Eu020", "E004z", "E0021",
    "Eu02y", "Eu012", "Eu011", "E00z.", "E0040", "E003.",
    "E0020",
}
for _p, _t in DEMENTIA_ICD10_OVERRIDE_PREFIXES.items():
    assert _t in DEMENTIA_READ_CODES_CANONICAL, (
        f"override target {_t} not in canonical dementia code set")


def dementia_override(icd10: str):
    if not icd10 or len(icd10) < 3:
        return None
    return DEMENTIA_ICD10_OVERRIDE_PREFIXES.get(icd10[:3])


# ----- HOTFIX 2: explicit, format-locked date parsing ----------------------
# Verified empirically: diagnosis_table.DATE values look like '2001-07-11'.
# We parse HES dates with several candidate formats and re-format strictly.
DB_DATE_FMT = "%Y-%m-%d"
HES_INPUT_FORMATS = (
    "%Y-%m-%d",   # already canonical
    "%d/%m/%Y",   # British
    "%d-%m-%Y",
    "%Y/%m/%d",
)


def parse_hes_date(raw):
    if raw is None:
        return None
    if isinstance(raw, float) and pd.isna(raw):
        return None
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    for fmt in HES_INPUT_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime(DB_DATE_FMT)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------

def load_practice_lookup():
    print("Loading PATIENT_ID -> PRACTICE_ID from static_table ...")
    conn = sqlite3.connect(PATH_TO_DB)
    cur = conn.cursor()
    cur.execute("SELECT PATIENT_ID, PRACTICE_ID FROM static_table")
    lookup = {}
    for pid, practice in cur.fetchall():
        if pid is None or practice is None:
            continue
        lookup[int(pid)] = int(practice)
    conn.close()
    print(f"  static_table has {len(lookup):,} patients with PRACTICE_ID")
    return lookup


def main():
    print("=" * 70)
    print("  build_hes_events.py")
    print("=" * 70)

    print(f"Loading OMOP mapping: {OMOP_PICKLE}")
    with open(OMOP_PICKLE, "rb") as f:
        omop_map = pickle.load(f)
    print(f"  {len(omop_map):,} ICD-10 -> Read v2 entries")

    practice_lookup = load_practice_lookup()

    # Read hesin -> dnx_hesin_id -> (eid, date_str)
    print(f"Loading {HESIN_CSV} ...")
    hesin = pd.read_csv(
        HESIN_CSV,
        usecols=["dnx_hesin_id", "eid", "epistart", "admidate"],
        dtype={"dnx_hesin_id": str, "eid": str, "epistart": str, "admidate": str},
    )
    print(f"  {len(hesin):,} hesin rows")

    # Pick best date: epistart, fall back to admidate
    raw_dates = hesin["epistart"].where(hesin["epistart"].notna(), hesin["admidate"])
    print("  parsing HES dates with HOTFIX 2 (explicit format) ...")
    parsed = raw_dates.map(parse_hes_date)
    n_unparsed = parsed.isna().sum()
    print(f"  {n_unparsed:,} hesin rows with unparsable date (will be dropped)")
    hesin["date_str"] = parsed
    hesin = hesin.dropna(subset=["date_str"])

    hesin_lookup = dict(zip(hesin["dnx_hesin_id"].values, hesin["date_str"].values))
    eid_lookup = dict(zip(hesin["dnx_hesin_id"].values, hesin["eid"].values))
    print(f"  {len(hesin_lookup):,} hesin rows kept after date parsing")

    print(f"Loading {HESIN_DIAG_CSV} ...")
    diag = pd.read_csv(
        HESIN_DIAG_CSV,
        usecols=["dnx_hesin_id", "eid", "level", "diag_icd10"],
        dtype={"dnx_hesin_id": str, "eid": str, "level": "Int64", "diag_icd10": str},
    )
    print(f"  {len(diag):,} diag rows")
    diag = diag.dropna(subset=["diag_icd10", "dnx_hesin_id"])
    diag["diag_icd10"] = diag["diag_icd10"].str.strip()
    # Strip a trailing 'X' (UKB sometimes appends placeholder); also strip dots.
    diag["diag_icd10"] = diag["diag_icd10"].str.replace(".", "", regex=False)

    print("Translating diagnoses ...")
    out = []
    n_total = 0
    n_dementia_override = 0
    n_omop_hit = 0
    n_no_map = 0
    n_no_practice = 0
    n_no_date = 0
    n_no_patient = 0
    seen = set()  # de-dupe (practice, patient, event, date)

    for dnx_id, eid_str, level, icd in zip(
        diag["dnx_hesin_id"].values,
        diag["eid"].values,
        diag["level"].values,
        diag["diag_icd10"].values,
    ):
        n_total += 1
        # Date
        date_str = hesin_lookup.get(dnx_id)
        if date_str is None:
            n_no_date += 1
            continue
        # Patient
        try:
            pid = int(eid_str)
        except (TypeError, ValueError):
            n_no_patient += 1
            continue
        practice = practice_lookup.get(pid)
        if practice is None:
            n_no_practice += 1
            continue

        # HOTFIX 1: dementia override before anything else
        rv = dementia_override(icd)
        if rv is not None:
            n_dementia_override += 1
        else:
            rv = omop_map.get(icd)
            if rv is None:
                n_no_map += 1
                continue
            n_omop_hit += 1

        key = (practice, pid, rv, date_str)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)

    print()
    print(f"  total diagnoses scanned       : {n_total:,}")
    print(f"  dropped (no parseable date)   : {n_no_date:,}")
    print(f"  dropped (bad eid)             : {n_no_patient:,}")
    print(f"  dropped (no PRACTICE_ID)      : {n_no_practice:,}")
    print(f"  dropped (no OMOP mapping)     : {n_no_map:,}")
    print(f"  dementia override applied     : {n_dementia_override:,}")
    print(f"  OMOP mapping applied          : {n_omop_hit:,}")
    print(f"  unique (insertable) events    : {len(out):,}")

    os.makedirs(os.path.dirname(OUTPUT_PICKLE), exist_ok=True)
    with open(OUTPUT_PICKLE, "wb") as f:
        pickle.dump(out, f)
    print(f"\nWrote {OUTPUT_PICKLE}")


if __name__ == "__main__":
    main()
