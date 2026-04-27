"""
build_hes_database.py
=====================
Build a standalone HES-only SQLite database for HES backbone pretraining.
Uses the same schema as the GP database (diagnosis_table + static_table)
so FoundationalDataModule can load it directly.

Data sources:
  - hesin.csv: admission records (dnx_hesin_id, eid, admidate, ...)
  - hesin_diag.csv: diagnosis records (dnx_hesin_id, eid, diag_icd10, level, ...)
  - GP DB static_table: reuse demographics for consistent practice splits

HES event construction rules:
  1. For each admission (hesin row), get all diagnoses (hesin_diag)
  2. Each ICD-10 code becomes an EVENT, date = admidate
  3. ICD-10 codes truncated to 3 chars (e.g. "I219" -> "I21") to reduce vocab
  4. Only keep level=1 (primary diagnosis) by default
  5. PRACTICE_ID: from GP static_table to maintain same train/val/test splits

Output: data/hes_pretrain_database.db

Usage:
    python build_hes_database.py
"""

import os
import sqlite3
import pandas as pd
from tqdm import tqdm

GP_DB = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
HES_DB = "/Data0/swangek_data/991/CPRD/data/hes_pretrain_database.db"
HESIN_CSV = "/Data0/swangek_data/991/CPRD/data/hesin.csv"
HESIN_DIAG_CSV = "/Data0/swangek_data/991/CPRD/data/hesin_diag.csv"

# ICD-10 truncation length (3 = major category)
ICD10_TRUNCATE_LEN = 3
# Only keep primary diagnoses (level=1). Set to 2 for primary + secondary.
MAX_DIAG_LEVEL = 1
BATCH_SIZE = 200_000


def main():
    print("=" * 60)
    print("  Building HES Pretrain Database")
    print("=" * 60)
    print(f"  GP DB (for static):  {GP_DB}")
    print(f"  HES DB (output):     {HES_DB}")
    print(f"  ICD-10 truncation:   {ICD10_TRUNCATE_LEN} chars")
    print(f"  Max diagnosis level: {MAX_DIAG_LEVEL}")
    print()

    # ---- Step 1: Copy static_table from GP DB ----
    print("===== Step 1: Copy static_table from GP DB =====")
    if os.path.exists(HES_DB):
        os.remove(HES_DB)
        print(f"  Removed existing {HES_DB}")

    conn_gp = sqlite3.connect(GP_DB)
    static_df = pd.read_sql("SELECT * FROM static_table", conn_gp)
    conn_gp.close()

    # Build PATIENT_ID -> PRACTICE_ID mapping
    pid_to_practice = dict(
        zip(static_df["PATIENT_ID"].astype(int), static_df["PRACTICE_ID"].astype(int))
    )
    print(f"  static_table: {len(static_df)} rows")
    print(f"  PATIENT_ID -> PRACTICE_ID mapping: {len(pid_to_practice)} entries")

    conn_hes = sqlite3.connect(HES_DB)
    cur_hes = conn_hes.cursor()

    # Write static_table
    static_df.to_sql("static_table", conn_hes, if_exists="replace", index=False)

    # Create static index (same as GP DB)
    cur_hes.execute("""
        CREATE INDEX IF NOT EXISTS static_index
        ON static_table (
            PRACTICE_ID,
            PATIENT_ID,
            HEALTH_AUTH,
            COUNTRY,
            SEX,
            ETHNICITY
        )
    """)
    conn_hes.commit()

    # ---- Step 2: Create empty diagnosis_table ----
    print("\n===== Step 2: Create diagnosis_table =====")
    cur_hes.execute("""
        CREATE TABLE diagnosis_table (
            PRACTICE_ID integer,
            PATIENT_ID integer,
            EVENT text,
            DATE text
        )
    """)
    conn_hes.commit()

    # ---- Step 3: Read hesin.csv for admission dates ----
    print("\n===== Step 3: Read hesin.csv =====")
    hesin = pd.read_csv(
        HESIN_CSV,
        usecols=["dnx_hesin_id", "eid", "admidate"],
        dtype={"dnx_hesin_id": str, "eid": str},
    )
    hesin = hesin.dropna(subset=["admidate"])
    print(f"  hesin.csv: {len(hesin)} records with valid admidate")

    # Build dnx_hesin_id -> admidate lookup (dates already in YYYY-MM-DD format)
    hesin_date_lookup = dict(zip(hesin["dnx_hesin_id"], hesin["admidate"]))
    print(f"  Unique admissions: {len(hesin_date_lookup)}")

    # ---- Step 4: Read hesin_diag.csv and build HES events ----
    print(f"\n===== Step 4: Read hesin_diag.csv (level <= {MAX_DIAG_LEVEL}) =====")
    hesin_diag = pd.read_csv(
        HESIN_DIAG_CSV,
        usecols=["dnx_hesin_id", "eid", "diag_icd10", "level"],
        dtype={"dnx_hesin_id": str, "eid": str, "diag_icd10": str},
    )
    print(f"  Total diagnosis records: {len(hesin_diag)}")

    # Filter: valid ICD-10 codes
    hesin_diag = hesin_diag.dropna(subset=["diag_icd10"])
    hesin_diag["diag_icd10"] = hesin_diag["diag_icd10"].str.strip()
    print(f"  After dropping NaN ICD-10: {len(hesin_diag)}")

    # Filter: diagnosis level
    hesin_diag["level"] = pd.to_numeric(hesin_diag["level"], errors="coerce")
    hesin_diag = hesin_diag[hesin_diag["level"] <= MAX_DIAG_LEVEL]
    print(f"  After level <= {MAX_DIAG_LEVEL} filter: {len(hesin_diag)}")

    # Truncate ICD-10 codes
    hesin_diag["EVENT"] = hesin_diag["diag_icd10"].str[:ICD10_TRUNCATE_LEN]

    # Map admission dates via dnx_hesin_id (string-based join)
    hesin_diag["DATE"] = hesin_diag["dnx_hesin_id"].map(hesin_date_lookup)
    hesin_diag = hesin_diag.dropna(subset=["DATE"])
    print(f"  After joining admission dates: {len(hesin_diag)}")

    # Map PATIENT_ID and PRACTICE_ID
    hesin_diag["PATIENT_ID"] = hesin_diag["eid"].astype(int)
    hesin_diag["PRACTICE_ID"] = hesin_diag["PATIENT_ID"].map(pid_to_practice)

    # Drop patients not in GP database (no PRACTICE_ID mapping)
    n_before = len(hesin_diag)
    hesin_diag = hesin_diag.dropna(subset=["PRACTICE_ID"])
    hesin_diag["PRACTICE_ID"] = hesin_diag["PRACTICE_ID"].astype(int)
    n_dropped = n_before - len(hesin_diag)
    print(f"  Dropped {n_dropped} records (patients not in GP DB)")
    print(f"  Final HES events: {len(hesin_diag)}")

    # ---- Step 5: Batch insert into diagnosis_table ----
    print(f"\n===== Step 5: Insert into diagnosis_table =====")
    records = list(
        zip(
            hesin_diag["PRACTICE_ID"],
            hesin_diag["PATIENT_ID"],
            hesin_diag["EVENT"],
            hesin_diag["DATE"],
        )
    )

    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Inserting"):
        cur_hes.executemany(
            "INSERT INTO diagnosis_table VALUES (?, ?, ?, ?)",
            records[i : i + BATCH_SIZE],
        )
    conn_hes.commit()

    # ---- Step 6: Create index ----
    print("\n===== Step 6: Create index =====")
    cur_hes.execute("""
        CREATE INDEX IF NOT EXISTS diagnosis_index
        ON diagnosis_table (PRACTICE_ID)
    """)
    conn_hes.commit()

    # ---- Step 7: Statistics ----
    print("\n===== Statistics =====")
    cur_hes.execute("SELECT COUNT(*) FROM diagnosis_table")
    n_events = cur_hes.fetchone()[0]
    cur_hes.execute("SELECT COUNT(DISTINCT PATIENT_ID) FROM diagnosis_table")
    n_patients = cur_hes.fetchone()[0]
    cur_hes.execute("SELECT COUNT(DISTINCT EVENT) FROM diagnosis_table")
    n_codes = cur_hes.fetchone()[0]

    # Top 10 most frequent ICD-10 codes
    cur_hes.execute("""
        SELECT EVENT, COUNT(*) as cnt
        FROM diagnosis_table
        GROUP BY EVENT
        ORDER BY cnt DESC
        LIMIT 10
    """)
    top_codes = cur_hes.fetchall()

    # Events per patient distribution
    cur_hes.execute("""
        SELECT AVG(cnt), MIN(cnt), MAX(cnt) FROM (
            SELECT COUNT(*) as cnt FROM diagnosis_table GROUP BY PATIENT_ID
        )
    """)
    avg_events, min_events, max_events = cur_hes.fetchone()

    conn_hes.close()

    print(f"  Total events:     {n_events:,}")
    print(f"  Unique patients:  {n_patients:,}")
    print(f"  Unique ICD-10 codes (truncated to {ICD10_TRUNCATE_LEN} chars): {n_codes:,}")
    print(f"  Events/patient:   mean={avg_events:.1f}, min={min_events}, max={max_events}")
    print(f"\n  Top 10 ICD-10 codes:")
    for code, cnt in top_codes:
        print(f"    {code}: {cnt:,}")
    print(f"\n  Output: {HES_DB}")
    print(f"  Size:   {os.path.getsize(HES_DB) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
