"""
build_hes_database.py
=====================
Build a standalone HES-only SQLite database for HES backbone pretraining.
Uses the same schema as the GP database so FoundationalDataModule can load it directly.

Data sources:
  - hesin.csv: admission records (eid, admidate, disdate, ...)
  - hesin_diag.csv: diagnosis records (eid, diag_icd10, level, ...)
  - GP DB static_table: reuse demographics

HES event construction rules:
  1. For each admission (hesin row), get all diagnoses (hesin_diag)
  2. Each ICD-10 code becomes an EVENT, date = admidate
  3. ICD-10 codes truncated to 3 chars (e.g. "I219" -> "I21") to reduce vocab
  4. Only keep level=1 (primary diagnosis)
  5. PRACTICE_ID: from GP static_table to maintain consistent splits

Output: /Data0/swangek_data/991/CPRD/data/hes_pretrain_database.db
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm

GP_DB = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
HES_DB = "/Data0/swangek_data/991/CPRD/data/hes_pretrain_database.db"
HESIN_CSV = "/Data0/swangek_data/991/CPRD/data/hesin.csv"
HESIN_DIAG_CSV = "/Data0/swangek_data/991/CPRD/data/hesin_diag.csv"

ICD10_TRUNCATE_LEN = 3


def main():
    # 1. Copy static_table from GP DB
    if os.path.exists(HES_DB):
        os.remove(HES_DB)

    conn_gp = sqlite3.connect(GP_DB)
    conn_hes = sqlite3.connect(HES_DB)
    cur_hes = conn_hes.cursor()

    static_df = pd.read_sql("SELECT * FROM static_table", conn_gp)
    static_df.to_sql("static_table", conn_hes, if_exists="replace", index=False)
    print(f"Copied static_table: {len(static_df)} rows")

    # Build PATIENT_ID -> PRACTICE_ID mapping
    pid_to_practice = dict(zip(static_df["PATIENT_ID"].astype(int),
                               static_df["PRACTICE_ID"].astype(int)))
    conn_gp.close()

    # 2. Create empty diagnosis_table
    cur_hes.execute("""
        CREATE TABLE diagnosis_table (
            PRACTICE_ID integer,
            PATIENT_ID integer,
            EVENT text,
            DATE text
        )
    """)

    # 3. Read hesin + hesin_diag, build HES events
    print("Reading hesin.csv...")
    hesin = pd.read_csv(HESIN_CSV, usecols=["dnx_hesin_id", "eid", "admidate"])
    hesin = hesin.dropna(subset=["admidate"])
    hesin["eid"] = hesin["eid"].astype(int)

    print("Reading hesin_diag.csv...")
    hesin_diag = pd.read_csv(HESIN_DIAG_CSV,
                              usecols=["dnx_hesin_id", "eid", "diag_icd10", "level"])
    hesin_diag = hesin_diag.dropna(subset=["diag_icd10"])
    hesin_diag["eid"] = hesin_diag["eid"].astype(int)

    # Only keep primary diagnoses (level=1)
    hesin_diag = hesin_diag[hesin_diag["level"] == 1]

    # Truncate ICD-10 codes
    hesin_diag["EVENT"] = hesin_diag["diag_icd10"].str[:ICD10_TRUNCATE_LEN]

    # Merge admission dates
    hesin_diag = hesin_diag.merge(
        hesin[["dnx_hesin_id", "admidate"]],
        on="dnx_hesin_id",
        how="left"
    )
    hesin_diag = hesin_diag.dropna(subset=["admidate"])

    # Add PRACTICE_ID
    hesin_diag["PRACTICE_ID"] = hesin_diag["eid"].map(pid_to_practice)
    hesin_diag = hesin_diag.dropna(subset=["PRACTICE_ID"])
    hesin_diag["PRACTICE_ID"] = hesin_diag["PRACTICE_ID"].astype(int)

    # 4. Batch insert
    records = list(zip(
        hesin_diag["PRACTICE_ID"],
        hesin_diag["eid"],
        hesin_diag["EVENT"],
        hesin_diag["admidate"]
    ))
    print(f"Inserting {len(records):,} HES diagnosis events...")

    BATCH = 200_000
    for i in tqdm(range(0, len(records), BATCH)):
        cur_hes.executemany(
            "INSERT INTO diagnosis_table VALUES (?, ?, ?, ?)",
            records[i:i+BATCH]
        )
    conn_hes.commit()

    # 5. Create index
    cur_hes.execute("""
        CREATE INDEX IF NOT EXISTS diagnosis_index
        ON diagnosis_table (PRACTICE_ID)
    """)
    conn_hes.commit()

    # 6. Statistics
    cur_hes.execute("SELECT COUNT(*) FROM diagnosis_table")
    n_events = cur_hes.fetchone()[0]
    cur_hes.execute("SELECT COUNT(DISTINCT PATIENT_ID) FROM diagnosis_table")
    n_patients = cur_hes.fetchone()[0]
    cur_hes.execute("SELECT COUNT(DISTINCT EVENT) FROM diagnosis_table")
    n_codes = cur_hes.fetchone()[0]

    print(f"\nHES Database built:")
    print(f"  Events: {n_events:,}")
    print(f"  Patients: {n_patients:,}")
    print(f"  Unique ICD-10 codes (truncated to {ICD10_TRUNCATE_LEN} chars): {n_codes:,}")
    print(f"  Output: {HES_DB}")

    conn_hes.close()


if __name__ == "__main__":
    main()
