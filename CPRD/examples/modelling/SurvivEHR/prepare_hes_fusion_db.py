"""
prepare_hes_fusion_db.py
========================
1. Make a fresh copy of the original SQLite DB at a new path.
2. Insert all translated HES events into diagnosis_table on the COPY.
3. Verify counts and split coverage.

The original DB is NEVER modified.
"""

import os
import pickle
import shutil
import sqlite3

SRC_DB = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
DST_DB = "/Data0/swangek_data/991/CPRD/data/example_exercise_database_hes_fusion.db"
EVENTS_PICKLE = "/Data0/swangek_data/991/CPRD/data/hes_events_for_db.pickle"
SPLITS_PATH = (
    "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/"
    "practice_id_splits.pickle"
)


def main():
    print("=" * 70)
    print("  prepare_hes_fusion_db.py")
    print("=" * 70)
    print(f"  src: {SRC_DB}")
    print(f"  dst: {DST_DB}")

    # Always start from a clean copy.
    if os.path.exists(DST_DB):
        os.remove(DST_DB)
    shutil.copy2(SRC_DB, DST_DB)
    print("  copied OK")

    with open(EVENTS_PICKLE, "rb") as f:
        events = pickle.load(f)
    print(f"  loaded {len(events):,} HES events to insert")

    conn = sqlite3.connect(DST_DB)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM diagnosis_table")
    n_before = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT PATIENT_ID) FROM diagnosis_table")
    p_before = cur.fetchone()[0]
    print(f"  before insert: {n_before:,} rows, {p_before:,} patients")

    # Bulk insert
    BATCH = 200_000
    for i in range(0, len(events), BATCH):
        cur.executemany(
            "INSERT INTO diagnosis_table (PRACTICE_ID, PATIENT_ID, EVENT, DATE) "
            "VALUES (?, ?, ?, ?)",
            events[i:i + BATCH],
        )
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM diagnosis_table")
    n_after = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT PATIENT_ID) FROM diagnosis_table")
    p_after = cur.fetchone()[0]
    print(f"  after  insert: {n_after:,} rows, {p_after:,} patients")
    print(f"  delta:        +{n_after - n_before:,} rows, "
          f"+{p_after - p_before:,} new patients")

    # Sanity: a sample
    cur.execute(
        "SELECT * FROM diagnosis_table "
        "WHERE EVENT IN ('Eu00.','Eu01.','Eu02z','F110.') "
        "ORDER BY DATE DESC LIMIT 5"
    )
    print("  sample inserted-or-existing dementia rows:")
    for row in cur.fetchall():
        print(f"    {row}")

    # Coverage check vs splits
    if os.path.exists(SPLITS_PATH):
        with open(SPLITS_PATH, "rb") as f:
            splits = pickle.load(f)
        all_split_practices = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        cur.execute("SELECT DISTINCT PRACTICE_ID FROM diagnosis_table")
        all_practices = {r[0] for r in cur.fetchall() if r[0] is not None}
        in_splits = all_practices & all_split_practices
        outside = all_practices - all_split_practices
        print(f"  practices present in any split: {len(in_splits):,}")
        print(f"  practices NOT in any split:     {len(outside):,}")
    else:
        print(f"  WARN: splits file not found at {SPLITS_PATH}")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
