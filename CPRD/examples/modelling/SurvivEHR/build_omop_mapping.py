"""
build_omop_mapping.py
=====================
Build a {ICD-10 -> Read v2} mapping dictionary by scanning
omop_condition_occurrence.csv.

Strategy:
  1. Load all distinct EVENT codes (with counts) from the original
     diagnosis_table. These are the Read v2 codes the pretrained model
     "knows".
  2. Stream omop_condition_occurrence.csv in chunks. For each
     condition_concept_id, collect the set of source codes that map to it.
  3. For each concept, partition the source codes into:
        - readv2_set : codes that exist in diagnosis_table
        - icd10_set  : codes that match an ICD-10 shape and are NOT in
                       diagnosis_table
     (Plan disambiguation rule: if a code exists in diagnosis_table, treat
     it as Read v2; else if it matches the ICD-10 regex, treat it as ICD-10.)
  4. For each ICD-10 code, accumulate Read v2 candidates across all
     concepts that link them, then pick the candidate with the highest
     count in diagnosis_table.

Output: omop_icd10_to_readv2.pickle  -- dict {icd10_code: read_v2_code}

NOTE: The dementia outcome ICD-10 codes (F00*/F01*/F02*/F03*/G30*) are
handled by a HARDCODED override in build_hes_events.py (HOTFIX 1) so we
do not depend on whatever OMOP gives us. This script still emits whatever
mappings OMOP supports for them, but they will be ignored downstream.
"""

import os
import re
import pickle
import sqlite3
from collections import defaultdict

import pandas as pd

PATH_TO_DB = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
OMOP_CSV = "/Data0/swangek_data/991/CPRD/data/omop_condition_occurrence.csv"
OUTPUT_PICKLE = "/Data0/swangek_data/991/CPRD/data/omop_icd10_to_readv2.pickle"

# ICD-10 shape: one letter followed by 2 digits, then optionally a dot and
# more alphanumerics, OR (the UKB hesin convention) no dot at all.
# Examples that should match: I10, I251, C10E, F001, G309, F03
ICD10_RE = re.compile(r"^[A-Z][0-9]{2}[A-Z0-9]{0,4}$")

CHUNK = 1_000_000


def load_diagnosis_event_counts():
    print("Loading EVENT counts from diagnosis_table ...")
    conn = sqlite3.connect(PATH_TO_DB)
    cur = conn.cursor()
    cur.execute("SELECT EVENT, COUNT(*) FROM diagnosis_table GROUP BY EVENT")
    counts = {row[0]: int(row[1]) for row in cur.fetchall() if row[0]}
    conn.close()
    print(f"  diagnosis_table has {len(counts)} distinct EVENT codes")
    return counts


def is_icd10(code: str) -> bool:
    if code is None:
        return False
    return bool(ICD10_RE.match(code))


def main():
    readv2_counts = load_diagnosis_event_counts()
    readv2_set = set(readv2_counts.keys())

    # concept_id -> set of source codes
    concept_to_sources = defaultdict(set)

    print(f"Streaming {OMOP_CSV} ...")
    n_rows = 0
    reader = pd.read_csv(
        OMOP_CSV,
        usecols=["condition_concept_id", "condition_source_value"],
        dtype={"condition_concept_id": "Int64", "condition_source_value": str},
        chunksize=CHUNK,
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["condition_concept_id", "condition_source_value"])
        # Strip whitespace
        chunk["condition_source_value"] = chunk["condition_source_value"].str.strip()
        # Aggregate uniques per chunk
        for cid, src in zip(
            chunk["condition_concept_id"].astype(int).values,
            chunk["condition_source_value"].values,
        ):
            concept_to_sources[cid].add(src)
        n_rows += len(chunk)
        if n_rows % (5 * CHUNK) == 0:
            print(f"  ... {n_rows:,} rows scanned, {len(concept_to_sources):,} concepts")

    print(f"  total rows scanned: {n_rows:,}")
    print(f"  unique concept_ids: {len(concept_to_sources):,}")

    # icd10 -> {readv2 -> total count over linking concepts}
    icd10_candidates = defaultdict(lambda: defaultdict(int))
    for cid, src_set in concept_to_sources.items():
        readv2_here = [s for s in src_set if s in readv2_set]
        if not readv2_here:
            continue
        icd10_here = [
            s for s in src_set
            if s not in readv2_set and is_icd10(s)
        ]
        if not icd10_here:
            continue
        for icd in icd10_here:
            for rv in readv2_here:
                # Use the read v2 code's diagnosis_table count as the score.
                icd10_candidates[icd][rv] += readv2_counts.get(rv, 0)

    # Resolve 1-to-N: pick the read v2 with the highest cumulative score.
    mapping = {}
    for icd, cands in icd10_candidates.items():
        best_rv = max(cands.items(), key=lambda kv: kv[1])[0]
        mapping[icd] = best_rv

    print(f"\nFinal ICD-10 -> Read v2 mapping: {len(mapping):,} entries")

    # Spot-check a few critical disease codes
    critical = [
        "I10", "I251", "I50", "I639", "E119", "F03", "F009", "F019",
        "G309", "S060", "F329",
    ]
    print("Spot-check (critical codes):")
    for c in critical:
        print(f"  {c} -> {mapping.get(c, '<MISSING>')}")

    os.makedirs(os.path.dirname(OUTPUT_PICKLE), exist_ok=True)
    with open(OUTPUT_PICKLE, "wb") as f:
        pickle.dump(mapping, f)
    print(f"\nWrote {OUTPUT_PICKLE}")


if __name__ == "__main__":
    main()
