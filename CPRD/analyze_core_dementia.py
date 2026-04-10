"""
Analyze core dementia patients:
1. Count unique patients diagnosed with the 31 core dementia codes
2. Show time distribution of dementia diagnoses

Uses BOTH data sources:
  - gp_clinical.csv (read_2 and read_3 columns)
  - ready_for_code_diagnosis.csv (medcode column)
"""
import csv
import sys
from collections import defaultdict, Counter
from datetime import datetime

# 31 Core Dementia Read Codes (ICD-10 F00-F03, G30)
CORE_DEMENTIA_CODES = set([
    # F110. hierarchy: Alzheimer's disease (Neurology chapter)
    "F110.", "F1100", "F1101",
    # E00.. hierarchy: Senile and presenile organic psychotic conditions
    "E00..", "E000.", "E001.", "E001z", "E0020", "E0021",
    "E003.", "E004.", "E004z", "E0040", "E00z.",
    # Eu00.: Dementia in Alzheimer's disease [F00]
    "Eu00.", "Eu000", "Eu001", "Eu002", "Eu00z",
    # Eu01.: Vascular dementia [F01]
    "Eu01.", "Eu011", "Eu012", "Eu013", "Eu01y", "Eu01z",
    # Eu02.: Dementia in other diseases [F02]
    "Eu02.", "Eu020", "Eu023", "Eu025", "Eu02y", "Eu02z",
])

# Also include the 10 broader codes for comparison
BROADER_CODES = set([
    "Eu057", "Eu04.", "Eu053", "Eu04z", "Eu0z.",
    "Eu060", "Eu054", "Eu05y", "Eu052", "Eu062",
])

ALL_41_CODES = CORE_DEMENTIA_CODES | BROADER_CODES


def parse_date(date_str):
    """Parse date string, return (year, month) or None."""
    if not date_str or date_str.strip() == '':
        return None
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return dt
    except ValueError:
        try:
            dt = datetime.strptime(date_str.strip(), "%d/%m/%Y")
            return dt
        except ValueError:
            return None


def analyze_file(filepath, patient_col, date_col, code_cols):
    """
    Scan a CSV file and collect dementia diagnosis records.

    Args:
        filepath: path to CSV
        patient_col: column name for patient ID
        date_col: column name for event date
        code_cols: list of column names to check for codes

    Returns:
        core_patients: dict {patient_id: [list of (date, code)]}
        broader_patients: dict {patient_id: [list of (date, code)]}
    """
    core_patients = defaultdict(list)
    broader_patients = defaultdict(list)

    total_lines = 0
    matched_core = 0
    matched_broader = 0

    print(f"\n📂 Processing: {filepath}")
    print(f"   Patient col: {patient_col}, Date col: {date_col}, Code cols: {code_cols}")

    with open(filepath, 'r', newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)

        for row in reader:
            total_lines += 1
            if total_lines % 10_000_000 == 0:
                print(f"   ... processed {total_lines:,} lines (core matches: {matched_core:,}, broader: {matched_broader:,})")

            # Check all code columns
            for code_col in code_cols:
                code = row.get(code_col, '').strip()
                if not code:
                    continue

                patient_id = row[patient_col].strip()
                event_date = row.get(date_col, '').strip()

                if code in CORE_DEMENTIA_CODES:
                    core_patients[patient_id].append((event_date, code))
                    matched_core += 1
                elif code in BROADER_CODES:
                    broader_patients[patient_id].append((event_date, code))
                    matched_broader += 1

    print(f"   ✅ Done! Total lines: {total_lines:,}")
    print(f"   Core dementia records: {matched_core:,} | Broader records: {matched_broader:,}")

    return core_patients, broader_patients


def print_analysis(core_patients, broader_patients, source_name):
    """Print detailed analysis."""

    print(f"\n{'='*90}")
    print(f"ANALYSIS RESULTS — {source_name}")
    print(f"{'='*90}")

    # ---- 1. Patient counts ----
    core_only = set(core_patients.keys()) - set(broader_patients.keys())
    broader_only = set(broader_patients.keys()) - set(core_patients.keys())
    both = set(core_patients.keys()) & set(broader_patients.keys())
    all_patients = set(core_patients.keys()) | set(broader_patients.keys())

    print(f"\n📊 Patient Counts:")
    print(f"   🔴 Core dementia patients (31 codes, F00-F03/G30):    {len(core_patients):,}")
    print(f"   🟡 Broader organic disorder patients (10 codes, F05-F09): {len(broader_patients):,}")
    print(f"   ├── Core only (not in broader):                       {len(core_only):,}")
    print(f"   ├── Broader only (not in core):                       {len(broader_only):,}")
    print(f"   ├── Both core + broader:                              {len(both):,}")
    print(f"   └── Total unique (all 41 codes):                      {len(all_patients):,}")

    # ---- 2. Core dementia code frequency ----
    print(f"\n📋 Core Dementia Code Frequency (top codes):")
    code_counts = Counter()
    for records in core_patients.values():
        for _, code in records:
            code_counts[code] += 1

    total_records = sum(code_counts.values())
    print(f"   Total diagnosis records: {total_records:,}")
    print(f"   {'Code':<10} {'Count':>8} {'%':>8}  Description")
    print(f"   {'-'*75}")

    CODE_DESCRIPTIONS = {
        "F110.": "Alzheimer's disease",
        "F1100": "Alzheimer's disease, early onset",
        "F1101": "Alzheimer's disease, late onset",
        "E00..": "Senile/presenile organic psychotic conditions",
        "E000.": "Uncomplicated senile dementia",
        "E001.": "Presenile dementia",
        "E001z": "Presenile dementia NOS",
        "E0020": "Senile dementia with depressive/paranoid features",
        "E0021": "Senile dementia with paranoia",
        "E003.": "Senile dementia with delirium",
        "E004.": "Arteriosclerotic dementia",
        "E004z": "Arteriosclerotic dementia NOS",
        "E0040": "Uncomplicated arteriosclerotic dementia",
        "E00z.": "Senile or presenile psychoses NOS",
        "Eu00.": "Dementia in Alzheimer's disease",
        "Eu000": "Dementia in Alzheimer's, early onset",
        "Eu001": "Dementia in Alzheimer's, late onset",
        "Eu002": "Dementia in Alzheimer's, atypical/mixed",
        "Eu00z": "Dementia in Alzheimer's, unspecified",
        "Eu01.": "Vascular dementia",
        "Eu011": "Multi-infarct dementia",
        "Eu012": "Subcortical vascular dementia",
        "Eu013": "Mixed cortical/subcortical vascular dementia",
        "Eu01y": "Other vascular dementia",
        "Eu01z": "Vascular dementia, unspecified",
        "Eu02.": "Dementia in other diseases",
        "Eu020": "Dementia in Pick's disease",
        "Eu023": "Dementia in Parkinson's disease",
        "Eu025": "Dementia in other specified diseases",
        "Eu02y": "Dementia in other diseases classified elsewhere",
        "Eu02z": "Unspecified dementia",
    }

    for code, count in code_counts.most_common():
        desc = CODE_DESCRIPTIONS.get(code, "")
        pct = 100.0 * count / total_records if total_records > 0 else 0
        print(f"   {code:<10} {count:>8,} {pct:>7.1f}%  {desc}")

    # ---- 3. Time distribution ----
    print(f"\n📅 Diagnosis Time Distribution (by year):")
    print(f"   (Based on FIRST diagnosis date per patient)")

    # Get first diagnosis date per patient
    first_diag = {}
    for patient_id, records in core_patients.items():
        dates = []
        for date_str, code in records:
            dt = parse_date(date_str)
            if dt:
                dates.append(dt)
        if dates:
            first_diag[patient_id] = min(dates)

    if not first_diag:
        print("   ⚠️  No valid dates found!")
        return

    # Year distribution
    year_counts = Counter()
    for dt in first_diag.values():
        year_counts[dt.year] += 1

    total_with_dates = len(first_diag)
    no_date_count = len(core_patients) - total_with_dates

    print(f"   Patients with valid dates: {total_with_dates:,}")
    if no_date_count > 0:
        print(f"   Patients without valid dates: {no_date_count:,}")

    min_year = min(year_counts.keys())
    max_year = max(year_counts.keys())
    max_count = max(year_counts.values())
    bar_width = 50

    print(f"\n   {'Year':<6} {'Count':>6}  {'%':>6}  Distribution")
    print(f"   {'-'*75}")

    for year in range(min_year, max_year + 1):
        count = year_counts.get(year, 0)
        pct = 100.0 * count / total_with_dates if total_with_dates > 0 else 0
        bar_len = int(bar_width * count / max_count) if max_count > 0 else 0
        bar = '█' * bar_len
        print(f"   {year:<6} {count:>6,}  {pct:>5.1f}%  {bar}")

    # ---- 4. Summary statistics ----
    all_dates = sorted(first_diag.values())
    median_date = all_dates[len(all_dates) // 2]

    print(f"\n📈 Summary Statistics (first diagnosis per patient):")
    print(f"   Earliest diagnosis: {all_dates[0].strftime('%Y-%m-%d')}")
    print(f"   Latest diagnosis:   {all_dates[-1].strftime('%Y-%m-%d')}")
    print(f"   Median diagnosis:   {median_date.strftime('%Y-%m-%d')}")

    # By 5-year bins
    print(f"\n   5-Year Period Breakdown:")
    period_counts = Counter()
    for dt in first_diag.values():
        period_start = (dt.year // 5) * 5
        period_counts[period_start] += 1

    for period in sorted(period_counts.keys()):
        count = period_counts[period]
        pct = 100.0 * count / total_with_dates
        print(f"   {period}-{period+4}: {count:>6,} patients ({pct:>5.1f}%)")

    # ---- 5. Dementia subtype breakdown ----
    print(f"\n🧠 Dementia Subtype Breakdown (by patient, based on any code received):")

    subtype_map = {
        "Alzheimer's": {"F110.", "F1100", "F1101", "Eu00.", "Eu000", "Eu001", "Eu002", "Eu00z"},
        "Vascular": {"Eu01.", "Eu011", "Eu012", "Eu013", "Eu01y", "Eu01z"},
        "Other diseases": {"Eu02.", "Eu020", "Eu023", "Eu025", "Eu02y"},
        "Unspecified/Senile": {"E00..", "E000.", "E001.", "E001z", "E0020", "E0021",
                               "E003.", "E004.", "E004z", "E0040", "E00z.", "Eu02z"},
    }

    subtype_patients = defaultdict(set)
    for patient_id, records in core_patients.items():
        for _, code in records:
            for subtype, codes_set in subtype_map.items():
                if code in codes_set:
                    subtype_patients[subtype].add(patient_id)

    for subtype in ["Alzheimer's", "Vascular", "Other diseases", "Unspecified/Senile"]:
        count = len(subtype_patients[subtype])
        pct = 100.0 * count / len(core_patients) if core_patients else 0
        print(f"   {subtype:<22} {count:>6,} patients ({pct:>5.1f}%)")

    # Check overlap
    multi_subtype = set()
    for patient_id in core_patients:
        subtypes_for_patient = set()
        for _, code in core_patients[patient_id]:
            for subtype, codes_set in subtype_map.items():
                if code in codes_set:
                    subtypes_for_patient.add(subtype)
        if len(subtypes_for_patient) > 1:
            multi_subtype.add(patient_id)

    print(f"\n   Patients with multiple subtypes: {len(multi_subtype):,} "
          f"({100.0*len(multi_subtype)/len(core_patients):.1f}%)")


def main():
    print("=" * 90)
    print("CORE DEMENTIA PATIENTS ANALYSIS")
    print("Analyzing 31 core dementia Read codes (F00-F03, G30)")
    print("=" * 90)

    # ============================================================
    # Analyze ready_for_code_diagnosis.csv
    # Columns: PRACTICE_ID, PATIENT_ID, PRACTICE_PATIENT_ID, YEAR_OF_BIRTH,
    #          DEATH_DATE, eventdate, medcode
    # ============================================================
    core_diag, broader_diag = analyze_file(
        filepath="/Data0/swangek_data/991/CPRD/data/ready_for_code_diagnosis.csv",
        patient_col="PATIENT_ID",
        date_col="eventdate",
        code_cols=["medcode"],
    )
    print_analysis(core_diag, broader_diag, "ready_for_code_diagnosis.csv")

    # ============================================================
    # Analyze gp_clinical.csv
    # Columns: eid, data_provider, event_dt, read_2, read_3, value1, value2, value3
    # ============================================================
    core_gp, broader_gp = analyze_file(
        filepath="/Data0/swangek_data/991/CPRD/data/gp_clinical.csv",
        patient_col="eid",
        date_col="event_dt",
        code_cols=["read_2", "read_3"],
    )
    print_analysis(core_gp, broader_gp, "gp_clinical.csv")

    # ============================================================
    # Cross-source comparison
    # ============================================================
    print(f"\n{'='*90}")
    print("CROSS-SOURCE COMPARISON")
    print(f"{'='*90}")

    diag_patients = set(core_diag.keys())
    gp_patients = set(core_gp.keys())

    # Note: patient IDs may differ between files (PATIENT_ID vs eid)
    print(f"\n   ready_for_code_diagnosis (PATIENT_ID): {len(diag_patients):,} core dementia patients")
    print(f"   gp_clinical (eid):                      {len(gp_patients):,} core dementia patients")

    # Check if IDs overlap (they might use different ID systems)
    overlap = diag_patients & gp_patients
    if overlap:
        print(f"   Overlapping IDs:                        {len(overlap):,}")
    else:
        print(f"   ⚠️  No overlapping IDs — likely different ID systems (PATIENT_ID vs eid)")


if __name__ == "__main__":
    main()
