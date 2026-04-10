"""
Analyze the AGE at which patients were first diagnosed with core dementia.
Uses: ready_for_code_diagnosis.csv
  Columns: PRACTICE_ID, PATIENT_ID, PRACTICE_PATIENT_ID, YEAR_OF_BIRTH,
           DEATH_DATE, eventdate, medcode
"""
import csv
from collections import defaultdict, Counter
from datetime import datetime

# 31 Core Dementia Read Codes
CORE_DEMENTIA_CODES = set([
    "F110.", "F1100", "F1101",
    "E00..", "E000.", "E001.", "E001z", "E0020", "E0021",
    "E003.", "E004.", "E004z", "E0040", "E00z.",
    "Eu00.", "Eu000", "Eu001", "Eu002", "Eu00z",
    "Eu01.", "Eu011", "Eu012", "Eu013", "Eu01y", "Eu01z",
    "Eu02.", "Eu020", "Eu023", "Eu025", "Eu02y", "Eu02z",
])


def main():
    filepath = "/Data0/swangek_data/991/CPRD/data/ready_for_code_diagnosis.csv"

    # Step 1: Collect all dementia records per patient
    # {patient_id: {"yob": int, "records": [(eventdate_str, code), ...]}}
    patients = defaultdict(lambda: {"yob": None, "records": []})

    print("📂 Scanning ready_for_code_diagnosis.csv ...")
    total = 0
    matched = 0

    with open(filepath, 'r', newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if total % 20_000_000 == 0:
                print(f"   ... {total:,} lines processed, {matched:,} dementia records found")

            code = row["medcode"].strip()
            if code not in CORE_DEMENTIA_CODES:
                continue

            matched += 1
            pid = row["PATIENT_ID"].strip()
            yob = row["YEAR_OF_BIRTH"].strip()
            eventdate = row["eventdate"].strip()

            if yob:
                patients[pid]["yob"] = int(yob)
            patients[pid]["records"].append((eventdate, code))

    print(f"   ✅ Done! {total:,} lines, {matched:,} dementia records, {len(patients):,} patients\n")

    # Step 2: Compute age at FIRST diagnosis
    ages_at_first_diag = []
    ages_all_diag = []  # all diagnosis ages (not just first)
    no_date = 0
    no_yob = 0
    subtype_ages = defaultdict(list)  # subtype -> [ages]

    subtype_map = {
        "Alzheimer's": {"F110.", "F1100", "F1101", "Eu00.", "Eu000", "Eu001", "Eu002", "Eu00z"},
        "Vascular": {"Eu01.", "Eu011", "Eu012", "Eu013", "Eu01y", "Eu01z"},
        "Other diseases": {"Eu02.", "Eu020", "Eu023", "Eu025", "Eu02y"},
        "Unspecified/Senile": {"E00..", "E000.", "E001.", "E001z", "E0020", "E0021",
                               "E003.", "E004.", "E004z", "E0040", "E00z.", "Eu02z"},
    }

    for pid, info in patients.items():
        yob = info["yob"]
        if yob is None:
            no_yob += 1
            continue

        # Parse all dates
        dated_records = []
        for date_str, code in info["records"]:
            if not date_str:
                continue
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                dated_records.append((dt, code))
            except ValueError:
                try:
                    dt = datetime.strptime(date_str, "%d/%m/%Y")
                    dated_records.append((dt, code))
                except ValueError:
                    pass

        if not dated_records:
            no_date += 1
            continue

        # Sort by date
        dated_records.sort(key=lambda x: x[0])

        # First diagnosis age
        first_dt, first_code = dated_records[0]
        first_age = first_dt.year - yob
        ages_at_first_diag.append((first_age, pid, first_dt, first_code))

        # All diagnosis ages
        for dt, code in dated_records:
            age = dt.year - yob
            ages_all_diag.append(age)

        # Subtype-specific first age
        subtypes_seen = set()
        for dt, code in dated_records:
            for subtype, codes_set in subtype_map.items():
                if code in codes_set and subtype not in subtypes_seen:
                    subtypes_seen.add(subtype)
                    subtype_ages[subtype].append(dt.year - yob)

    # Sort by age
    ages_at_first_diag.sort(key=lambda x: x[0])

    print("=" * 90)
    print("DEMENTIA DIAGNOSIS AGE DISTRIBUTION")
    print("(Based on 31 core dementia codes, first diagnosis per patient)")
    print("=" * 90)

    if no_yob > 0:
        print(f"\n⚠️  {no_yob} patients missing YEAR_OF_BIRTH")
    if no_date > 0:
        print(f"⚠️  {no_date} patients missing event date")

    n = len(ages_at_first_diag)
    ages_only = [a[0] for a in ages_at_first_diag]

    print(f"\n📊 Total patients with valid age at diagnosis: {n}")

    # ---- Summary statistics ----
    mean_age = sum(ages_only) / n
    sorted_ages = sorted(ages_only)
    median_age = sorted_ages[n // 2]
    q1 = sorted_ages[n // 4]
    q3 = sorted_ages[3 * n // 4]
    variance = sum((a - mean_age) ** 2 for a in ages_only) / n
    std_age = variance ** 0.5

    print(f"\n📈 Summary Statistics:")
    print(f"   Mean age at first diagnosis:   {mean_age:.1f} years")
    print(f"   Median age:                    {median_age} years")
    print(f"   Std deviation:                 {std_age:.1f} years")
    print(f"   Q1 (25th percentile):          {q1} years")
    print(f"   Q3 (75th percentile):          {q3} years")
    print(f"   IQR:                           {q3 - q1} years")
    print(f"   Minimum age:                   {sorted_ages[0]} years")
    print(f"   Maximum age:                   {sorted_ages[-1]} years")

    # ---- Age histogram (1-year bins) ----
    age_counts = Counter(ages_only)
    min_age = min(age_counts.keys())
    max_age = max(age_counts.keys())
    max_count = max(age_counts.values())
    bar_width = 50

    print(f"\n📅 Age Distribution (1-year bins):")
    print(f"   {'Age':<6} {'Count':>6}  {'%':>6}  Distribution")
    print(f"   {'-'*75}")

    for age in range(min_age, max_age + 1):
        count = age_counts.get(age, 0)
        pct = 100.0 * count / n
        bar_len = int(bar_width * count / max_count) if max_count > 0 else 0
        bar = '█' * bar_len
        if count > 0:
            print(f"   {age:<6} {count:>6}  {pct:>5.1f}%  {bar}")

    # ---- Age histogram (5-year bins) ----
    print(f"\n📅 Age Distribution (5-year bins):")
    bin_counts = Counter()
    for age in ages_only:
        bin_start = (age // 5) * 5
        bin_counts[bin_start] += 1

    max_bin_count = max(bin_counts.values())
    print(f"   {'Age Range':<12} {'Count':>6}  {'%':>6}  {'Cumul%':>7}  Distribution")
    print(f"   {'-'*80}")

    cumul = 0
    for bin_start in sorted(bin_counts.keys()):
        count = bin_counts[bin_start]
        cumul += count
        pct = 100.0 * count / n
        cumul_pct = 100.0 * cumul / n
        bar_len = int(bar_width * count / max_bin_count) if max_bin_count > 0 else 0
        bar = '█' * bar_len
        print(f"   {bin_start}-{bin_start+4:<7} {count:>6}  {pct:>5.1f}%  {cumul_pct:>6.1f}%  {bar}")

    # ---- Percentile table ----
    print(f"\n📐 Key Percentiles:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"   {'Percentile':>12}  {'Age':>5}")
    print(f"   {'-'*22}")
    for p in percentiles:
        idx = int(n * p / 100)
        idx = min(idx, n - 1)
        print(f"   {p:>10}th  {sorted_ages[idx]:>5}")

    # ---- Subtype-specific age ----
    print(f"\n🧠 Age at First Diagnosis by Dementia Subtype:")
    print(f"   {'Subtype':<22} {'N':>5}  {'Mean':>6}  {'Median':>7}  {'Min':>5}  {'Max':>5}  {'Std':>5}")
    print(f"   {'-'*70}")

    for subtype in ["Alzheimer's", "Vascular", "Other diseases", "Unspecified/Senile"]:
        s_ages = sorted(subtype_ages[subtype])
        if not s_ages:
            continue
        s_n = len(s_ages)
        s_mean = sum(s_ages) / s_n
        s_median = s_ages[s_n // 2]
        s_std = (sum((a - s_mean) ** 2 for a in s_ages) / s_n) ** 0.5
        print(f"   {subtype:<22} {s_n:>5}  {s_mean:>5.1f}  {s_median:>7}  {min(s_ages):>5}  {max(s_ages):>5}  {s_std:>5.1f}")

    # ---- Index date guidance ----
    print(f"\n{'='*90}")
    print("INDEX DATE / AGE GUIDANCE FOR FINE-TUNING")
    print(f"{'='*90}")

    # Count patients by age thresholds
    thresholds = [50, 55, 60, 65, 70, 75]
    print(f"\n   If index age = X, how many dementia patients diagnosed AFTER age X?")
    print(f"   {'Index Age':<12} {'Patients ≥ X':>14}  {'%':>6}  {'Patients < X':>14}  {'Lost %':>7}")
    print(f"   {'-'*65}")

    for threshold in thresholds:
        after = sum(1 for a in ages_only if a >= threshold)
        before = n - after
        pct_after = 100.0 * after / n
        pct_before = 100.0 * before / n
        marker = " ◀ recommended" if threshold == 65 else ""
        print(f"   {threshold:<12} {after:>14,}  {pct_after:>5.1f}%  {before:>14,}  {pct_before:>5.1f}%{marker}")

    # Patients diagnosed before certain ages (could be missed)
    print(f"\n   ⚠️  Patients diagnosed at very young ages (potential data quality issue):")
    young_patients = [(age, pid, dt, code) for age, pid, dt, code in ages_at_first_diag if age < 50]
    if young_patients:
        print(f"   {'Age':>5}  {'Patient ID':<15}  {'Date':<12}  {'Code':<8}")
        print(f"   {'-'*50}")
        for age, pid, dt, code in young_patients:
            print(f"   {age:>5}  {pid:<15}  {dt.strftime('%Y-%m-%d'):<12}  {code:<8}")
    else:
        print(f"   None found (all patients ≥ 50)")

    print(f"\n{'='*90}")
    print("CONCLUSION")
    print(f"{'='*90}")
    print(f"""
   ✅ {n} core dementia patients with valid diagnosis age

   📊 Typical diagnosis age: {q1}-{q3} years (IQR), median {median_age}
      Mean ± SD: {mean_age:.1f} ± {std_age:.1f} years

   💡 Index age recommendation for fine-tuning:
      - Age 65: captures {sum(1 for a in ages_only if a >= 65):,}/{n} ({100.0*sum(1 for a in ages_only if a >= 65)/n:.1f}%) patients
      - Age 60: captures {sum(1 for a in ages_only if a >= 60):,}/{n} ({100.0*sum(1 for a in ages_only if a >= 60)/n:.1f}%) patients
      - Age 55: captures {sum(1 for a in ages_only if a >= 55):,}/{n} ({100.0*sum(1 for a in ages_only if a >= 55)/n:.1f}%) patients
""")


if __name__ == "__main__":
    main()
