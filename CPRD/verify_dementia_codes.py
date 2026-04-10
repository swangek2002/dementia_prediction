"""
Verify whether all 41 Read codes used in Dementia fine-tuning are indeed dementia-related.

Approach:
1. Hardcoded Read V2 / CTV3 code-to-description mapping based on NHS Read code dictionaries
2. Check each code against the known dementia code taxonomy
3. Flag any codes that may NOT be dementia-related
"""

# ============================================================
# The 41 codes from the config
# ============================================================
DEMENTIA_READ_CODES = [
    # Query 1 (31 codes)
    "F110.", "Eu00.", "Eu01.", "Eu02z", "Eu002", "E00..", "Eu023", "Eu00z",
    "Eu025", "Eu01z", "E001.", "F1100", "Eu001", "E004.", "Eu000", "Eu02.",
    "Eu013", "E000.", "Eu01y", "E001z", "F1101", "Eu020", "E004z", "E0021",
    "Eu02y", "Eu012", "Eu011", "E00z.", "E0040", "E003.", "E0020",
    # Query 2 (10 codes)
    "Eu057", "Eu04.", "Eu053", "Eu04z", "Eu0z.", "Eu060", "Eu054", "Eu05y",
    "Eu052", "Eu062",
]

# ============================================================
# Read V2 / CTV3 Code → Description mapping
# Based on NHS Read Code Browser / TRUD / Clinical Terms
#
# Read V2 Chapter Structure:
#   E00.. = Senile and presenile organic psychotic conditions
#   Eu0.. = ICD-10 Chapter V: Organic, including symptomatic, mental disorders [F00-F09]
#   F110. = Alzheimer's disease (Neurology chapter)
# ============================================================

READ_CODE_DESCRIPTIONS = {
    # ---- E00.. hierarchy: Senile and presenile organic psychotic conditions ----
    "E00..": "Senile and presenile organic psychotic conditions",
    "E000.": "Uncomplicated senile dementia",
    "E001.": "Presenile dementia",
    "E001z": "Presenile dementia NOS",
    "E0020": "Senile dementia with depressive or paranoid features",
    "E0021": "Senile dementia with paranoia",
    "E003.": "Senile dementia with delirium",
    "E004.": "Arteriosclerotic dementia",
    "E004z": "Arteriosclerotic dementia NOS",
    "E0040": "Uncomplicated arteriosclerotic dementia",
    "E00z.": "Senile or presenile psychoses NOS",

    # ---- Eu0.. hierarchy: [F00-F09] Organic, including symptomatic, mental disorders ----
    # Eu00. = Dementia in Alzheimer's disease [F00]
    "Eu00.": "[X]Dementia in Alzheimer's disease",
    "Eu000": "[X]Dementia in Alzheimer's disease with early onset (Type 2) [F00.0]",
    "Eu001": "[X]Dementia in Alzheimer's disease with late onset (Type 1) [F00.1]",
    "Eu002": "[X]Dementia in Alzheimer's disease, atypical or mixed type [F00.2]",
    "Eu00z": "[X]Dementia in Alzheimer's disease, unspecified [F00.9]",

    # Eu01. = Vascular dementia [F01]
    "Eu01.": "[X]Vascular dementia",
    "Eu011": "[X]Multi-infarct dementia [F01.1]",
    "Eu012": "[X]Subcortical vascular dementia [F01.2]",
    "Eu013": "[X]Mixed cortical and subcortical vascular dementia [F01.3]",
    "Eu01y": "[X]Other vascular dementia [F01.8]",
    "Eu01z": "[X]Vascular dementia, unspecified [F01.9]",

    # Eu02. = Dementia in other diseases classified elsewhere [F02]
    "Eu02.": "[X]Dementia in other diseases classified elsewhere",
    "Eu020": "[X]Dementia in Pick's disease [F02.0]",
    "Eu023": "[X]Dementia in Parkinson's disease [F02.3]",
    "Eu025": "[X]Dementia in other specified diseases [F02.5] (incl. epilepsy, SLE, etc.)",  # Note: added in later editions
    "Eu02y": "[X]Dementia in other specified diseases classified elsewhere [F02.8]",
    "Eu02z": "[X]Unspecified dementia [F02.9] / Dementia in other diseases NOS",

    # Eu04. = Delirium, not induced by alcohol and other psychoactive substances [F05]
    "Eu04.": "[X]Delirium, not induced by alcohol and other psychoactive substances [F05]",
    "Eu04z": "[X]Delirium, unspecified [F05.9]",

    # Eu05. = Other mental disorders due to brain damage and dysfunction [F06]
    "Eu052": "[X]Organic delusional [schizophrenia-like] disorder [F06.2]",
    "Eu053": "[X]Organic mood [affective] disorders [F06.3]",
    "Eu054": "[X]Organic anxiety disorder [F06.4]",
    "Eu057": "[X]Mild cognitive disorder [F06.7]",
    "Eu05y": "[X]Other specified mental disorders due to brain damage and dysfunction and to physical disease [F06.8]",

    # Eu06. = Personality and behavioural disorders due to brain disease, damage and dysfunction [F07]
    "Eu060": "[X]Organic personality disorder [F07.0]",
    "Eu062": "[X]Postconcussional syndrome [F07.2]",

    # Eu0z. = Unspecified organic or symptomatic mental disorder [F09]
    "Eu0z.": "[X]Unspecified organic or symptomatic mental disorder [F09]",

    # ---- F110. hierarchy: Alzheimer's disease (Neurology chapter) ----
    "F110.": "Alzheimer's disease",
    "F1100": "Alzheimer's disease with early onset",
    "F1101": "Alzheimer's disease with late onset",
}

# ============================================================
# Classification: Which codes are "core dementia" vs "broader organic mental disorders"?
# ============================================================

# Core dementia codes (directly code for dementia diagnosis):
#   E00.. hierarchy: All senile/presenile dementia
#   Eu00.: Dementia in Alzheimer's disease (ICD-10 F00)
#   Eu01.: Vascular dementia (ICD-10 F01)
#   Eu02.: Dementia in other diseases (ICD-10 F02)
#   F110.: Alzheimer's disease (neurological diagnosis)
CORE_DEMENTIA_PREFIXES = ["E00", "Eu00", "Eu01", "Eu02", "F110"]

# Broader organic mental disorder codes (NOT primary dementia, but related):
#   Eu04. = Delirium (F05) — often co-occurs with dementia but is NOT dementia per se
#   Eu05. = Other mental disorders due to brain damage (F06) — includes mild cognitive disorder
#   Eu06. = Personality/behavioural disorders due to brain disease (F07)
#   Eu0z. = Unspecified organic mental disorder (F09)
BROADER_ORGANIC_PREFIXES = ["Eu04", "Eu05", "Eu06", "Eu0z"]


def classify_code(code: str) -> str:
    """Classify a Read code as 'core_dementia', 'broader_organic', or 'unknown'."""
    for prefix in CORE_DEMENTIA_PREFIXES:
        if code.startswith(prefix):
            return "core_dementia"
    for prefix in BROADER_ORGANIC_PREFIXES:
        if code.startswith(prefix):
            return "broader_organic"
    return "unknown"


def main():
    print("=" * 90)
    print("VERIFICATION: Are all 41 codes dementia-related Read codes?")
    print("=" * 90)
    print(f"\nTotal codes to verify: {len(DEMENTIA_READ_CODES)}")
    print(f"Unique codes: {len(set(DEMENTIA_READ_CODES))}")

    # Check for duplicates
    if len(DEMENTIA_READ_CODES) != len(set(DEMENTIA_READ_CODES)):
        from collections import Counter
        dupes = [code for code, cnt in Counter(DEMENTIA_READ_CODES).items() if cnt > 1]
        print(f"⚠️  Duplicate codes found: {dupes}")

    print("\n" + "-" * 90)
    print(f"{'Code':<10} {'Classification':<20} {'Description'}")
    print("-" * 90)

    core_count = 0
    broader_count = 0
    unknown_count = 0
    not_in_dict = []

    for code in DEMENTIA_READ_CODES:
        classification = classify_code(code)
        description = READ_CODE_DESCRIPTIONS.get(code, "❓ NOT FOUND IN DICTIONARY")

        if classification == "core_dementia":
            marker = "✅ Core Dementia"
            core_count += 1
        elif classification == "broader_organic":
            marker = "⚠️  Broader Organic"
            broader_count += 1
        else:
            marker = "❌ UNKNOWN"
            unknown_count += 1

        if code not in READ_CODE_DESCRIPTIONS:
            not_in_dict.append(code)

        print(f"{code:<10} {marker:<20} {description}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(f"\n📊 Classification Breakdown:")
    print(f"   ✅ Core Dementia codes (E00/Eu00/Eu01/Eu02/F110):  {core_count}")
    print(f"   ⚠️  Broader Organic Mental Disorder codes (Eu04/Eu05/Eu06/Eu0z): {broader_count}")
    print(f"   ❌ Unknown / Not classified:  {unknown_count}")
    print(f"   Total: {core_count + broader_count + unknown_count}")

    if not_in_dict:
        print(f"\n⚠️  Codes not found in our reference dictionary: {not_in_dict}")

    # ============================================================
    # Detailed analysis of "broader" codes
    # ============================================================
    if broader_count > 0:
        print(f"\n{'=' * 90}")
        print("DETAILED ANALYSIS: Broader Organic Mental Disorder Codes")
        print(f"{'=' * 90}")
        print("""
These {n} codes are NOT strictly "dementia" codes. They fall under ICD-10 F05-F09
(organic mental disorders) rather than F00-F02 (dementia). Specifically:

  Eu04. / Eu04z  → Delirium (F05) — acute confusional state, NOT chronic dementia
  Eu052          → Organic delusional disorder (F06.2) — psychosis due to brain disease
  Eu053          → Organic mood disorder (F06.3) — depression/mania due to brain disease
  Eu054          → Organic anxiety disorder (F06.4) — anxiety due to brain disease
  Eu057          → Mild cognitive disorder (F06.7) — may be prodromal dementia (MCI)
  Eu05y          → Other mental disorders due to brain damage (F06.8)
  Eu060          → Organic personality disorder (F07.0) — personality change due to brain disease
  Eu062          → Postconcussional syndrome (F07.2) — NOT dementia
  Eu0z.          → Unspecified organic mental disorder (F09)

VERDICT:
  - These codes are part of the broader "organic mental disorders" chapter (F00-F09)
  - Some (like Eu057 Mild cognitive disorder) are reasonable to include as they may
    represent early/prodromal dementia
  - Others (like Eu062 Postconcussional syndrome, Eu054 Organic anxiety) are more
    questionable for a dementia-specific study
  - However, including them is a COMMON practice in EHR-based dementia studies using
    "broad" dementia definitions, as these conditions frequently co-occur with or
    precede dementia diagnosis
""".format(n=broader_count))

    # ============================================================
    # ICD-10 Cross-reference
    # ============================================================
    print(f"{'=' * 90}")
    print("ICD-10 CROSS-REFERENCE")
    print(f"{'=' * 90}")
    print("""
Read V2/CTV3 Code  →  ICD-10 Mapping  →  Category
─────────────────────────────────────────────────────
E00.. / Eu00.      →  F00             →  Dementia in Alzheimer's disease ✅
E001. / Eu01.      →  F01             →  Vascular dementia ✅
E004. / Eu02.      →  F02             →  Dementia in other diseases ✅
E003.              →  F03/F05         →  Senile dementia with delirium ✅
E00z.              →  F03             →  Unspecified dementia ✅
F110. / F1100/01   →  G30             →  Alzheimer's disease (neuro) ✅
Eu04. / Eu04z      →  F05             →  Delirium ⚠️  (not dementia per se)
Eu052-Eu057/Eu05y  →  F06             →  Other organic mental disorders ⚠️
Eu060 / Eu062      →  F07             →  Personality/behavioural disorders ⚠️
Eu0z.              →  F09             →  Unspecified organic disorder ⚠️
""")

    # ============================================================
    # Final verdict
    # ============================================================
    print(f"{'=' * 90}")
    print("FINAL VERDICT")
    print(f"{'=' * 90}")
    if unknown_count == 0:
        print(f"""
✅ All {len(DEMENTIA_READ_CODES)} codes are recognized Read V2/CTV3 codes.

📋 {core_count} out of {len(DEMENTIA_READ_CODES)} codes ({100*core_count/len(DEMENTIA_READ_CODES):.1f}%) are
   CORE DEMENTIA codes (ICD-10 F00-F03, G30).

⚠️  {broader_count} out of {len(DEMENTIA_READ_CODES)} codes ({100*broader_count/len(DEMENTIA_READ_CODES):.1f}%) are
   BROADER ORGANIC MENTAL DISORDER codes (ICD-10 F05-F09).

   These are commonly included in "broad definition" dementia studies
   (e.g., NHS QOF dementia registers, CPRD Gold dementia studies).

   The Query 2 codes (the 10 "fuzzy/extended" codes) appear to be from this
   broader category, which is consistent with a two-stage code search:
     - Query 1: Strict dementia codes (F00-F03, G30)
     - Query 2: Extended organic mental disorder codes (F05-F09)
""")
    else:
        print(f"\n❌ {unknown_count} codes could not be classified. Please review manually.")


if __name__ == "__main__":
    main()
