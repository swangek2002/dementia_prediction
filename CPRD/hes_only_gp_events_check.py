import sqlite3, csv

conn = sqlite3.connect('/Data0/swangek_data/991/CPRD/data/example_exercise_database.db')
cur = conn.cursor()

# Get HES-only dementia patients
GP_DEMENTIA = [
    'F110.','Eu00.','Eu01.','Eu02z','Eu002','E00..','Eu023','Eu00z','Eu025',
    'Eu01z','E001.','F1100','Eu001','E004.','Eu000','Eu02.','Eu013','E000.','Eu01y',
    'E001z','F1101','Eu020','E004z','E0021','Eu02y','Eu012','Eu011','E00z.','E0040',
    'E003.','E0020',
]
placeholders = ','.join(['?'] * len(GP_DEMENTIA))
cur.execute(f"SELECT DISTINCT PATIENT_ID FROM diagnosis_table WHERE EVENT IN ({placeholders})", GP_DEMENTIA)
gp_dementia = set(r[0] for r in cur.fetchall())

hes_dementia = set()
with open('/Data0/swangek_data/991/CPRD/data/hesin_diag.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        icd = row.get('diag_icd10', '')
        if icd and (icd.startswith('F00') or icd.startswith('F01') or
                    icd.startswith('F02') or icd == 'F03' or icd.startswith('G30')):
            hes_dementia.add(int(row['eid']))

hes_only = hes_dementia - gp_dementia

# Check: do these HES-only patients have GP event sequences?
has_gp_events = 0
no_gp_events = 0
event_counts = []

for pid in hes_only:
    cur.execute("SELECT COUNT(*) FROM diagnosis_table WHERE PATIENT_ID = ?", (pid,))
    n = cur.fetchone()[0]
    if n > 0:
        has_gp_events += 1
        event_counts.append(n)
    else:
        no_gp_events += 1

import numpy as np
event_counts = np.array(event_counts)

print(f"HES-only dementia patients: {len(hes_only)}")
print(f"  Have GP event sequence: {has_gp_events} ({100*has_gp_events/len(hes_only):.1f}%)")
print(f"  No GP events at all:   {no_gp_events} ({100*no_gp_events/len(hes_only):.1f}%)")
if len(event_counts) > 0:
    print(f"  GP events per patient: min={event_counts.min()}, median={np.median(event_counts):.0f}, max={event_counts.max()}")

conn.close()
