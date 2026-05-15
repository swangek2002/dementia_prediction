"""
Generic cohort-level C_td (cause-specific + true CR) for any saved inference NPZ.

Usage:
    python compute_cohort_ctd_generic.py <npz_path> [model_tag]
"""
import sys
import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv

# Re-import true_cr_ctd from compute_v5_cohort_ctd
sys.path.insert(0, '/Data0/swangek_data/991/CPRD')
from compute_v5_cohort_ctd import true_cr_ctd, cause_specific_ctd, cohort_ibs_inbll


LEAKY_TXT = "/Data0/swangek_data/991/CPRD/data/leaky_patients_test.txt"


def compute_all_metrics(npz_path, tag=None, do_bootstrap=False, n_boots=200):
    if tag is None:
        tag = npz_path.split('/')[-1].replace('test_cif_', '').replace('_full.npz', '')
    print(f"\n{'='*70}")
    print(f"COHORT-LEVEL METRICS FOR: {tag}")
    print(f"  Source: {npz_path}")
    print(f"{'='*70}")

    data = np.load(npz_path, allow_pickle=True)
    pids = data['patient_ids']
    labels = data['labels']
    event_time = data['event_time_scaled']
    cif_dem = data['cif_dementia']
    cif_dth = data['cif_death']
    t_eval = data['t_eval']

    print(f"  N_patients (full)      = {len(pids)}")
    print(f"  Label dist (full)      = {pd.Series(labels).value_counts().to_dict()}")

    # Build canonical clean cohort: V5 PIDs (= V2 test) minus 16 GP-prevalent leaky
    v5_data = np.load('/Data0/swangek_data/991/CPRD/data/test_cif_v5_full.npz', allow_pickle=True)
    v5_pids = set(int(p) for p in v5_data['patient_ids'])
    with open(LEAKY_TXT) as f:
        leaky_pids = set(int(line.strip()) for line in f if line.strip())
    canonical_clean_pids = v5_pids - leaky_pids

    # Check overlap with this model's test set
    overlap = sum(1 for p in pids if int(p) in canonical_clean_pids)
    if overlap > 0.5 * len(canonical_clean_pids):
        # Significant overlap → this is an idx72-cohort model, filter to canonical
        print(f"  Canonical clean cohort = {len(canonical_clean_pids)} PIDs (V5 test minus 16 leaky)")
        keep_mask = np.isin(pids, list(canonical_clean_pids))
        pids_c = pids[keep_mask]
        labels_c = labels[keep_mask]
        event_time_c = event_time[keep_mask]
        cif_dem_c = cif_dem[keep_mask]
        cif_dth_c = cif_dth[keep_mask]
        print(f"  N_patients (cleaned)   = {len(pids_c)}")
        if len(pids_c) != len(canonical_clean_pids):
            n_missing = len(canonical_clean_pids) - len(pids_c)
            print(f"  WARN: {n_missing} canonical-clean PIDs not present in this model's inference set")
    else:
        # Different cohort (e.g. idx60/70/74/75 or fusion expanded) → use model's own test set
        print(f"  This model's cohort overlaps poorly with canonical clean ({overlap}/{len(canonical_clean_pids)})")
        print(f"  Falling back to model's OWN test cohort (drop label=='empty'/'unknown' only)")
        valid = (labels != 'empty') & (labels != 'unknown')
        pids_c = pids[valid]
        labels_c = labels[valid]
        event_time_c = event_time[valid]
        cif_dem_c = cif_dem[valid]
        cif_dth_c = cif_dth[valid]
        print(f"  N_patients (model own cohort) = {len(pids_c)}")

    # Build event codes: 0=censored, 1=dementia, 2=death
    event_codes = np.zeros(len(labels_c), dtype=np.int64)
    event_codes[labels_c == 'dementia'] = 1
    event_codes[labels_c == 'death'] = 2

    # Cause-specific Dementia
    evt_dem_cs = (event_codes == 1).astype(np.int64)
    cs_dem = cause_specific_ctd(cif_dem_c, event_time_c, evt_dem_cs, t_eval)

    # True CR Dementia
    tcr_dem, n_conc_dem, n_pair_dem = true_cr_ctd(cif_dem_c, event_time_c, event_codes, t_eval, target_cause=1)

    # Cause-specific Death
    evt_dth_cs = (event_codes == 2).astype(np.int64)
    cs_dth = cause_specific_ctd(cif_dth_c, event_time_c, evt_dth_cs, t_eval)

    # True CR Death
    event_codes_for_death = np.zeros_like(event_codes)
    event_codes_for_death[event_codes == 2] = 1
    event_codes_for_death[event_codes == 1] = 2
    tcr_dth, n_conc_dth, n_pair_dth = true_cr_ctd(cif_dth_c, event_time_c, event_codes_for_death, t_eval, target_cause=1)

    # Overall
    cif_any = cif_dem_c + cif_dth_c
    evt_any = ((event_codes == 1) | (event_codes == 2)).astype(np.int64)
    cs_any = cause_specific_ctd(cif_any, event_time_c, evt_any, t_eval)

    # IBS / INBLL
    ibs_dem, inbll_dem = cohort_ibs_inbll(cif_dem_c, event_time_c, evt_dem_cs, t_eval)
    ibs_dth, inbll_dth = cohort_ibs_inbll(cif_dth_c, event_time_c, evt_dth_cs, t_eval)

    results = {
        'tag': tag,
        'n_full': len(pids),
        'n_clean': len(pids_c),
        'dementia_cs': cs_dem,
        'dementia_tcr': tcr_dem,
        'death_cs': cs_dth,
        'death_tcr': tcr_dth,
        'overall_cs': cs_any,
        'ibs_dem': ibs_dem,
        'inbll_dem': inbll_dem,
        'ibs_dth': ibs_dth,
        'inbll_dth': inbll_dth,
    }

    if do_bootstrap:
        np.random.seed(1337)
        N = len(pids_c)
        cs_boots = []
        tcr_boots = []
        for b in range(n_boots):
            idx = np.random.choice(N, N, replace=True)
            cif_b = cif_dem_c[idx]
            et_b = event_time_c[idx]
            ec_b = event_codes[idx]
            evt_b = (ec_b == 1).astype(np.int64)
            if evt_b.sum() == 0:
                continue
            try:
                cs_boots.append(cause_specific_ctd(cif_b, et_b, evt_b, t_eval))
            except: pass
            try:
                tcr_b, _, _ = true_cr_ctd(cif_b, et_b, ec_b, t_eval, target_cause=1)
                if not np.isnan(tcr_b):
                    tcr_boots.append(tcr_b)
            except: pass
        results['cs_dem_ci'] = (np.percentile(cs_boots, 2.5), np.percentile(cs_boots, 97.5))
        results['tcr_dem_ci'] = (np.percentile(tcr_boots, 2.5), np.percentile(tcr_boots, 97.5))

    print(f"\n  Dementia C_td:")
    print(f"    Cause-specific (pycox)  = {cs_dem:.4f}")
    print(f"    True CR (Wolbers)       = {tcr_dem:.4f}  ({int(n_conc_dem)}/{int(n_pair_dem)} pairs)")
    if 'cs_dem_ci' in results:
        ci = results['cs_dem_ci']
        print(f"    Cause-specific 95% CI   = [{ci[0]:.4f}, {ci[1]:.4f}]")
        ci = results['tcr_dem_ci']
        print(f"    True CR 95% CI          = [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  Death C_td:")
    print(f"    Cause-specific (pycox)  = {cs_dth:.4f}")
    print(f"    True CR (Wolbers)       = {tcr_dth:.4f}  ({int(n_conc_dth)}/{int(n_pair_dth)} pairs)")
    print(f"  Overall (any-event) C_td  = {cs_any:.4f}")
    print(f"  IBS    dementia / death   = {ibs_dem:.4f} / {ibs_dth:.4f}")
    print(f"  INBLL  dementia / death   = {inbll_dem:.4f} / {inbll_dth:.4f}")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_cohort_ctd_generic.py <npz_path> [tag]")
        sys.exit(1)
    npz_path = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else None
    do_boot = '--bootstrap' in sys.argv
    compute_all_metrics(npz_path, tag, do_bootstrap=do_boot)
