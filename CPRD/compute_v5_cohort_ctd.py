"""
V5 Cohort-level C_td computation: both cause-specific (pycox) and true competing-risk (Wolbers).

Steps:
1. Load NPZ + leaky patient list
2. Filter to cleaned cohort (8241)
3. Cause-specific C_td via pycox EvalSurv
4. True CR C_td via custom Wolbers implementation
5. Verify Wolbers implementation against synthetic test cases
6. IBS, INBLL cohort-level
7. Bootstrap 95% CI for headline metric

Usage:
    python compute_v5_cohort_ctd.py
"""
import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
import sys

# ================================================================
# Part A: True Competing Risk C_td implementation (Wolbers 2014 + Antolini 2005)
# ================================================================

def true_cr_ctd(cif_curves, target_times, event_codes, t_eval, target_cause=1):
    """
    Compute Wolbers-style True Competing Risk time-dependent C-index.

    For each pair (i, j) where event_i == target_cause and T_i < T_j:
        - j must be 'at risk' at T_i (no event of any kind before T_i)
        - Concordant if CIF_target(T_i; x_i) > CIF_target(T_i; x_j)

    Critical: patients who had a COMPETING event before T_i are NOT comparable
    (they are precluded from target cause), unlike cause-specific which treats
    them as 'censored' at their event time.

    Args:
        cif_curves: (N, T) - CIF for target_cause for each patient over time grid
        target_times: (N,) - event/censoring time
        event_codes: (N,) - 0=censored, 1=target_cause, 2=competing_event
        t_eval: (T,) - time grid for CIF curves
        target_cause: int - cause label for which we compute C-index (default 1)

    Returns:
        (concordance, n_concordant, n_comparable)
    """
    N = len(target_times)
    target_times = np.asarray(target_times, dtype=np.float64)
    event_codes = np.asarray(event_codes, dtype=np.int64)
    t_eval = np.asarray(t_eval, dtype=np.float64)

    target_event_idx = np.where(event_codes == target_cause)[0]

    n_concordant = 0.0
    n_comparable = 0.0

    # Precompute CIF at each patient's event time, interpolated
    # For each i with event, we'll need cif_j(T_i) for all j

    for i in target_event_idx:
        T_i = target_times[i]
        # Interpolate CIF_i at T_i
        cif_i_at_Ti = np.interp(T_i, t_eval, cif_curves[i])

        # For each other patient j, check comparability
        # Vectorize: comparable if T_j > T_i (j was at risk at T_i)
        #            tied if T_j == T_i (handled with 0.5 weight)
        mask_greater = target_times > T_i
        mask_equal = (target_times == T_i) & (np.arange(N) != i)

        # Vectorized interp for all j with T_j > T_i
        if mask_greater.sum() > 0:
            # Find left/right index in t_eval for T_i, then linear interpolate
            idx_right = np.searchsorted(t_eval, T_i)
            if idx_right >= len(t_eval):
                idx_right = len(t_eval) - 1
                idx_left = idx_right
                cif_j_at_Ti = cif_curves[mask_greater, idx_right]
            elif idx_right == 0:
                cif_j_at_Ti = cif_curves[mask_greater, 0]
            else:
                idx_left = idx_right - 1
                t_left = t_eval[idx_left]
                t_right = t_eval[idx_right]
                if t_right == t_left:
                    alpha = 0.0
                else:
                    alpha = (T_i - t_left) / (t_right - t_left)
                cif_j_at_Ti = (1 - alpha) * cif_curves[mask_greater, idx_left] + alpha * cif_curves[mask_greater, idx_right]

            n_comparable += mask_greater.sum()
            n_concordant += (cif_i_at_Ti > cif_j_at_Ti).sum()
            n_concordant += 0.5 * (cif_i_at_Ti == cif_j_at_Ti).sum()

        # Handle ties (T_j == T_i) with 0.5 weight
        if mask_equal.sum() > 0:
            idx_right = np.searchsorted(t_eval, T_i)
            if idx_right >= len(t_eval):
                idx_right = len(t_eval) - 1
                cif_j_at_Ti = cif_curves[mask_equal, idx_right]
            elif idx_right == 0:
                cif_j_at_Ti = cif_curves[mask_equal, 0]
            else:
                idx_left = idx_right - 1
                t_left = t_eval[idx_left]
                t_right = t_eval[idx_right]
                if t_right == t_left:
                    alpha = 0.0
                else:
                    alpha = (T_i - t_left) / (t_right - t_left)
                cif_j_at_Ti = (1 - alpha) * cif_curves[mask_equal, idx_left] + alpha * cif_curves[mask_equal, idx_right]
            n_comparable += 0.5 * mask_equal.sum()
            n_concordant += 0.5 * (cif_i_at_Ti > cif_j_at_Ti).sum()
            n_concordant += 0.25 * (cif_i_at_Ti == cif_j_at_Ti).sum()

    if n_comparable == 0:
        return float('nan'), 0, 0
    return n_concordant / n_comparable, n_concordant, n_comparable


# ================================================================
# Part B: Verification on synthetic data
# ================================================================

def verify_true_cr():
    """Verify true_cr_ctd on hand-checkable synthetic cases."""
    print("\n========== Verifying true_cr_ctd on synthetic cases ==========")
    t_eval = np.linspace(0, 1, 1000)

    # Case 1: Perfect ranking, 3 patients, 1 target event, 2 censored
    # A: target event at T=0.3, high CIF_dem
    # B: censored at T=0.9, medium CIF_dem
    # C: censored at T=0.9, low CIF_dem
    # Expected: C = 1.0 (A > B and A > C at T_A=0.3)

    cif = np.zeros((3, 1000), dtype=np.float64)
    # CIF curves grow monotonically; just set the values at intermediate points
    cif[0] = np.linspace(0, 0.9, 1000)  # A: high CIF
    cif[1] = np.linspace(0, 0.5, 1000)  # B: medium
    cif[2] = np.linspace(0, 0.1, 1000)  # C: low

    target_times = np.array([0.3, 0.9, 0.9])
    event_codes = np.array([1, 0, 0])  # A=dementia, B,C=censored

    c, nc, nt = true_cr_ctd(cif, target_times, event_codes, t_eval, target_cause=1)
    expected = 1.0
    print(f"  Case 1 (perfect ranking):  C={c:.4f}, expected={expected}, n_pairs={nt}, n_conc={nc}")
    assert abs(c - expected) < 1e-6, f"FAILED: {c} != {expected}"

    # Case 2: Perfect REVERSE ranking
    # A: target at T=0.3, LOW CIF; B,C censored at 0.9, higher CIFs
    cif = np.zeros((3, 1000), dtype=np.float64)
    cif[0] = np.linspace(0, 0.1, 1000)  # A: low CIF
    cif[1] = np.linspace(0, 0.5, 1000)
    cif[2] = np.linspace(0, 0.9, 1000)

    c, nc, nt = true_cr_ctd(cif, target_times, event_codes, t_eval, target_cause=1)
    expected = 0.0
    print(f"  Case 2 (reverse ranking):  C={c:.4f}, expected={expected}, n_pairs={nt}, n_conc={nc}")
    assert abs(c - expected) < 1e-6, f"FAILED: {c} != {expected}"

    # Case 3: Competing event PRECLUSION test
    # A: dementia at T=0.6, mid CIF
    # B: death (competing) at T=0.3 (BEFORE A), some CIF
    # C: censored at T=0.9, low CIF
    # Expected: B is precluded (not comparable to A). Only pair (A, C) counts.

    cif = np.zeros((3, 1000), dtype=np.float64)
    cif[0] = np.linspace(0, 0.6, 1000)  # A
    cif[1] = np.linspace(0, 0.9, 1000)  # B (high, but precluded)
    cif[2] = np.linspace(0, 0.1, 1000)  # C (low)

    target_times = np.array([0.6, 0.3, 0.9])
    event_codes = np.array([1, 2, 0])  # A=dementia, B=death, C=censored

    c, nc, nt = true_cr_ctd(cif, target_times, event_codes, t_eval, target_cause=1)
    # Expected: only (A,C) comparable. CIF_A(0.6) > CIF_C(0.6) → concordant.
    # n_comparable=1, n_concordant=1, C=1.0
    expected_pairs = 1
    expected_c = 1.0
    print(f"  Case 3 (B precluded):      C={c:.4f}, expected_C={expected_c}, n_pairs={nt} (expected {expected_pairs}), n_conc={nc}")
    assert abs(c - expected_c) < 1e-6 and nt == expected_pairs, f"FAILED"

    # Case 4: Competing event AFTER target event (j had death after T_i)
    # A: dementia at T=0.3, mid CIF
    # B: death (competing) at T=0.9 (AFTER A), high CIF
    # Expected: B IS comparable to A (B was at risk at T_A=0.3)

    cif = np.zeros((3, 1000), dtype=np.float64)
    cif[0] = np.linspace(0, 0.6, 1000)  # A: mid
    cif[1] = np.linspace(0, 0.9, 1000)  # B: high CIF (will be wrong because B died but model thought dementia)
    cif[2] = np.linspace(0, 0.1, 1000)  # C: low

    target_times = np.array([0.3, 0.9, 0.9])
    event_codes = np.array([1, 2, 0])  # A=dementia, B=death (after A), C=censored

    c, nc, nt = true_cr_ctd(cif, target_times, event_codes, t_eval, target_cause=1)
    # Both B and C comparable. CIF_A(0.3) = 0.18, CIF_B(0.3) = 0.27, CIF_C(0.3) = 0.03
    # A < B: discordant
    # A > C: concordant
    # Expected: 1/2 = 0.5
    expected_c = 0.5
    print(f"  Case 4 (B after A):        C={c:.4f}, expected={expected_c}, n_pairs={nt} (expected 2), n_conc={nc}")
    assert abs(c - expected_c) < 1e-6, f"FAILED: {c} != {expected_c}"

    # Case 5: Two target events
    # A: dementia at T=0.2, high CIF
    # B: dementia at T=0.5, mid CIF
    # C: censored at T=0.9, low CIF
    # Pairs: (A,B), (A,C), (B,C)
    # At T_A=0.2: A is being compared against B (still alive) and C → CIF_A > CIF_B, CIF_A > CIF_C
    # At T_B=0.5: B is being compared against C (still alive) → CIF_B > CIF_C
    # A is NOT compared again as 'j' to B because T_A < T_B so when iterating B, A is in the past (event happened earlier).
    # Wait — when i=B, we look for j with T_j > T_B. A has T_A=0.2 < T_B=0.5, so A is NOT comparable to B (in our formulation).
    # Right. So pairs: (A vs B at T_A), (A vs C at T_A), (B vs C at T_B) = 3 pairs.

    cif = np.zeros((3, 1000), dtype=np.float64)
    cif[0] = np.linspace(0, 0.9, 1000)
    cif[1] = np.linspace(0, 0.5, 1000)
    cif[2] = np.linspace(0, 0.1, 1000)

    target_times = np.array([0.2, 0.5, 0.9])
    event_codes = np.array([1, 1, 0])  # A=dementia, B=dementia, C=censored

    c, nc, nt = true_cr_ctd(cif, target_times, event_codes, t_eval, target_cause=1)
    expected_c = 1.0  # all 3 pairs concordant
    expected_pairs = 3
    print(f"  Case 5 (two target events): C={c:.4f}, expected_C={expected_c}, n_pairs={nt} (expected {expected_pairs}), n_conc={nc}")
    assert abs(c - expected_c) < 1e-6 and nt == expected_pairs, f"FAILED"

    # Case 6: Tie in CIF
    # A: dementia at T=0.3, CIF=0.5
    # B: censored at T=0.9, CIF=0.5 (SAME as A's curve, exactly tied)
    cif = np.zeros((2, 1000), dtype=np.float64)
    cif[0] = np.linspace(0, 1.0, 1000)  # A
    cif[1] = np.linspace(0, 1.0, 1000)  # B exactly same

    target_times = np.array([0.3, 0.9])
    event_codes = np.array([1, 0])

    c, nc, nt = true_cr_ctd(cif, target_times, event_codes, t_eval, target_cause=1)
    expected_c = 0.5
    print(f"  Case 6 (CIF tied):          C={c:.4f}, expected={expected_c}, n_pairs={nt}, n_conc={nc}")
    assert abs(c - expected_c) < 1e-6, f"FAILED"

    print("\n  ALL 6 SYNTHETIC TESTS PASSED.")
    return True


# ================================================================
# Part C: Cause-specific C_td via pycox (wrapper for consistent interface)
# ================================================================

def cause_specific_ctd(cif_curves, target_times, evt_indicator, t_eval):
    """
    Compute Antolini's C_td treating competing events as censoring.

    Args:
        cif_curves: (N, T) - CIF for target cause
        target_times: (N,) - event/censoring time
        evt_indicator: (N,) - 1 if target event, 0 otherwise (including competing events)
        t_eval: (T,) - time grid

    Returns:
        C_td (cause-specific)
    """
    surv_df = pd.DataFrame((1.0 - cif_curves).T, index=t_eval)
    ev = EvalSurv(surv_df, np.asarray(target_times, dtype=np.float64),
                  np.asarray(evt_indicator, dtype=np.int64), censor_surv='km')
    return ev.concordance_td('antolini')


# ================================================================
# Part D: IBS, INBLL cohort-level
# ================================================================

def cohort_ibs_inbll(cif_curves, target_times, evt_indicator, t_eval, n_time_grid=300):
    """Compute IBS and INBLL via pycox at cohort level."""
    surv_df = pd.DataFrame((1.0 - cif_curves).T, index=t_eval)
    ev = EvalSurv(surv_df, np.asarray(target_times, dtype=np.float64),
                  np.asarray(evt_indicator, dtype=np.int64), censor_surv='km')
    time_grid = np.linspace(0, t_eval.max(), n_time_grid)
    ibs = ev.integrated_brier_score(time_grid)
    inbll = ev.integrated_nbll(time_grid)
    return ibs, inbll


# ================================================================
# Part E: Main V5 computation
# ================================================================

def main():
    # ----- Step 1: Verify true CR implementation -----
    verify_true_cr()

    # ----- Step 2: Load V5 NPZ -----
    print("\n========== Loading V5 NPZ ==========")
    data = np.load('/Data0/swangek_data/991/CPRD/data/test_cif_v5_full.npz', allow_pickle=True)
    pids = data['patient_ids']
    labels = data['labels']
    event_time = data['event_time_scaled']
    cif_dem = data['cif_dementia']
    cif_dth = data['cif_death']
    t_eval = data['t_eval']

    print(f"  N_patients = {len(pids)}")
    print(f"  Labels: {pd.Series(labels).value_counts().to_dict()}")

    # ----- Step 3: Filter out leaky patients -----
    with open('/Data0/swangek_data/991/CPRD/data/leaky_patients_test.txt') as f:
        leaky_pids = set(int(line.strip()) for line in f if line.strip())
    print(f"\n  Leaky PIDs to exclude: {len(leaky_pids)}")

    keep_mask = ~np.isin(pids, list(leaky_pids))
    pids_c = pids[keep_mask]
    labels_c = labels[keep_mask]
    event_time_c = event_time[keep_mask]
    cif_dem_c = cif_dem[keep_mask]
    cif_dth_c = cif_dth[keep_mask]
    print(f"  After cleaning: N={len(pids_c)} (expected 8241)")

    # ----- Step 4: Build event codes -----
    # 0=censored, 1=dementia, 2=death
    event_codes = np.zeros(len(labels_c), dtype=np.int64)
    event_codes[labels_c == 'dementia'] = 1
    event_codes[labels_c == 'death'] = 2

    n_dem = (event_codes == 1).sum()
    n_dth = (event_codes == 2).sum()
    n_cen = (event_codes == 0).sum()
    print(f"  Events: {n_dem} dementia, {n_dth} death, {n_cen} censored")

    # ----- Step 5: Compute all metrics -----
    print("\n========== Computing V5 cohort-level metrics ==========")

    results = {}

    # ----- 5a: Dementia C_td -----
    # Cause-specific: dementia=1, others (death, censored) = 0
    evt_dem_cs = (event_codes == 1).astype(np.int64)
    cs_dem = cause_specific_ctd(cif_dem_c, event_time_c, evt_dem_cs, t_eval)

    # True CR: dementia=1, death=2 (competing), censored=0
    tcr_dem, n_conc_dem, n_pair_dem = true_cr_ctd(cif_dem_c, event_time_c, event_codes, t_eval, target_cause=1)

    results['dementia_cs'] = cs_dem
    results['dementia_tcr'] = tcr_dem
    print(f"\n  Dementia C_td:")
    print(f"    Cause-specific (pycox)        = {cs_dem:.4f}")
    print(f"    True CR (Wolbers)             = {tcr_dem:.4f}  ({int(n_conc_dem)}/{int(n_pair_dem)} pairs)")

    # ----- 5b: Death C_td -----
    evt_dth_cs = (event_codes == 2).astype(np.int64)
    cs_dth = cause_specific_ctd(cif_dth_c, event_time_c, evt_dth_cs, t_eval)
    # For true CR death: target_cause=2, swap so death=1, dementia=2 (competing)
    event_codes_for_death = np.zeros_like(event_codes)
    event_codes_for_death[event_codes == 2] = 1  # death becomes target
    event_codes_for_death[event_codes == 1] = 2  # dementia becomes competing
    # event_codes_for_death[event_codes == 0] stays 0
    tcr_dth, n_conc_dth, n_pair_dth = true_cr_ctd(cif_dth_c, event_time_c, event_codes_for_death, t_eval, target_cause=1)

    results['death_cs'] = cs_dth
    results['death_tcr'] = tcr_dth
    print(f"\n  Death C_td:")
    print(f"    Cause-specific (pycox)        = {cs_dth:.4f}")
    print(f"    True CR (Wolbers)             = {tcr_dth:.4f}  ({int(n_conc_dth)}/{int(n_pair_dth)} pairs)")

    # ----- 5c: Overall C_td (any event) -----
    cif_any = cif_dem_c + cif_dth_c
    evt_any = ((event_codes == 1) | (event_codes == 2)).astype(np.int64)
    cs_any = cause_specific_ctd(cif_any, event_time_c, evt_any, t_eval)
    # For "any event" true CR doesn't really apply (no competing event in this framing)
    # Just report cause-specific style

    results['overall_cs'] = cs_any
    print(f"\n  Overall (any event) C_td:")
    print(f"    Cause-specific (pycox)        = {cs_any:.4f}")

    # ----- 5d: IBS / INBLL for Dementia & Death -----
    print(f"\n  IBS / INBLL (cohort-level):")
    ibs_dem, inbll_dem = cohort_ibs_inbll(cif_dem_c, event_time_c, evt_dem_cs, t_eval)
    ibs_dth, inbll_dth = cohort_ibs_inbll(cif_dth_c, event_time_c, evt_dth_cs, t_eval)

    results['ibs_dem'] = ibs_dem
    results['inbll_dem'] = inbll_dem
    results['ibs_dth'] = ibs_dth
    results['inbll_dth'] = inbll_dth
    print(f"    Dementia IBS    = {ibs_dem:.4f}")
    print(f"    Dementia INBLL  = {inbll_dem:.4f}")
    print(f"    Death    IBS    = {ibs_dth:.4f}")
    print(f"    Death    INBLL  = {inbll_dth:.4f}")

    # ----- 5e: Bootstrap CI for headline dementia C_td -----
    print(f"\n  Bootstrap 95% CI for V5 dementia C_td (1000 reps)...")
    np.random.seed(1337)
    n_boots = 1000
    cs_boots = []
    tcr_boots = []
    N = len(pids_c)
    for b in range(n_boots):
        idx = np.random.choice(N, N, replace=True)
        cif_b = cif_dem_c[idx]
        et_b = event_time_c[idx]
        ec_b = event_codes[idx]
        evt_dem_b = (ec_b == 1).astype(np.int64)
        if evt_dem_b.sum() == 0:
            continue
        try:
            cs_b = cause_specific_ctd(cif_b, et_b, evt_dem_b, t_eval)
            cs_boots.append(cs_b)
        except Exception:
            pass
        try:
            tcr_b, _, _ = true_cr_ctd(cif_b, et_b, ec_b, t_eval, target_cause=1)
            if not np.isnan(tcr_b):
                tcr_boots.append(tcr_b)
        except Exception:
            pass
        if (b + 1) % 100 == 0:
            print(f"    ... {b+1}/{n_boots} bootstrap reps done")
    cs_boots = np.array(cs_boots)
    tcr_boots = np.array(tcr_boots)
    cs_ci = (np.percentile(cs_boots, 2.5), np.percentile(cs_boots, 97.5))
    tcr_ci = (np.percentile(tcr_boots, 2.5), np.percentile(tcr_boots, 97.5))

    results['cs_dem_ci'] = cs_ci
    results['tcr_dem_ci'] = tcr_ci

    print(f"\n    Cause-specific Dementia C_td 95% CI = [{cs_ci[0]:.4f}, {cs_ci[1]:.4f}]  (point={cs_dem:.4f})")
    print(f"    True CR Dementia C_td 95% CI        = [{tcr_ci[0]:.4f}, {tcr_ci[1]:.4f}]  (point={tcr_dem:.4f})")

    # ----- Final Summary -----
    print(f"\n{'='*70}")
    print("V5 COHORT-LEVEL HEADLINE METRICS (cleaned cohort, N=8241)")
    print(f"{'='*70}")
    print(f"{'Metric':<35} {'Cause-Specific':<18} {'True CR (Wolbers)':<18}")
    print(f"{'-'*70}")
    print(f"{'Dementia C_td':<35} {cs_dem:<18.4f} {tcr_dem:<18.4f}")
    print(f"{'  95% CI':<35} {f'[{cs_ci[0]:.3f},{cs_ci[1]:.3f}]':<18} {f'[{tcr_ci[0]:.3f},{tcr_ci[1]:.3f}]':<18}")
    print(f"{'Death C_td':<35} {cs_dth:<18.4f} {tcr_dth:<18.4f}")
    print(f"{'Overall (any-event) C_td':<35} {cs_any:<18.4f} {'(N/A)':<18}")
    print(f"{'Dementia IBS':<35} {ibs_dem:<18.4f}")
    print(f"{'Dementia INBLL':<35} {inbll_dem:<18.4f}")
    print(f"{'Death IBS':<35} {ibs_dth:<18.4f}")
    print(f"{'Death INBLL':<35} {inbll_dth:<18.4f}")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    main()
