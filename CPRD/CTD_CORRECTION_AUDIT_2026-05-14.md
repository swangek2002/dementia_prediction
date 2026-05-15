# C_td Methodology Correction — Full Audit Trail (2026-05-14)

This document records the entire chain of discovery from the per-batch C_td issue to the final cohort-level result table and new comparison strategy. Preserved chronologically for future reference and reproducibility.

## Timeline

### 2026-05-14 morning: First inconsistency noticed

V5 had just finished training (per-batch C_td = 0.7810). While preparing literature comparison, an independent pycox `EvalSurv.concordance_td('antolini')` recompute on the same V5 inference output gave:
- Full test (8257): 0.8496
- Cleaned (8241, after 16 leaky removed): 0.8467

Gap to per-batch reported = +0.07. Too large for numerical precision.

Initial hypothesis (wrong): "cause-specific inflation" — competing risk model treating death as censoring inflates C-index.

### 2026-05-14 midday: Root cause traced

User pushed back on cause-specific hypothesis. Recommended verifying from first principles. Reading `clinical_prediction_model.py:213-229`:

```python
def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    self.run_callback(_trainer=trainer, _pl_module=pl_module, batch=batch, ...)

def get_metrics(self, cdf, lbls, target_ages, _trainer, _pl_module, log_name):
    surv = pd.DataFrame((1 - cdf).T, index=t_eval)
    ev = EvalSurv(surv, target_ages, lbls, censor_surv='km')
    ctd = ev.concordance_td()  # computed on current BATCH (16 patients)
    self.log_dict({log_name+"ctd": ctd})
```

In PyTorch Lightning, `self.log()` in `on_test_batch_end` defaults to `on_step=False, on_epoch=True, reduce_fx="mean"`. So Lightning aggregates per-batch C_td values via mean across all 517 test batches.

**This is NOT the standard Antolini's C_td** (which uses ALL ranking pairs across the entire cohort).

### 2026-05-14 afternoon: User skepticism — trend should NOT reverse

User observation (correct intuition):
> "怎么可能说把 C_td 的计算方式一改，整个 C_td 在 self-training 中的变化方式都变了啊...trend 理论上不会变啊"

Per-batch trend: V5 (0.7810) > V4 (0.7732) > V3 (0.7685) — "self-training keeps improving"
Cohort trend: V3 (0.8506) > V4 (0.8487) > V5 (0.8467) — "1st-round SST is the only round that helps"

User's intuition assumed both computations are valid estimators. My response: per-batch averaging is **not** a valid estimator of C_td — it's biased by:
1. Selection bias (batches with 0 events excluded)
2. Small-sample noise
3. Cross-batch pair loss
4. Order sensitivity

Therefore the trend reversal is consistent with per-batch being biased, not necessarily implementation bug.

### 2026-05-14 evening: Independent verification

User pushed: "你真的确定是计算方法没出错吗" — demanding cross-checks.

Three independent verifications performed:

#### Check 1: Same NPZ structure across all models
- V2, V3, V4, V5 NPZs: identical PIDs, identical labels (V2 corrections), identical event times
- Only CIF predictions differ
- → Trend reversal is not from data preprocessing differences

#### Check 2: Manual Antolini's C_td from numpy (independent of pycox)
| Model | pycox cohort | numpy manual | Δ |
|-------|:------------:|:------------:|:--:|
| V3 | 0.8506 | 0.8507 | 0.0001 |
| V4 | 0.8487 | 0.8488 | 0.0001 |
| V5 | 0.8467 | 0.8468 | 0.0001 |
| V2 | 0.8447 | 0.8447 | 0.0000 |

→ Two independent implementations agree to 4 decimals. Pycox is not buggy. Cohort C_td 0.8506 is real.

#### Check 3: Wolbers true CR C_td (additional sanity check, custom implementation, 6 synthetic test cases passed)
- V3 cohort: cause-specific 0.8506 vs true CR 0.8506 (identical to 4 decimals)
- V5 cohort: cause-specific 0.8467 vs true CR 0.8468

→ Cause-specific and Wolbers give essentially identical C-index. The "inflation" concern applies to hazard/calibration, not C-index.

### 2026-05-14 evening: Batch-size sweep to verify mechanism

User: "请你跑一个真正的两种方法的对比...让我亲自亲眼看到你说的那些原因是否真实发生了"

Implemented `batch_size_sweep_v3.py` — sweeps batch_size from 16 to 8241 for V3 cohort:

| batch_size | n_batches | %0-dem | avg C_td | std | range |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 16 | 516 | 47.3% | 0.7549 | 0.289 | [0.000, 1.000] |
| 32 | 258 | 25.2% | 0.7637 | 0.248 | [0.000, 1.000] |
| 64 | 129 | 4.7% | 0.7788 | 0.206 | [0.000, 1.000] |
| 128 | 65 | 1.5% | 0.7930 | 0.149 | [0.337, 0.991] |
| 256 | 33 | 0.0% | 0.8165 | 0.090 | [0.613, 0.989] |
| 512 | 17 | 0.0% | 0.8184 | 0.069 | [0.670, 0.978] |
| 1024 | 9 | 0.0% | 0.8334 | 0.064 | [0.749, 0.978] |
| 2048 | 5 | 0.0% | 0.8557 | 0.069 | [0.777, 0.978] |
| 8241 (cohort) | 1 | 0% | 0.8506 | 0 | exact |

All 4 failure mechanisms empirically verified:
1. **Selection bias**: 47.3% of bs=16 batches have 0 dementia → silently dropped
2. **Small-sample noise**: bs=16 std=0.29, range [0, 1] → essentially random
3. **Order sensitivity**: bs=16 sequential = 0.7549 vs shuffled (5 seeds) = 0.8339, Δ = +0.078 → cross-batch pair loss
4. **Smooth convergence**: bs=16 → bs=8241 monotonic increase, asymptotes near cohort

User's "fewer batches middle approach" hypothesis confirmed:
- bs=256: 0.8165 (close to cohort 0.8506)
- bs=1024: 0.8334
- bs=2048: 0.8557 (slightly over due to small n_batches=5)
- bs=8241 cohort: 0.8506 (exact)

### 2026-05-14 late evening: Complete cohort eval for all 7 post-leakage models

Built generic pipeline:
- `inference_dual_cohort_ctd.py` — dual model inference
- `inference_single_v2ablation.py` — single backbone inference
- `compute_cohort_ctd_generic.py` — generic cohort metrics
- `compute_v5_cohort_ctd.py` — V5 with bootstrap + verification

Ran chain:
1. V5 (NPZ already existed from previous agent): 0.8467
2. V4 inference + metrics: 0.8487
3. V3 inference + metrics: 0.8506 (peak)
4. V2 labels inference + metrics: 0.8447
5. Dual baseline (V1 labels) inference + V2-labels re-eval: 0.8416
6. Cross-attention (V1 labels) inference + V2-labels re-eval: 0.8428
7. V2 ablation (single backbone) inference + metrics: 0.8451

Final ranking (cohort dementia C_td):
1. V3 (1st SST): **0.8506** ⭐
2. V4 (2nd SST): 0.8487
3. V5 (3rd SST): 0.8467
4. V2 ablation: 0.8451
5. V2 labels: 0.8447
6. Cross-attention: 0.8428
7. Dual baseline: 0.8416

### 2026-05-14 night: SurvivEHR paper read for comparison framing

User suggested looking at SurvivEHR paper (our backbone) for direct comparison since we share backbone.

Found Table 2:
- SurvivEHR FFT Hypertension single-risk: 0.824 ± 0.002
- SurvivEHR FFT CVD competing-risk: 0.667 ± 0.005
- SurvivEHR FFT Multi-morbidity: 0.663 ± 0.002

Their Table 2 uses same pycox EvalSurv pipeline (we forked their codebase). Their numbers are per-batch averaged from same Lightning callback we used. But per-batch vs cohort gap is smaller for them because hypertension/CVD/multimorbidity have higher event prevalence than our 4.55% dementia.

### 2026-05-14 night: New comparison strategy decided

User asked: "我们应该不会直接和 survivehr 这篇 paper 直接进行对比，而是我们要学习这篇 paper，看看人家是如何对比其他工作来凸显自己工作有多厉害的"

Adopted SurvivEHR's comparison pattern:
1. All baselines in-house on same cohort (LR, Cox PH, RSF, DeSurv head, DeepHit head)
2. Methodology gradient (classical → DL head only → transformer + pretrain)
3. Internal ablation isolating each component's value
4. Multi-task/multi-index-age generalization (optional)

Decision: abandon all cross-paper number comparison (Yuan / DemRisk / Anatürk / Zhang). Build methodology gradient on our cohort.

### V6 Plan (Next headline model)

| Component | Spec |
|-----------|------|
| Backbone | Single GP transformer (no HES backbone, V2 ablation showed dual ≈ 0) |
| Static | 49-d (27 demographic + 22 HES) |
| Labels | V2 corrections (1397 relabel, 487 prevalent removed) |
| New: GP-prevalent | Extend prevalent filter to GP codes (+246 patients removed) |
| Self-training | 1 round only, top 1% (V3-style; later rounds trade C_td for IBS) |
| New: Pseudo time | Sample from V2-type-A empirical lag distribution (~2.5y median) |
| Eval | Cohort-level C_td (cause-spec + true CR) + IBS + INBLL + bootstrap CI + calibration slope |

Expected: cohort C_td 0.85-0.86, IBS 0.30-0.32, @2y calibration slope 0.5-0.7 (vs V4's catastrophic 0.27).

---

## Key Files Created This Day

| File | Purpose |
|------|---------|
| `compute_v5_cohort_ctd.py` | V5-specific cohort metrics + bootstrap + true CR verification on 6 synthetic cases |
| `compute_cohort_ctd_generic.py` | Generic cohort metrics for any NPZ |
| `inference_dual_cohort_ctd.py` | Full dual-model inference saving NPZ |
| `inference_single_v2ablation.py` | Full single-backbone inference (V2 ablation) |
| `run_all_post_leakage_inference.sh` | Master orchestrator for all 6 models |
| `batch_size_sweep_v3.py` | Per-batch failure mechanism empirical verification |
| `COHORT_CTD_RESULTS.md` | Final results report for all 7 models |
| `SURVIVEHR_COMPARISON.md` | Direct SurvivEHR paper comparison + framing recommendations |
| `CTD_CORRECTION_AUDIT_2026-05-14.md` | This file — chronological audit trail |

NPZs:
- `data/test_cif_v5_full.npz` (8257, V2 labels)
- `data/test_cif_v4_full.npz` (8257, V2 labels)
- `data/test_cif_v3_full.npz` (8257, V2 labels)
- `data/test_cif_v2_full.npz` (8257, V2 labels)
- `data/test_cif_dual_baseline_full.npz` (8292, V1 labels — needs V2 override for fair comparison)
- `data/test_cif_crossattn_full.npz` (8292, V1 labels — same)
- `data/test_cif_v2_ablation_full.npz` (8257, V2 labels)

Saved batch sweep:
- `data/leaky_patients_test.txt` (16 PIDs)
- `data/leaky_patients_train.txt` (213 PIDs)
- `data/leaky_patients_val.txt` (17 PIDs)
- `batch_sweep_v3.json`

---

## What Has Been Corrected in Project Documentation

### PROJECT_KNOWLEDGE.md
- Section 1 Key Findings: Replaced "V5 is current best 0.7810" with "V3 is cohort peak 0.8506"
- Section 1 Literature: Replaced cross-paper comparison with new in-house strategy
- Section 6.7 V3: Added correction note — actually cohort peak
- Section 6.8 V4: Added correction note — discrimination-calibration trade-off, not improvement
- Section 6.10 V5: Added correction note — IBS peak, not C_td peak
- New Section 14: C_td Methodology Correction (full audit, mechanism, protocol)
- New Section 15: Cohort-level results for all 7 models
- New Section 16: New comparison strategy (in-house baselines)
- New Section 17: V6 experiment plan

### PROGRESS_REPORT.md (Chinese)
- Section 0 TL;DR: Replaced per-batch trend table with dual per-batch + cohort table
- Section 0: Added 2026-05-14 修正 subsection with cohort findings
- Three contributions paragraph: Rewrote with cohort-level attribution
- New Section 15: C_td methodology correction (Chinese)
- New Section 16: New comparison strategy (Chinese)
- New Section 17: V6 plan (Chinese)

### Things deliberately NOT deleted (preserved for audit history)
- All Section 6 results tables retain per-batch values (clearly labeled)
- Old comparison attempts (Yuan 2024, DemRisk) preserved with strikethrough
- Old "V5 is best" claims preserved with correction notes

This satisfies the project protocol (Section 13 of PROJECT_KNOWLEDGE.md): "Don't delete prior claims; mark with strikethrough + correction notes".
