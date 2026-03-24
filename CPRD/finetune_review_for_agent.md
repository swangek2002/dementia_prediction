# SurvivEHR Fine-tuning: Paper Methodology vs Current Implementation
# ================================================================
# This document compares what the original SurvivEHR paper does for
# fine-tuning with what has been implemented so far, identifies issues,
# and provides specific fixes.

## ============================
## 1. WHAT THE PAPER DOES
## ============================

Reference: Paper Section "Fine-tuning experiments" + Supplementary Table S7

### 1.1 Cohort Definition

The paper defines TWO fine-tuning cohorts:

**Cohort A: T2DM → Hypertension/CVD prediction**
- Population: patients diagnosed with Type 2 Diabetes
- Index date: date of T2DM diagnosis
- Training: 572,096 patients; Val: 35,758; Test: 33,280
- Prediction window: 5 years from index date
- Task 1 (single-risk): 5-year hypertension onset
- Task 2 (competing-risk): 5-year CVD (IHD + stroke as competing events)

**Cohort B: Multimorbidity prediction**
- Population: patients with pre-existing multimorbidity
- Index date: age 50
- Training: 20,000 patients (random sample)
- Prediction window: from age 50 onward

Key principle: For each patient, the model sees ONLY events BEFORE the
index date as input. The TARGET is what happens AFTER the index date.

### 1.2 Fine-tuning Architecture Changes

From the paper (Section "Fine-tuning experiments"):
- Context window INCREASED from 256 (pretrain) to 512
- PREPEND any previous diagnoses that may be lost from the context window
  (i.e., if a patient has >512 events, diagnoses from before the window
  are prepended so "all health conditions that could have lifelong
  implications are included")
- REMOVE repeated events (global_diagnoses=True, repeating_events=False)
- The model uses is_generation=True mode: only the LAST hidden state
  (representing patient state at index date) is forwarded through the
  survival head

### 1.3 Fine-tuning Hyperparameters (Supplementary Table S7)

| Parameter                    | Pretrain | Fine-tuning |
|------------------------------|----------|-------------|
| Max sequence length          | 256      | 512         |
| Context window               | Repeated | All Last unique |
| Global diagnoses             | False    | Append to context |
| Batch size                   | 64       | 512         |
| Epochs                       | 10       | 20          |
| Early stopping               | False    | True        |
| Optimizer                    | AdamW    | AdamW       |
| Backbone learning rate       | 3e-4     | 5e-5        |
| Head learning rate           | 3e-4     | 5e-4        |
| Scheduler                    | Cosine   | Reduce on plateau |
| Warmup steps                 | 10,000   | 0           |

### 1.4 Survival Head Replacement

During fine-tuning, the pretrained survival head (108K or 263 output dims)
is REMOVED and replaced with a NEW, randomly initialized head:
- For single-risk tasks: new DeSurv head with 1 outcome
- For competing-risk tasks: new DeSurv head with K outcomes (e.g., 2-3)

The Transformer backbone KEEPS the pretrained weights.

### 1.5 Three Model Variants for Comparison

The paper compares:
1. SurvivEHR-FFT: Pretrained backbone + fine-tuned new head (THIS IS THE MAIN MODEL)
2. SurvivEHR-SFT: Same architecture, trained from scratch (no pretrain)
3. SurvivEHR-ZS: Pretrained model, zero-shot (no fine-tuning)
4. Baselines: RSF, DeepHit, DeSurv (cross-sectional inputs)

### 1.6 Evaluation Metrics

- C_td (time-dependent concordance): measures risk ranking accuracy
- IBS (Integrated Brier Score): measures calibration
- INBLL (Integrated Negative Binomial Log-Likelihood): measures fit quality
- All reported as mean ± 95% CI over 5 random seeds


## ============================
## 2. WHAT IS CURRENTLY CORRECT
## ============================

### 2.1 Dataset construction (FIXED - now correct)
The indexed cohort dataset has been built with proper index dates.
199,698 train / 10,369 val / 12,524 test patients.
This is the right approach.

### 2.2 Loading pretrained checkpoint
Using crPreTrain_small_1337.ckpt as initialization and replacing the
survival head is correct. This is the FFT approach from the paper.

### 2.3 Sparsity-Aware Weighting addition
This is a NOVEL contribution not in the original paper. The implementation
in desurv.py (sample_weights parameter) and setup_finetune_experiment.py
(compute_sparsity_aware_weights function) is correctly designed. The
backward compatibility (sample_weights=None → original behavior) is good.

### 2.4 CDF clamp safety
Adding .clamp(0, 1-eps) to CDF values and .clamp(min=eps) to dudt values
is a reasonable safety measure for fine-tuning with longer time horizons.

### 2.5 scipy.integrate.simps → simpson
Correct API compatibility fix.


## ============================
## 3. WHAT NEEDS TO BE FIXED
## ============================

### 3.1 CRITICAL: experiment type should be "fine-tune" (competing-risk), NOT "fine-tune-sr"

Current config:
    experiment.type: 'fine-tune-sr'   # WRONG for dementia

The paper explicitly states that for the dementia task, death is the most
important competing risk. If death is treated as ordinary censoring
(single-risk), the model will systematically OVERESTIMATE dementia risk.

The paper's CVD task uses competing-risk. The hypertension task uses
single-risk only because hypertension has no strong competing risk.

For dementia, the correct setup is:
    experiment.type: 'fine-tune'      # This triggers competing-risk head

With TWO outcomes:
    - Dementia onset (primary event)
    - Death before dementia (competing risk)

This requires the fine_tune_outcomes config to include BOTH dementia codes
AND death codes, with the data pipeline correctly labeling:
    - k=1 for dementia events
    - k=2 for death events
    - k=0 for censored (neither dementia nor death in observation window)

If implementing competing-risk is too complex right now, single-risk is
acceptable as a FIRST experiment, but the thesis/paper MUST eventually
include the competing-risk version.


### 3.2 CRITICAL: Fine-tuning hyperparameters don't match paper

Current config vs paper's Table S7:

| Parameter              | Current    | Paper       | Fix needed? |
|------------------------|------------|-------------|-------------|
| Batch size             | 32         | 512         | YES - increase |
| Max sequence length    | 256        | 512         | YES - increase |
| Global diagnoses       | False      | True        | YES - change |
| Repeating events       | True       | False       | YES - change |
| Backbone LR            | 1e-4       | 5e-5        | YES - decrease |
| Head LR                | 5e-4       | 5e-4        | OK |
| Epochs                 | 30         | 20          | Minor |
| Scheduler              | cosine     | ReduceOnPlateau | YES - change |
| Warmup steps           | 10000      | 0           | YES - remove |

The most impactful ones are:

**batch_size=512**: The paper uses 512, not 32. With 4 GPUs you'd need
batch_size=128 per GPU, or batch_size=32 with accumulate_grad_batches=4.
Effective batch size matters for survival models because DeSurv loss
depends on the distribution of events within the batch.

**block_size=512**: The paper INCREASES context from 256 to 512 for
fine-tuning. This lets the model see more patient history before the
index date. Currently set to 256.

**global_diagnoses=True**: The paper says "prepend any previous diagnoses
which may otherwise be lost from the context window." This ensures all
chronic conditions are visible even for patients with very long histories.
Currently set to False.

**repeating_events=False**: The paper says "remove repeated events" for
fine-tuning. This means if a patient had 10 blood pressure measurements,
only the last one is kept. This reduces sequence length and focuses on
unique clinical events. Currently set to True.

**Backbone LR=5e-5**: The paper uses 5e-5 for the backbone (10x lower
than the head LR of 5e-4). This is standard practice for fine-tuning:
you want the pretrained backbone to change slowly while the new head
learns quickly. Currently set to 1e-4 (2x too high).

**No warmup**: The paper uses 0 warmup steps for fine-tuning (warmup is
only for pretrain). Currently warmup is still enabled.


### 3.3 IMPORTANT: supervised_time_scale needs verification

The agent added supervised_time_scale=3.0 to fix the time range issue.
This is a reasonable fix, but:

1. Need to verify what the PAPER uses. The paper scales time to [0,1]
   using t_eval = np.linspace(0, 1, 1000). The original pretrain data
   already uses time_scale=1825 (5 years = 1 unit). For fine-tuning with
   a 5-year prediction window, target_age_delta should naturally be in
   [0, 1] if the same time_scale is used. If target_age_delta reaches
   2.8, it means the observation window extends beyond 5*1825 = ~14 years
   from index date, which may not be intended.

2. A cleaner fix might be to CLAMP the prediction window to 5 years
   (or 10 years) at the data level, rather than rescaling. Patients
   observed beyond the window should be censored at the window boundary.


### 3.4 MINOR: Value head weight should be 0 for fine-tuning

The paper's Table S7 shows:
    fine_tuning.head.value_weight: 0

During fine-tuning, only the survival head is trained. The value
prediction head should be disabled. Check if the current config sets:
    head.value_weight: 0


## ============================
## 4. RECOMMENDED CONFIG
## ============================

Based on paper's Supplementary Table S7, the config should be:

```yaml
data:
  batch_size: 128            # 128 * 4 GPUs = 512 effective
  global_diagnoses: True     # Prepend diagnoses lost from context
  repeating_events: False    # Only keep last unique events

transformer:
  block_size: 512            # Increased from 256 for fine-tuning

experiment:
  type: 'fine-tune'          # Competing-risk (not fine-tune-sr)

fine_tuning:
  head:
    surv_weight: 1
    value_weight: 0          # Disable value head for fine-tuning
    learning_rate: 5e-4      # Head LR

optim:
  num_epochs: 20
  learning_rate: 5e-5        # Backbone LR (10x lower than head)
  scheduler_warmup: False    # No warmup for fine-tuning
  scheduler: ReduceOnPlateau # Not cosine
  early_stop: True
  early_stop_patience: 10
  accumulate_grad_batches: 1 # With batch_size=128 * 4 GPUs = 512
  val_check_interval: 1.0    # Validate every epoch
```


## ============================
## 5. EXECUTION ORDER
## ============================

1. Fix the config according to Section 4 above
2. Verify that the data pipeline correctly handles:
   - global_diagnoses=True (prepending lost diagnoses)
   - repeating_events=False (deduplication)
   - block_size=512 (longer context)
3. Run FFT (fine-tune from pretrained checkpoint)
4. Run SFT (fine-tune from scratch, for comparison)
5. Report C_td, IBS, INBLL on test set with 95% CI over multiple seeds
6. Compare FFT vs SFT to demonstrate pretrain value
