#!/usr/bin/env python
"""
eval_dementia_pretrain.py
=========================
Evaluate pretrained SurvivEHR model's dementia prediction quality
using the pi network ONLY (no ODE computation, minimal GPU memory).

Mathematical basis:
  The model's CIF is: F^(k)(t|T_c) = pi^(k)(T_c) * ODE^(k)(t|T_c)
  pi^(k) represents the model's prior probability that event k is the next event.
  It is computed by: Linear(384,32) -> ReLU -> Linear(32,108117) -> Softmax
  This requires ONE matrix multiply, zero ODE expansion, ~0 extra GPU memory.

  The full IEC risk score is: R_k = sum_t F^(k)(t) = pi^(k) * sum_t ODE^(k)(t)
  Since ODE shapes are similar across events, ranking by pi^(k) closely
  approximates ranking by full R_k.

What this script measures:
  1. RANK: When dementia actually occurs, where does the model rank it? (out of 108K)
  2. DISCRIMINATION: Do pre-dementia transitions get higher pi(dementia) than others?
  3. TRANSITION PATTERNS: Which prior events trigger high dementia predictions?
  4. TOP-K ACCURACY: Is dementia in the model's top 10/100/1000 when it occurs?

Usage:
  1. Fill in CHECKPOINT_PATH and verify DEMENTIA_READ_CODES below
  2. Fill in the data module initialization in main() -- marked with TODO
  3. Run:  CUDA_VISIBLE_DEVICES=0 python eval_dementia_pretrain.py
"""

import os
import sys
import json
import torch
import numpy as np
from collections import defaultdict
from datetime import datetime

# ======================== USER CONFIG ========================
CHECKPOINT_PATH = "/Data0/swangek_data/991/CPRD/output/checkpoints/crPreTrain_small_1337.ckpt"

# Dementia Read codes from your database queries (image).
# Combined from Query 1 (Specific) and Query 2 (Fuzzy).
DEMENTIA_READ_CODES = [
    # Query 1 - Specific Read Codes
    "F110.", "Eu00.", "Eu01.", "Eu02z", "Eu002", "E00..",
    "Eu023", "Eu00z", "Eu025", "Eu01z", "E001.", "F1100",
    "Eu001", "E004.", "Eu000", "Eu02.", "Eu013", "E000.",
    "Eu01y", "E001z", "F1101", "Eu020", "E004z", "E0021",
    "Eu02y", "Eu012", "Eu011", "E00z.", "E0040", "E003.",
    "E0020",
    # Query 2 - Fuzzy Search (additional codes not in Q1)
    "Eu057", "Eu04.", "Eu053", "Eu04z", "Eu0z.", "Eu060",
    "Eu054", "Eu05y", "Eu052", "Eu062",
]

# If you already know the token IDs, set them here directly.
# Otherwise leave as None to look up from Read codes via dm.encode().
DEMENTIA_TOKEN_IDS = None

DEVICE = "cuda:0"
BATCH_SIZE_OVERRIDE = 8   # use smaller batch for test to be safe
MAX_BATCHES = None         # None = full test set; set e.g. 200 for quick check
OUTPUT_PATH = "dementia_eval_results.json"
# =============================================================


def lookup_dementia_token_ids(dm, read_codes):
    """
    Convert Read codes to token IDs using the data module's tokenizer.
    Returns only valid tokens (not UNK=1, not PAD=0).
    """
    token_ids = []
    found_codes = []
    missing_codes = []

    for code in read_codes:
        try:
            tid = dm.encode([code])[0]
            if tid > 1:  # 0=PAD, 1=UNK
                token_ids.append(tid)
                found_codes.append((code, tid))
            else:
                missing_codes.append((code, tid))
        except Exception as e:
            missing_codes.append((code, str(e)))

    token_ids = sorted(set(token_ids))

    print(f"\n--- Dementia Token Lookup ---")
    print(f"Read codes provided:   {len(read_codes)}")
    print(f"Valid tokens found:    {len(token_ids)}")
    print(f"Missing/UNK codes:     {len(missing_codes)}")
    if missing_codes:
        for code, reason in missing_codes[:10]:
            print(f"  {code} -> {reason}")
    print(f"\nDementia token IDs: {token_ids}")
    print()

    return token_ids, found_codes, missing_codes


@torch.no_grad()
def evaluate_dementia(model, dataloader, dementia_token_ids, device,
                      max_batches=None, vocab_size=None):
    """
    Core evaluation: compute pi for every valid transition in the test set,
    then analyze how well the model predicts dementia.
    """
    model.eval()
    model.to(device)

    # pi output has shape (M, vocab_size-1) where index i = token i+1
    # (PAD=0 is excluded from the survival model)
    dementia_pi_indices = torch.tensor(
        [tid - 1 for tid in dementia_token_ids], device=device
    )
    dementia_token_set = set(dementia_token_ids)

    if vocab_size is None:
        # infer from pinet output layer
        vocab_size = model.surv_layer.sr_ode.pinet.mapping[-2].out_features + 1

    n_events = vocab_size - 1  # excluding PAD
    print(f"Vocab size: {vocab_size}, Event types: {n_events}")
    print(f"Dementia tokens: {len(dementia_token_ids)}")
    print(f"Dementia pi indices: {dementia_pi_indices.tolist()[:5]}... (showing first 5)")

    # ---- Accumulators ----
    dementia_ranks = []
    dementia_pi_values = []
    dementia_concordances = []

    all_pi_dementia_sum = []     # pi(all dementia tokens summed) per transition
    all_is_dementia = []         # bool: was true next event a dementia code?

    # prior_event -> list of pi(dementia_sum) values
    prior_event_pi = defaultdict(list)

    # per-dementia-code statistics
    per_code_ranks = defaultdict(list)

    total_transitions = 0
    dementia_transitions = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        tokens = batch['tokens'].to(device)
        ages = batch['ages'].to(device)
        values = batch['values'].to(device)
        covariates = batch['static_covariates'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        bsz, seq_len = tokens.shape

        # 1. Transformer forward
        hidden_states = model.transformer(
            tokens=tokens, ages=ages, values=values,
            covariates=covariates, attention_mask=attention_mask
        )

        # 2. Prepare inputs (replicating competing_risk.py predict() logic)
        #    hidden_states[:, :-1, :] predicts tokens[:, 1:]
        n_embd = model.n_embd
        n_private = model.n_embd_private
        h = hidden_states[:, :-1, :n_embd - n_private]     # (bsz, seq_len-1, hidden_dim)

        target_k = tokens[:, 1:]                             # (bsz, seq_len-1)
        prior_k = tokens[:, :-1]                             # (bsz, seq_len-1)
        obs_mask = attention_mask[:, :-1] & attention_mask[:, 1:]

        # Flatten
        h_flat = h.reshape(-1, h.shape[-1])
        target_flat = target_k.reshape(-1)
        prior_flat = prior_k.reshape(-1)
        mask_flat = obs_mask.reshape(-1)

        # Apply observation mask
        valid_idx = (mask_flat == 1)
        h_valid = h_flat[valid_idx]
        target_valid = target_flat[valid_idx]
        prior_valid = prior_flat[valid_idx]

        n_valid = h_valid.shape[0]
        if n_valid == 0:
            continue

        total_transitions += n_valid

        # 3. Compute pi (the key computation - one matmul, tiny memory)
        pi = model.surv_layer.sr_ode.pinet(h_valid.float())  # (n_valid, n_events)

        # 4. Sum of pi mass on all dementia tokens for each transition
        pi_dem_sum = pi[:, dementia_pi_indices].sum(dim=1)    # (n_valid,)

        target_np = target_valid.cpu().numpy()
        prior_np = prior_valid.cpu().numpy()
        pi_dem_sum_np = pi_dem_sum.cpu().numpy()

        # Store bulk statistics
        all_pi_dementia_sum.extend(pi_dem_sum_np.tolist())

        for i in range(n_valid):
            true_tok = int(target_np[i])
            prior_tok = int(prior_np[i])
            is_dem = true_tok in dementia_token_set

            all_is_dementia.append(is_dem)
            prior_event_pi[prior_tok].append(float(pi_dem_sum_np[i]))

            if is_dem:
                dementia_transitions += 1

                # Rank of the TRUE dementia token among all events
                pi_row = pi[i].cpu().numpy()          # (n_events,)
                true_pi_idx = true_tok - 1            # pi index for this token
                true_pi_val = pi_row[true_pi_idx]

                # 1-based rank (how many events have pi >= this one)
                rank = int(np.sum(pi_row >= true_pi_val))
                concordance = 1.0 - (rank - 1) / (n_events - 1)

                dementia_ranks.append(rank)
                dementia_pi_values.append(float(true_pi_val))
                dementia_concordances.append(concordance)
                per_code_ranks[true_tok].append(rank)

        if (batch_idx + 1) % 50 == 0:
            print(f"  [{batch_idx+1} batches] "
                  f"{total_transitions:,} transitions, "
                  f"{dementia_transitions} dementia events")

    # ---- Aggregate results ----
    print(f"\nDone. {total_transitions:,} total transitions, "
          f"{dementia_transitions} dementia events.")

    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": CHECKPOINT_PATH,
        "n_dementia_tokens": len(dementia_token_ids),
        "dementia_token_ids": dementia_token_ids,
        "total_transitions": total_transitions,
        "dementia_transitions": dementia_transitions,
        "n_events": n_events,
    }

    if dementia_transitions > 0:
        ranks = np.array(dementia_ranks)
        results["ranking"] = {
            "avg_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)),
            "std_rank": float(np.std(ranks)),
            "min_rank": int(np.min(ranks)),
            "max_rank": int(np.max(ranks)),
            "avg_concordance": float(np.mean(dementia_concordances)),
            "top_10": float(np.mean(ranks <= 10) * 100),
            "top_50": float(np.mean(ranks <= 50) * 100),
            "top_100": float(np.mean(ranks <= 100) * 100),
            "top_500": float(np.mean(ranks <= 500) * 100),
            "top_1000": float(np.mean(ranks <= 1000) * 100),
            "avg_pi_value": float(np.mean(dementia_pi_values)),
        }

        # Per-code breakdown
        per_code_summary = {}
        for tok, rk_list in per_code_ranks.items():
            rk = np.array(rk_list)
            per_code_summary[int(tok)] = {
                "count": len(rk_list),
                "avg_rank": float(np.mean(rk)),
                "median_rank": float(np.median(rk)),
            }
        results["per_code"] = per_code_summary

    # Discrimination: pi(dementia) when dementia occurs vs when it doesn't
    all_pi = np.array(all_pi_dementia_sum)
    all_is = np.array(all_is_dementia)
    pi_when_true = all_pi[all_is] if np.any(all_is) else np.array([0.0])
    pi_when_false = all_pi[~all_is] if np.any(~all_is) else np.array([1e-10])

    results["discrimination"] = {
        "avg_pi_dementia_when_dementia_occurs": float(np.mean(pi_when_true)),
        "avg_pi_dementia_when_other_occurs": float(np.mean(pi_when_false)),
        "median_pi_when_true": float(np.median(pi_when_true)),
        "median_pi_when_false": float(np.median(pi_when_false)),
        "ratio": float(np.mean(pi_when_true) / max(np.mean(pi_when_false), 1e-10)),
    }

    # Top prior events that trigger highest dementia predictions
    prior_avg = {}
    for tok, vals in prior_event_pi.items():
        if len(vals) >= 5:  # require at least 5 observations
            prior_avg[int(tok)] = {
                "avg_pi_dementia": float(np.mean(vals)),
                "count": len(vals),
            }
    sorted_priors = sorted(prior_avg.items(), key=lambda x: x[1]["avg_pi_dementia"], reverse=True)
    results["top_prior_events"] = sorted_priors[:50]

    return results


def print_results(results, dm=None):
    """Pretty-print evaluation results to console."""

    print("\n" + "=" * 65)
    print("  DEMENTIA PREDICTION EVALUATION (Pi Network, Pretrain Phase)")
    print("=" * 65)

    n_events = results["n_events"]
    print(f"\nDataset:   {results['total_transitions']:,} valid transitions")
    print(f"Dementia:  {results['dementia_transitions']} dementia events "
          f"({results['n_dementia_tokens']} unique dementia token types)")

    if results["dementia_transitions"] == 0:
        print("\n  *** No dementia transitions found in test set! ***")
        print("  Check that your dementia token IDs are correct.")
        print("=" * 65)
        return

    # ---- Ranking ----
    r = results["ranking"]
    print(f"\n--- Ranking (out of {n_events:,} event types) ---")
    print(f"  Average rank:   {r['avg_rank']:.1f}  (random baseline = {n_events/2:.0f})")
    print(f"  Median rank:    {r['median_rank']:.1f}")
    print(f"  Best rank:      {r['min_rank']}")
    print(f"  Worst rank:     {r['max_rank']}")
    print(f"  Avg concordance (IEC): {r['avg_concordance']:.4f}")
    print(f"  Avg pi value:   {r['avg_pi_value']:.2e}")
    print(f"\n  Top-K accuracy (% of dementia events ranked within top K):")
    print(f"    Top 10:    {r['top_10']:5.1f}%")
    print(f"    Top 50:    {r['top_50']:5.1f}%")
    print(f"    Top 100:   {r['top_100']:5.1f}%")
    print(f"    Top 500:   {r['top_500']:5.1f}%")
    print(f"    Top 1000:  {r['top_1000']:5.1f}%")

    # ---- Interpretation ----
    avg_rank = r['avg_rank']
    print(f"\n  Interpretation:")
    print(f"    Random baseline: dementia would rank ~{n_events//2:,} on average.")
    if avg_rank < 100:
        print(f"    Your model ranks dementia at {avg_rank:.0f} on average -> VERY STRONG")
        print(f"    (top {avg_rank/n_events*100:.2f}% of all events)")
    elif avg_rank < 1000:
        print(f"    Your model ranks dementia at {avg_rank:.0f} on average -> STRONG")
        print(f"    (top {avg_rank/n_events*100:.2f}% of all events)")
    elif avg_rank < 5000:
        print(f"    Your model ranks dementia at {avg_rank:.0f} on average -> MODERATE")
        print(f"    (top {avg_rank/n_events*100:.2f}% of all events)")
    else:
        print(f"    Your model ranks dementia at {avg_rank:.0f} on average -> WEAK")
        print(f"    (top {avg_rank/n_events*100:.2f}% of all events)")

    # ---- Discrimination ----
    d = results["discrimination"]
    print(f"\n--- Risk Discrimination ---")
    print(f"  Avg pi(dementia) BEFORE dementia occurs:    {d['avg_pi_dementia_when_dementia_occurs']:.2e}")
    print(f"  Avg pi(dementia) BEFORE non-dementia:       {d['avg_pi_dementia_when_other_occurs']:.2e}")
    print(f"  Ratio (higher = better discrimination):     {d['ratio']:.1f}x")

    if d['ratio'] > 5:
        print(f"  -> STRONG discrimination: model assigns {d['ratio']:.0f}x more risk before dementia")
    elif d['ratio'] > 2:
        print(f"  -> MODERATE discrimination")
    elif d['ratio'] > 1:
        print(f"  -> WEAK discrimination")
    else:
        print(f"  -> NO discrimination (model does not distinguish pre-dementia)")

    # ---- Per-code breakdown ----
    if "per_code" in results and results["per_code"]:
        print(f"\n--- Per Dementia Code Breakdown ---")
        sorted_codes = sorted(results["per_code"].items(),
                              key=lambda x: x[1]["count"], reverse=True)
        for tok_str, info in sorted_codes[:15]:
            tok = int(tok_str)
            code_name = f"token_{tok}"
            if dm:
                try:
                    code_name = dm.decode([tok])
                    if isinstance(code_name, list):
                        code_name = code_name[0]
                except:
                    pass
            print(f"  {code_name:25s}  n={info['count']:4d}  "
                  f"avg_rank={info['avg_rank']:.0f}  "
                  f"median_rank={info['median_rank']:.0f}")

    # ---- Prior events ----
    print(f"\n--- Top 20 Prior Events Triggering Highest Dementia Predictions ---")
    for tok, info in results["top_prior_events"][:20]:
        tok = int(tok)
        name = f"token_{tok}"
        if dm:
            try:
                name = dm.decode([tok])
                if isinstance(name, list):
                    name = name[0]
            except:
                pass
        print(f"  {name:40s}  pi(dem)={info['avg_pi_dementia']:.2e}  "
              f"(n={info['count']})")

    print("\n" + "=" * 65)


# ================================================================
#  MAIN
# ================================================================
if __name__ == "__main__":

    import hydra
    from omegaconf import DictConfig, OmegaConf
    from FastEHR.dataloader.foundational_loader import FoundationalDataModule
    from SurvivEHR.examples.modelling.SurvivEHR.setup_causal_experiment import CausalExperiment

    print("=" * 65)
    print("  SurvivEHR Dementia Pretrain Evaluation")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Step 0: Load config and initialize data module
    #         (same logic as run_experiment.py)
    # ------------------------------------------------------------------
    # NOTE: Hydra's @hydra.main decorator changes working directory.
    #       We use hydra.initialize + hydra.compose instead so the
    #       script can run standalone without Hydra's CLI machinery.

    # >>>>>>>>>> PATHS - VERIFY THESE MATCH YOUR SETUP <<<<<<<<<<
    HYDRA_CONFIG_PATH = "/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR/confs"
    HYDRA_CONFIG_NAME = "config_CompetingRisk11M"
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    with hydra.initialize_config_dir(config_dir=HYDRA_CONFIG_PATH, version_base=None):
        cfg = hydra.compose(config_name=HYDRA_CONFIG_NAME)

    # Override: we only need test, not train
    cfg.experiment.train = False
    cfg.experiment.test = True

    torch.manual_seed(cfg.experiment.seed)
    torch.set_float32_matmul_precision('medium')

    print(f"\nInitializing data module...")
    print(f"  path_to_db: {cfg.data.path_to_db}")
    print(f"  path_to_ds: {cfg.data.path_to_ds}")

    dm = FoundationalDataModule(
        path_to_db=cfg.data.path_to_db,
        path_to_ds=cfg.data.path_to_ds,
        load=True,
        tokenizer="tabular",
        batch_size=BATCH_SIZE_OVERRIDE,      # use smaller batch for eval
        max_seq_length=cfg.transformer.block_size,
        global_diagnoses=cfg.data.global_diagnoses,
        repeating_events=cfg.data.repeating_events,
        freq_threshold=cfg.data.unk_freq_threshold,
        min_workers=cfg.data.min_workers,
        overwrite_meta_information=cfg.data.meta_information_path,
        supervised=False,                    # pretrain mode = causal
        subsample_training=cfg.data.subsample_training,
        seed=cfg.experiment.seed,
    )

    # Populate cfg fields that run_experiment.py fills from dm
    vocab_size = dm.train_set.tokenizer.vocab_size
    measurements = dm.train_set.meta_information["measurement_tables"][
        dm.train_set.meta_information["measurement_tables"]["count_obs"] > 0
    ]["event"].to_list()
    cfg.head.tokens_for_univariate_regression = dm.encode(measurements)
    cfg.data.num_static_covariates = next(iter(dm.test_dataloader()))['static_covariates'].shape[1]

    print(f"  vocab_size: {vocab_size}")
    print(f"  num_static_covariates: {cfg.data.num_static_covariates}")
    print(f"  test set batches: {len(dm.test_dataloader())}")

    # ------------------------------------------------------------------
    # Step 1: Get dementia token IDs
    # ------------------------------------------------------------------
    if DEMENTIA_TOKEN_IDS is not None:
        dementia_token_ids = DEMENTIA_TOKEN_IDS
        print(f"\nUsing provided dementia token IDs: {len(dementia_token_ids)} tokens")
    else:
        dementia_token_ids, found, missing = lookup_dementia_token_ids(dm, DEMENTIA_READ_CODES)
        if len(dementia_token_ids) == 0:
            print("ERROR: No valid dementia tokens found! Check your Read codes.")
            print("Possible causes:")
            print("  1. These Read codes don't exist in your UK Biobank dataset")
            print("  2. They were below the frequency threshold and mapped to UNK")
            print("  3. Your dataset uses different code format (SNOMED vs Read V2)")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Load model from checkpoint
    # ------------------------------------------------------------------
    print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
    if not os.path.isfile(CHECKPOINT_PATH):
        # Try to find best checkpoint automatically
        ckpt_dir = cfg.experiment.ckpt_dir
        print(f"  Checkpoint not found. Searching in {ckpt_dir} ...")
        candidates = sorted(
            [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')],
            key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)),
            reverse=True
        )
        if candidates:
            print(f"  Available checkpoints:")
            for c in candidates[:10]:
                sz = os.path.getsize(os.path.join(ckpt_dir, c)) / 1e6
                print(f"    {c}  ({sz:.0f} MB)")
            print(f"\n  Please set CHECKPOINT_PATH to one of the above.")
        sys.exit(1)

    experiment = CausalExperiment.load_from_checkpoint(
        CHECKPOINT_PATH,
        cfg=cfg,
    )
    model = experiment.model
    model.eval()
    model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded. Parameters: {n_params:,}")
    print(f"  Survival head: {type(model.surv_layer).__name__}")
    print(f"  Pi network output dim: {model.surv_layer.sr_ode.pinet.mapping[-2].out_features}")

    # ------------------------------------------------------------------
    # Step 3: Run evaluation
    # ------------------------------------------------------------------
    print(f"\nStarting evaluation on test set...")
    print(f"  Device: {DEVICE}")
    print(f"  Batch size: {BATCH_SIZE_OVERRIDE}")
    print(f"  Max batches: {MAX_BATCHES or 'ALL'}")

    test_loader = dm.test_dataloader()

    results = evaluate_dementia(
        model=model,
        dataloader=test_loader,
        dementia_token_ids=dementia_token_ids,
        device=DEVICE,
        max_batches=MAX_BATCHES,
        vocab_size=vocab_size,
    )

    # ------------------------------------------------------------------
    # Step 4: Print and save results
    # ------------------------------------------------------------------
    print_results(results, dm=dm)

    # Save to JSON
    results_serializable = json.loads(json.dumps(results, default=str))
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_PATH}")
