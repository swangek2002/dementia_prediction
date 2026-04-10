"""
generate_cv5_configs.py
========================
Generate 5 pairs of train/eval YAML configs for 5-fold CV.
Index=68, lambda=6, 10 epochs max, early_stop_patience=5.

Usage:
    python generate_cv5_configs.py

Output:
    confs/config_FineTune_Dementia_CR_idx72_cv_fold{0..4}.yaml
    confs/config_FineTune_Dementia_CR_idx72_cv_fold{0..4}_eval.yaml
"""

from pathlib import Path

CONFS_DIR = "/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR/confs"

DEMENTIA_CODES_YAML = """    - - "F110."
      - "Eu00."
      - "Eu01."
      - "Eu02z"
      - "Eu002"
      - "E00.."
      - "Eu023"
      - "Eu00z"
      - "Eu025"
      - "Eu01z"
      - "E001."
      - "F1100"
      - "Eu001"
      - "E004."
      - "Eu000"
      - "Eu02."
      - "Eu013"
      - "E000."
      - "Eu01y"
      - "E001z"
      - "F1101"
      - "Eu020"
      - "E004z"
      - "E0021"
      - "Eu02y"
      - "Eu012"
      - "Eu011"
      - "E00z."
      - "E0040"
      - "E003."
      - "E0020"
    - - "DEATH"
"""


def make_config(fold: int, is_eval: bool) -> str:
    ds_path = f"/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_idx72_cv/fold{fold}/"
    fine_tune_id = f"FineTune_Dementia_CR_idx72_cv_fold{fold}"
    ckpt_name = f"crPreTrain_small_1337_{fine_tune_id}.ckpt"

    train_flag = "False" if is_eval else "True"
    mode_label = "eval" if is_eval else "train"
    notes = f"{'Single-GPU eval' if is_eval else 'Train'} — idx72 5-fold CV fold {fold}, event_only lambda=6"

    # supervised_time_scale: study ends 2019, birth ~1937 means index at 68 -> 2005.
    # Max follow-up ~ 14 years. Scale 3.0 is appropriate.
    config = f"""is_decoder: True

data:
  batch_size: 32
  unk_freq_threshold: 0.0
  min_workers: 12
  global_diagnoses: True
  repeating_events: False
  path_to_db: /Data0/swangek_data/991/CPRD/data/example_exercise_database.db
  path_to_ds: {ds_path}
  meta_information_path: /Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/meta_information_custom.pickle
  subsample_training: null
  num_static_covariates: 27
  supervised_time_scale: 3.0

experiment:
  type: 'fine-tune'
  project_name: SurvivEHR
  run_id: ${{head.SurvLayer}}PreTrain_small_${{experiment.seed}}
  fine_tune_id: {fine_tune_id}
  notes: "{notes}"
  tags: ["dementia", "fine-tune", "competing-risk", "idx72", "cv5", "fold{fold}", "{mode_label}"]
  train: {train_flag}
  test: True
  verbose: True
  seed: 1337
  log: True
  log_dir: /Data0/swangek_data/991/CPRD/output/
  ckpt_dir: /Data0/swangek_data/991/CPRD/output/checkpoints/

fine_tuning:
  fine_tune_outcomes:
{DEMENTIA_CODES_YAML}
  custom_outcome_method:
    _target_: null
  custom_stratification_method:
    _target_: null

  use_callbacks:
    hidden_embedding:
      num_batches: 0
      mask_static: False
      mask_value: False
    performance_metrics: True
    rmst: False

  compression_layer: False
  llrd: null

  PEFT:
    method: null
    adapter_dim: 8
  backbone:
    linear_probe_epochs: 0
    unfreeze_top_k: null
  head:
    surv_weight: 1
    value_weight: 0
    learning_rate: 5e-4

  sample_weighting:
    mode: "event_only"
    event_lambda: 6.0
    alpha: 2.0
    tau: 0.33
    w_t_max: 3.0
    w_total_max: 20.0

optim:
  num_epochs: 10
  learning_rate: 5e-5
  scheduler_warmup: False
  scheduler: reduceonplateau
  scheduler_periods: 10000
  learning_rate_decay: 0.8
  val_check_interval: 1.0
  early_stop: True
  early_stop_patience: 5
  log_every_n_steps: 20
  limit_val_batches: 1.0
  limit_test_batches: null
  accumulate_grad_batches: 4

transformer:
  block_type: "Neo"
  block_size: 512
  n_layer: 6
  n_head: 6
  n_embd: 384
  layer_norm_bias: False
  attention_type: "global"
  bias: True
  dropout: 0.1
  attention_dropout: 0.1
  resid_dropout: 0.1
  private_heads: 0

head:
  SurvLayer: "cr"
  surv_weight: 1
  tokens_for_univariate_regression: None
  value_weight: 0.1
"""
    return config


def main():
    confs_dir = Path(CONFS_DIR)
    confs_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(5):
        # Train config
        train_cfg = make_config(fold, is_eval=False)
        train_path = confs_dir / f"config_FineTune_Dementia_CR_idx72_cv_fold{fold}.yaml"
        train_path.write_text(train_cfg)
        print(f"Written: {train_path}")

        # Eval config
        eval_cfg = make_config(fold, is_eval=True)
        eval_path = confs_dir / f"config_FineTune_Dementia_CR_idx72_cv_fold{fold}_eval.yaml"
        eval_path.write_text(eval_cfg)
        print(f"Written: {eval_path}")

    print("\nDone. 10 YAML configs generated.")


if __name__ == "__main__":
    main()
