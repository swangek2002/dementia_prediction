"""
Standalone verification: evaluate the fully-clean v3 checkpoint with UNFILTERED
HES sequences to confirm the model does NOT exploit post-index information.

This script is self-contained and does NOT modify any existing code files.
It imports the necessary modules and overrides index_on_age at call time only.

Expected:
  - v3 + clean eval  = 0.757  (already obtained)
  - v3 + leaky eval  ~ similar (model can't exploit future info it never saw)
  - v3 + leaky eval >> 0.757  would indicate a problem
"""
import sys
import os
import pickle
import logging
from pathlib import Path

# Setup paths
sys.path.insert(0, "/Data0/swangek_data/991/FastEHR")
sys.path.insert(0, "/Data0/swangek_data/991/CPRD")
sys.path.insert(0, "/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR")

from dual_data_module import (
    build_hes_sequence_cache, HESTokenizer, DualCollateWrapper, load_yob_lookup
)

logging.basicConfig(level=logging.INFO)


def main():
    from hydra import compose, initialize_config_dir
    config_dir = "/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR/confs"

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config_FineTune_Dementia_CR_dual_eval")

    logging.info("=" * 70)
    logging.info("  VERIFICATION: v3 model with UNFILTERED HES sequences")
    logging.info("=" * 70)

    # Step 1: Load GP DataModule (same as normal eval)
    from FastEHR.dataloader.foundational_loader import FoundationalDataModule
    dm = FoundationalDataModule(
        path_to_db=cfg.data.path_to_db,
        path_to_ds=cfg.data.path_to_ds,
        load=True,
        tokenizer="tabular",
        batch_size=cfg.data.batch_size,
        max_seq_length=cfg.transformer.block_size,
        global_diagnoses=cfg.data.global_diagnoses,
        repeating_events=cfg.data.repeating_events,
        freq_threshold=cfg.data.unk_freq_threshold,
        min_workers=cfg.data.min_workers,
        overwrite_meta_information=cfg.data.meta_information_path,
        supervised=True,
        supervised_time_scale=getattr(cfg.data, 'supervised_time_scale', 1.0),
        subsample_training=cfg.data.subsample_training,
        seed=cfg.experiment.seed,
    )
    dm.setup("test")

    gp_vocab_size = dm.train_set.tokenizer.vocab_size
    measurements = dm.train_set.meta_information["measurement_tables"][
        dm.train_set.meta_information["measurement_tables"]["count_obs"] > 0
    ]["event"].to_list()
    cfg.head.tokens_for_univariate_regression = dm.encode(measurements)
    cfg.data.num_static_covariates = next(iter(dm.test_dataloader()))['static_covariates'].shape[1]

    # Step 2: Build HES tokenizer (clean — just for tokenizer)
    hes_db_path = cfg.hes_data.path_to_db
    hes_meta_path = cfg.hes_data.meta_information_path
    hes_block_size = getattr(cfg.hes_data, 'hes_block_size', 256)

    with open(hes_meta_path, "rb") as f:
        hes_meta = pickle.load(f)
    hes_tokenizer = HESTokenizer(hes_meta)
    hes_vocab_size = hes_tokenizer.vocab_size

    # Step 3: Build LEAKY HES cache (index_on_age=200 = effectively no filter)
    yob_lookup = load_yob_lookup(cfg.data.path_to_db)

    logging.info("Building HES cache with index_on_age=200 (NO temporal filter)...")
    hes_cache_leaky, _ = build_hes_sequence_cache(
        hes_db_path, hes_meta_path, yob_lookup, index_on_age=200
    )

    # Step 4: Wrap collate with LEAKY cache
    time_scale = getattr(cfg.data, 'supervised_time_scale', 1.0)
    original_collate = dm.collate_fn
    dm.collate_fn = DualCollateWrapper(
        original_collate, hes_cache_leaky, hes_tokenizer,
        hes_block_size=hes_block_size, time_scale=time_scale,
    )

    # Step 5: Use setup_dual_finetune_experiment for proper model + callback setup
    import wandb
    wandb.init(mode="disabled")

    from setup_dual_finetune_experiment import setup_dual_finetune_experiment

    experiment, Experiment, trainer = setup_dual_finetune_experiment(
        cfg=cfg, dm=dm, mode='load_from_finetune',
        checkpoint_gp=cfg.experiment.ckpt_dir + cfg.experiment.run_id + "_" + cfg.experiment.fine_tune_id + ".ckpt",
        checkpoint_hes=cfg.dual.hes_ckpt,
        logger=None,
        vocab_size=gp_vocab_size,
        hes_vocab_size=hes_vocab_size,
    )

    # Step 6: Run test
    logging.info("Running test with UNFILTERED HES sequences...")
    trainer.test(experiment, dataloaders=dm.test_dataloader())
    logging.info("=" * 70)
    logging.info("  VERIFICATION COMPLETE")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
