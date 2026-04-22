"""
run_dual_experiment.py
======================
Entry point for dual-backbone fine-tuning.

Differences from run_experiment.py:
  1. Loads GP DataModule (supervised, using hes_static dataset)
  2. Loads HES pretrain meta_info -> builds HES tokenizer
  3. Builds HES sequence cache from HES DB
  4. Wraps GP DataModule's collate_fn with DualCollateWrapper
  5. Uses DualFineTuneExperiment instead of FineTuneExperiment
  6. Loads GP + HES pretrain weights separately
"""

from omegaconf import DictConfig, OmegaConf
import os
import hydra
import torch
import pytorch_lightning as pl
import logging
import pickle
from pathlib import Path

from FastEHR.dataloader.foundational_loader import FoundationalDataModule
from SurvivEHR.examples.modelling.SurvivEHR.setup_dual_finetune_experiment import (
    setup_dual_finetune_experiment, DualFineTuneExperiment
)
from SurvivEHR.examples.modelling.SurvivEHR.dual_data_module import (
    build_hes_sequence_cache, HESTokenizer, DualCollateWrapper, load_yob_lookup
)
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")


@hydra.main(version_base=None, config_path="confs", config_name="default")
def run(cfg: DictConfig):
    logging.info(f"Running dual-backbone experiment on {os.cpu_count()} CPUs and {torch.cuda.device_count()} GPUs")

    # Logger
    log_id = cfg.experiment.run_id
    if cfg.experiment.fine_tune_id is not None:
        log_id += "_" + cfg.experiment.fine_tune_id
    logger = pl.loggers.WandbLogger(
        project=cfg.experiment.project_name, name=log_id,
        save_dir=cfg.experiment.log_dir, notes=cfg.experiment.notes,
        tags=cfg.experiment.tags
    )
    logging.basicConfig(level=logging.DEBUG)

    # Global settings
    torch.manual_seed(cfg.experiment.seed)
    torch.set_float32_matmul_precision('medium')
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # ========================================================
    # Step 1: Load GP DataModule (supervised, hes_static dataset)
    # ========================================================
    supervised = True
    logging.info("=" * 100)
    logging.info(f"Loading GP DataModule from {cfg.data.path_to_ds}")
    logging.info("=" * 100)

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
        supervised=supervised,
        supervised_time_scale=getattr(cfg.data, 'supervised_time_scale', 1.0),
        subsample_training=cfg.data.subsample_training,
        seed=cfg.experiment.seed,
    )

    gp_vocab_size = dm.train_set.tokenizer.vocab_size
    measurements = dm.train_set.meta_information["measurement_tables"][
        dm.train_set.meta_information["measurement_tables"]["count_obs"] > 0
    ]["event"].to_list()
    cfg.head.tokens_for_univariate_regression = dm.encode(measurements)
    cfg.data.num_static_covariates = next(iter(dm.test_dataloader()))['static_covariates'].shape[1]

    logging.info(f"GP vocab_size={gp_vocab_size}, num_static_covariates={cfg.data.num_static_covariates}")

    # ========================================================
    # Step 2: Load HES meta info and build HES tokenizer
    # ========================================================
    hes_data_cfg = cfg.hes_data
    hes_meta_path = hes_data_cfg.meta_information_path
    hes_db_path = hes_data_cfg.path_to_db
    hes_block_size = getattr(hes_data_cfg, 'hes_block_size', 256)

    logging.info("=" * 100)
    logging.info(f"Loading HES meta info from {hes_meta_path}")
    logging.info("=" * 100)

    with open(hes_meta_path, "rb") as f:
        hes_meta = pickle.load(f)
    hes_tokenizer = HESTokenizer(hes_meta)
    hes_vocab_size = hes_tokenizer.vocab_size

    logging.info(f"HES vocab_size={hes_vocab_size}")

    # ========================================================
    # Step 3: Build HES sequence cache
    # ========================================================
    logging.info("Building HES sequence cache...")
    yob_lookup = load_yob_lookup(cfg.data.path_to_db)
    hes_cache, _ = build_hes_sequence_cache(hes_db_path, hes_meta_path, yob_lookup)

    # ========================================================
    # Step 4: Wrap DataModule collate_fn with DualCollateWrapper
    # ========================================================
    time_scale = getattr(cfg.data, 'supervised_time_scale', 1.0)

    for split_set in [dm.train_set, dm.val_set, dm.test_set]:
        original_collate = split_set.collate_fn
        split_set.collate_fn = DualCollateWrapper(
            original_collate, hes_cache, hes_tokenizer,
            hes_block_size=hes_block_size, time_scale=time_scale,
        )

    # ========================================================
    # Step 5: Set up experiment
    # ========================================================
    dual_cfg = cfg.dual
    gp_ckpt_path = dual_cfg.gp_ckpt
    hes_ckpt_path = dual_cfg.hes_ckpt

    supervised_run_id = cfg.experiment.run_id + "_" + cfg.experiment.fine_tune_id
    supervised_ckpt_path = cfg.experiment.ckpt_dir + supervised_run_id + ".ckpt"

    logging.info("=" * 100)
    logging.info(f"Setting up dual-backbone fine-tune experiment")
    logging.info("=" * 100)

    if Path(supervised_ckpt_path).is_file():
        logging.info(f"Loading existing dual fine-tuned model from {supervised_ckpt_path}")
        assert cfg.experiment.train is False, "Further training on a checkpoint is not supported."
        mode = 'load_from_finetune'
        ft_gp_ckpt = supervised_ckpt_path
        ft_hes_ckpt = hes_ckpt_path
    elif Path(gp_ckpt_path).is_file() and Path(hes_ckpt_path).is_file():
        logging.info(f"Creating new dual fine-tuned model from pretrained checkpoints")
        logging.info(f"  GP ckpt: {gp_ckpt_path}")
        logging.info(f"  HES ckpt: {hes_ckpt_path}")
        assert cfg.experiment.train is True
        mode = 'load_from_pretrain'
        ft_gp_ckpt = gp_ckpt_path
        ft_hes_ckpt = hes_ckpt_path
    else:
        logging.info(f"Creating dual fine-tuned model from scratch")
        assert cfg.experiment.train is True
        mode = 'no_load'
        ft_gp_ckpt = None
        ft_hes_ckpt = None

    experiment_instance, Experiment, trainer = setup_dual_finetune_experiment(
        cfg=cfg, dm=dm, mode=mode,
        checkpoint_gp=ft_gp_ckpt, checkpoint_hes=ft_hes_ckpt,
        logger=logger, vocab_size=gp_vocab_size, hes_vocab_size=hes_vocab_size,
    )

    new_checkpoint = supervised_ckpt_path

    # ========================================================
    # Step 6: Train
    # ========================================================
    if cfg.experiment.train:
        logging.info("Training dual-backbone model.")
        last_ckpt_path = Path(cfg.experiment.ckpt_dir) / "last.ckpt"
        resume_ckpt = str(last_ckpt_path) if last_ckpt_path.is_file() else None
        if resume_ckpt:
            logging.info(f"Resuming from {resume_ckpt}")
        trainer.fit(experiment_instance, datamodule=dm, ckpt_path=resume_ckpt)

    # ========================================================
    # Step 7: Test
    # ========================================================
    if cfg.experiment.test:
        best_path = getattr(trainer.checkpoint_callback, "best_model_path", None) if hasattr(trainer, "checkpoint_callback") else None
        if best_path and Path(best_path).is_file():
            new_checkpoint = best_path
        logging.info(f"Re-loading from best checkpoint {new_checkpoint}")
        experiment_instance = Experiment.load_from_checkpoint(new_checkpoint)
        logging.info("Testing dual-backbone model.")
        trainer.test(experiment_instance, dataloaders=dm.test_dataloader())

    return experiment_instance, dm


if __name__ == "__main__":
    run()
