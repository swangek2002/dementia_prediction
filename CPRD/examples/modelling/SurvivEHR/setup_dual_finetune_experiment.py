"""
setup_dual_finetune_experiment.py
=================================
Dual-backbone fine-tuning experiment.

Key differences from setup_finetune_experiment.py:
  1. Model contains two TTETransformers (GP + HES)
  2. forward() processes GP and HES inputs separately
  3. Extracts last token from both hidden states, fuses them, then feeds to survival head
  4. Checkpoint loading is split: GP pretrain ckpt + HES pretrain ckpt
"""

import logging
import copy
import pytorch_lightning as pl
import torch
from torch import nn
from omegaconf import OmegaConf
from itertools import islice
import importlib
import functools

from SurvivEHR.src.models.survival.task_heads.dual_backbone import (
    DualBackboneSurvModel, FusionLayer
)
from SurvivEHR.src.modules.head_layers.survival.competing_risk import ODESurvCompetingRiskLayer
from SurvivEHR.examples.modelling.SurvivEHR.setup_finetune_experiment import compute_sample_weights
from SurvivEHR.src.models.base_callback import Embedding
from SurvivEHR.src.models.survival.custom_callbacks.clinical_prediction_model import PerformanceMetrics


class DualFineTuneExperiment(pl.LightningModule):

    def __init__(
        self,
        cfg,
        outcome_tokens,
        risk_model,
        gp_vocab_size: int,
        hes_vocab_size: int,
        outcome_token_groups=None,
        fusion_type: str = "gated",
        hes_block_size: int = 256,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Sample weighting configuration
        sw_cfg = getattr(cfg.fine_tuning, "sample_weighting", None)
        if sw_cfg is not None:
            self.weighting_mode = str(getattr(sw_cfg, "mode", "none"))
            self.event_lambda = float(getattr(sw_cfg, "event_lambda", 10.0))
            self.weight_alpha = float(getattr(sw_cfg, "alpha", 2.0))
            self.weight_tau = float(getattr(sw_cfg, "tau", 0.33))
            self.w_t_max = float(getattr(sw_cfg, "w_t_max", 3.0))
            self.w_total_max = float(getattr(sw_cfg, "w_total_max", 20.0))
        else:
            self.weighting_mode = "none"
            self.event_lambda = 10.0
            self.weight_alpha = 0.0
            self.weight_tau = 0.0
            self.w_t_max = 3.0
            self.w_total_max = 20.0

        logging.info(f"Sample weighting mode: {self.weighting_mode}")

        # Dual Backbone Model
        self.model = DualBackboneSurvModel(
            cfg=cfg,
            gp_vocab_size=gp_vocab_size,
            hes_vocab_size=hes_vocab_size,
            gp_num_static_covariates=cfg.data.num_static_covariates,
            hes_num_static_covariates=27,
            fusion_type=fusion_type,
            hes_block_size=hes_block_size,
        )

        # Survival Head (competing risk)
        hidden_dim = cfg.transformer.n_embd

        if outcome_token_groups is not None:
            num_risks = len(outcome_token_groups)
            frozen_groups = [list(g) for g in outcome_token_groups]
            def _grouped_reduce(target_token, _groups=frozen_groups):
                result = torch.zeros_like(target_token)
                for gidx, group in enumerate(_groups):
                    for tid in group:
                        result = torch.where(target_token == tid, gidx + 1, result)
                return result
            self.reduce_to_outcomes = _grouped_reduce
        else:
            num_risks = len(outcome_tokens)
            self.reduce_to_outcomes = lambda target_token: sum(
                [torch.where(target_token == i, idx + 1, 0)
                 for idx, i in enumerate(outcome_tokens)]
            )

        desurv_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.surv_layer = ODESurvCompetingRiskLayer(
            hidden_dim, [32, 32], num_risks=num_risks, device=desurv_device
        )

    def forward(self, batch, return_loss=True, return_generation=False, is_generation=False):
        """
        batch must contain both GP and HES inputs:
          - batch['tokens'], batch['ages'], etc. -> GP inputs
          - batch['hes_tokens'], batch['hes_ages'], etc. -> HES inputs
          - batch['target_token'], batch['target_age_delta'] -> labels
        """
        # GP inputs
        gp_tokens = batch['tokens'].to(self.device)
        gp_ages = batch['ages'].to(self.device)
        gp_values = batch['values'].to(self.device)
        gp_covariates = batch['static_covariates'].to(self.device)
        gp_attention_mask = batch['attention_mask'].to(self.device)

        # HES inputs
        hes_tokens = batch['hes_tokens'].to(self.device)
        hes_ages = batch['hes_ages'].to(self.device)
        hes_values = batch['hes_values'].to(self.device)
        hes_covariates = batch['hes_static_covariates'].to(self.device)
        hes_attention_mask = batch['hes_attention_mask'].to(self.device)

        # Targets
        target_token = batch['target_token'].reshape((-1, 1)).to(self.device)
        target_age_delta = batch['target_age_delta'].reshape((-1, 1)).to(self.device)
        bsz = gp_tokens.shape[0]

        # Forward through both backbones
        h_gp_seq, h_hes_seq = self.model(
            gp_tokens=gp_tokens, gp_ages=gp_ages,
            gp_values=gp_values, gp_covariates=gp_covariates,
            gp_attention_mask=gp_attention_mask,
            hes_tokens=hes_tokens, hes_ages=hes_ages,
            hes_values=hes_values, hes_covariates=hes_covariates,
            hes_attention_mask=hes_attention_mask,
        )

        # Extract GP last token hidden state
        _att_tmp = torch.hstack((gp_attention_mask,
                                  torch.zeros((bsz, 1), device=self.device)))
        gen_mask = gp_attention_mask - _att_tmp[:, 1:]
        h_gp = torch.zeros((bsz, h_gp_seq.shape[-1]), device=self.device)
        for idx in range(bsz):
            h_gp[idx] = h_gp_seq[idx, gen_mask[idx] == 1, :]

        # Extract HES last token hidden state
        # For patients with no HES records, hes_attention_mask is all 0, h_hes stays zero
        h_hes = torch.zeros((bsz, h_hes_seq.shape[-1]), device=self.device)
        _att_tmp_hes = torch.hstack((hes_attention_mask,
                                      torch.zeros((bsz, 1), device=self.device)))
        gen_mask_hes = hes_attention_mask - _att_tmp_hes[:, 1:]
        for idx in range(bsz):
            if gen_mask_hes[idx].sum() == 1:
                h_hes[idx] = h_hes_seq[idx, gen_mask_hes[idx] == 1, :]
            # else: h_hes[idx] remains zero — no HES records

        # Fusion
        h_fused = self.model.fusion(h_gp, h_hes)

        # Survival prediction — DeSurv needs (bsz, seq_len=2, embed_dim)
        in_hidden_state = torch.stack((h_fused, h_fused), dim=1)
        target_tokens = torch.hstack((
            torch.zeros((bsz, 1), device=self.device), target_token
        ))
        target_ages = torch.hstack((
            torch.zeros((bsz, 1), device=self.device), target_age_delta
        ))
        target_attention_mask = torch.ones_like(target_tokens) == 1

        # Sample weights
        _sample_weights = None
        if self.weighting_mode != "none" and return_loss:
            reduced_k = self.reduce_to_outcomes(target_token.reshape(-1))
            _sample_weights = compute_sample_weights(
                t=target_age_delta.reshape(-1),
                k=reduced_k,
                mode=self.weighting_mode,
                event_lambda=self.event_lambda,
                alpha=self.weight_alpha,
                tau=self.weight_tau,
                w_t_max=self.w_t_max,
                w_total_max=self.w_total_max,
            )

        surv_dict, losses_desurv = self.surv_layer.predict(
            in_hidden_state,
            target_tokens=self.reduce_to_outcomes(target_tokens),
            target_ages=target_ages,
            attention_mask=target_attention_mask,
            is_generation=is_generation,
            return_loss=return_loss,
            return_cdf=return_generation,
            sample_weights=_sample_weights,
        )

        if return_loss:
            loss = torch.sum(torch.stack(losses_desurv))
        else:
            loss = None

        outputs = {"surv": surv_dict}
        losses = {"loss": loss, "loss_desurv": loss, "loss_values": torch.tensor(0.0)}

        return outputs, losses, h_fused

    def training_step(self, batch, batch_idx):
        _, loss_dict, _ = self(batch)
        for k, v in loss_dict.items():
            if v is not None:
                self.log(f"train_{k}", v, prog_bar=False, logger=True, sync_dist=True)
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        _, loss_dict, _ = self(batch)
        for k, v in loss_dict.items():
            if v is not None:
                self.log(f"val_{k}", v, prog_bar=False, logger=True, sync_dist=True)
        return loss_dict['loss']

    def test_step(self, batch, batch_idx):
        _, loss_dict, _ = self(batch)
        for k, v in loss_dict.items():
            if v is not None:
                self.log(f"test_{k}", v, prog_bar=False, logger=True, sync_dist=True)
        return loss_dict['loss']

    def configure_optimizers(self):
        # Differential learning rates:
        #   - GP/HES backbones: lower LR (pretrained)
        #   - Fusion + Survival head: higher LR (from scratch)
        params = [
            {"params": self.model.gp_transformer.parameters(),
             "lr": self.cfg.optim.learning_rate},
            {"params": self.model.hes_transformer.parameters(),
             "lr": self.cfg.optim.learning_rate},
            {"params": self.model.fusion.parameters(),
             "lr": self.cfg.fine_tuning.head.learning_rate},
            {"params": self.surv_layer.parameters(),
             "lr": self.cfg.fine_tuning.head.learning_rate},
        ]
        optimizer = torch.optim.AdamW(params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


def load_dual_pretrained_weights(model, gp_ckpt_path, hes_ckpt_path):
    """Load backbone weights from two independent pretrain checkpoints."""

    model_sd = model.state_dict()

    # --- GP backbone ---
    gp_ckpt = torch.load(gp_ckpt_path, map_location='cpu')
    gp_sd = gp_ckpt.get('state_dict', gp_ckpt)

    gp_mapping = {}
    for k, v in gp_sd.items():
        if k.startswith("model.transformer."):
            new_key = k.replace("model.transformer.", "gp_transformer.")
            gp_mapping[new_key] = v

    # Handle static_proj size mismatch (27 -> 35)
    for k, v in list(gp_mapping.items()):
        if k in model_sd and v.shape != model_sd[k].shape:
            if 'static_proj' in k:
                new_p = model_sd[k].clone()
                if v.dim() == 2:
                    new_p[:v.shape[0], :v.shape[1]] = v
                else:
                    new_p[:v.shape[0]] = v
                gp_mapping[k] = new_p
                logging.info(f"GP partial load {k}: {v.shape} -> {model_sd[k].shape}")
            else:
                logging.warning(f"GP skip {k}: {v.shape} vs {model_sd[k].shape}")
                del gp_mapping[k]

    # --- HES backbone ---
    hes_ckpt = torch.load(hes_ckpt_path, map_location='cpu')
    hes_sd = hes_ckpt.get('state_dict', hes_ckpt)

    hes_mapping = {}
    for k, v in hes_sd.items():
        if k.startswith("model.transformer."):
            new_key = k.replace("model.transformer.", "hes_transformer.")
            hes_mapping[new_key] = v

    for k, v in list(hes_mapping.items()):
        if k in model_sd and v.shape != model_sd[k].shape:
            logging.warning(f"HES skip {k}: {v.shape} vs {model_sd[k].shape}")
            del hes_mapping[k]

    # --- Combined load ---
    combined = {**gp_mapping, **hes_mapping}
    missing, unexpected = model.load_state_dict(combined, strict=False)

    logging.info(f"Loaded dual pretrained weights:")
    logging.info(f"  GP backbone: {len(gp_mapping)} keys")
    logging.info(f"  HES backbone: {len(hes_mapping)} keys")
    logging.info(f"  Missing (new layers): {len(missing)} keys")
    logging.info(f"  Unexpected: {len(unexpected)} keys")

    return model


def setup_dual_finetune_experiment(cfg, dm, mode, checkpoint_gp, checkpoint_hes,
                                    logger=None, vocab_size=None, hes_vocab_size=None):
    """
    Set up the dual-backbone fine-tuning experiment.
    """
    assert dm.is_supervised, "DataModule must be supervised."

    # Outcomes
    outcomes = cfg.fine_tuning.fine_tune_outcomes
    outcome_token_groups = None
    first_el = outcomes[0] if outcomes else None
    is_grouped = first_el is not None and hasattr(first_el, '__iter__') and not isinstance(first_el, str)

    if is_grouped:
        outcome_token_groups = [dm.encode(list(g)) for g in outcomes]
        outcome_tokens = [t for g in outcome_token_groups for t in g]
        logging.info(f"Grouped outcomes: {len(outcome_token_groups)} groups")
    else:
        outcome_tokens = dm.encode(outcomes)
        logging.info(f"Flat outcomes: {outcome_tokens}")

    # Fusion config
    dual_cfg = getattr(cfg, 'dual', None)
    fusion_type = getattr(dual_cfg, 'fusion_type', 'gated') if dual_cfg else 'gated'
    hes_block_size = 256
    if hasattr(cfg, 'hes_data'):
        hes_block_size = getattr(cfg.hes_data, 'hes_block_size', 256)

    # Create experiment
    if mode == 'load_from_finetune':
        experiment = DualFineTuneExperiment.load_from_checkpoint(
            checkpoint_gp, cfg=cfg, outcome_tokens=outcome_tokens,
            risk_model="competing-risk", gp_vocab_size=vocab_size,
            hes_vocab_size=hes_vocab_size, outcome_token_groups=outcome_token_groups,
            fusion_type=fusion_type, hes_block_size=hes_block_size,
        )
    else:
        experiment = DualFineTuneExperiment(
            cfg, outcome_tokens, risk_model="competing-risk",
            gp_vocab_size=vocab_size, hes_vocab_size=hes_vocab_size,
            outcome_token_groups=outcome_token_groups,
            fusion_type=fusion_type, hes_block_size=hes_block_size,
        )

        if mode == 'load_from_pretrain':
            load_dual_pretrained_weights(experiment.model, checkpoint_gp, checkpoint_hes)

    print(experiment)

    # Logger
    logger = logger if cfg.experiment.log else None

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.experiment.ckpt_dir,
        filename=cfg.experiment.run_id + "_" + cfg.experiment.fine_tune_id,
        verbose=cfg.experiment.verbose,
        monitor="val_loss",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_monitor]

    if cfg.optim.early_stop:
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_loss", mode="min", min_delta=0,
            patience=cfg.optim.early_stop_patience,
            verbose=cfg.experiment.verbose,
        )
        callbacks.append(early_stop_callback)

    # Performance metrics callback
    if cfg.fine_tuning.use_callbacks.performance_metrics:
        if outcome_token_groups is not None:
            outcome_token_to_desurv_output_index = {}
            for gidx, group in enumerate(outcome_token_groups):
                for token in group:
                    outcome_token_to_desurv_output_index[token] = gidx
        else:
            outcome_token_to_desurv_output_index = {
                token: token_idx for token_idx, token in enumerate(outcome_tokens)
            }
        metric_callback = PerformanceMetrics(
            outcome_token_to_desurv_output_index=outcome_token_to_desurv_output_index,
            log_combined=True, log_individual=True,
            log_ctd=True, log_ibs=True, log_inbll=True,
        )
        callbacks.append(metric_callback)

    # Trainer
    from helpers import is_interactive
    USE_GPU = torch.cuda.is_available()
    test_only = not cfg.experiment.train and cfg.experiment.test
    if test_only:
        strategy = "auto"
    elif is_interactive():
        strategy = "auto"
    elif USE_GPU:
        strategy = "ddp"
    else:
        strategy = "auto"

    _trainer = pl.Trainer(
        logger=logger,
        precision=32,
        strategy=strategy,
        callbacks=callbacks,
        max_epochs=cfg.optim.num_epochs,
        log_every_n_steps=cfg.optim.log_every_n_steps,
        val_check_interval=cfg.optim.val_check_interval,
        limit_val_batches=cfg.optim.limit_val_batches,
        limit_test_batches=cfg.optim.limit_test_batches,
        accumulate_grad_batches=cfg.optim.accumulate_grad_batches,
        gradient_clip_val=1.0,
    )

    return experiment, DualFineTuneExperiment, _trainer
