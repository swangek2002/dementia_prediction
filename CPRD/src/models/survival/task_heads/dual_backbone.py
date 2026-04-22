"""
dual_backbone.py
================
Dual-backbone model: GP Transformer + HES Transformer + Fusion + Survival Head

GP backbone processes GP event sequences (Read v2 codes).
HES backbone processes HES event sequences (ICD-10 codes).
Fusion layer combines the two hidden states for survival prediction.
"""

import copy
import torch
from torch import nn
import logging

from SurvivEHR.src.models.TTE.base import TTETransformer


class FusionLayer(nn.Module):
    """Fuse hidden states from two backbones into a single vector."""

    def __init__(self, embed_dim: int, fusion_type: str = "concat_linear"):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == "concat_linear":
            self.proj = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
        elif fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid(),
            )
            self.proj_gp = nn.Linear(embed_dim, embed_dim)
            self.proj_hes = nn.Linear(embed_dim, embed_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, h_gp: torch.Tensor, h_hes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_gp:  (bsz, embed_dim) - GP backbone last hidden state
            h_hes: (bsz, embed_dim) - HES backbone last hidden state
        Returns:
            h_fused: (bsz, embed_dim)
        """
        if self.fusion_type == "concat_linear":
            return self.proj(torch.cat([h_gp, h_hes], dim=-1))
        elif self.fusion_type == "gated":
            gate = self.gate(torch.cat([h_gp, h_hes], dim=-1))
            return gate * self.proj_gp(h_gp) + (1 - gate) * self.proj_hes(h_hes)


class DualBackboneSurvModel(nn.Module):
    """
    Dual-backbone survival model with GP + HES transformers and fusion.

    Contains:
      - gp_transformer: GP backbone (TTETransformer, loaded from GP pretrain checkpoint)
      - hes_transformer: HES backbone (TTETransformer, loaded from HES pretrain checkpoint)
      - fusion: FusionLayer (learned from scratch during fine-tuning)
    """

    def __init__(
        self,
        cfg,
        gp_vocab_size: int,
        hes_vocab_size: int,
        gp_num_static_covariates: int = 35,
        hes_num_static_covariates: int = 27,
        fusion_type: str = "gated",
        hes_block_size: int = 256,
    ):
        super().__init__()

        self.n_embd = cfg.transformer.n_embd
        self.block_size = cfg.transformer.block_size

        # GP Backbone
        self.gp_transformer = TTETransformer(
            cfg, gp_vocab_size,
            num_static_covariates=gp_num_static_covariates
        )

        # HES Backbone - may use different block_size
        hes_cfg = copy.deepcopy(cfg)
        hes_cfg.transformer.block_size = hes_block_size
        self.hes_transformer = TTETransformer(
            hes_cfg, hes_vocab_size,
            num_static_covariates=hes_num_static_covariates
        )

        # Fusion Layer
        self.fusion = FusionLayer(self.n_embd, fusion_type=fusion_type)

    def forward(
        self,
        gp_tokens, gp_ages, gp_values, gp_covariates, gp_attention_mask,
        hes_tokens, hes_ages, hes_values, hes_covariates, hes_attention_mask,
    ):
        """
        Returns:
            h_gp_seq:  (bsz, seq_len_gp, embed_dim)
            h_hes_seq: (bsz, seq_len_hes, embed_dim)
        """
        h_gp_seq = self.gp_transformer(
            tokens=gp_tokens, ages=gp_ages,
            values=gp_values, covariates=gp_covariates,
            attention_mask=gp_attention_mask
        )

        h_hes_seq = self.hes_transformer(
            tokens=hes_tokens, ages=hes_ages,
            values=hes_values, covariates=hes_covariates,
            attention_mask=hes_attention_mask
        )

        return h_gp_seq, h_hes_seq
