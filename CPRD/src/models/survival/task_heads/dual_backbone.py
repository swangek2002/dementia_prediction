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
        elif fusion_type == "cross_attention":
            # GP last token attends to HES full sequence, and vice versa
            self.cross_attn_gp2hes = nn.MultiheadAttention(
                embed_dim, num_heads=6, batch_first=True, dropout=0.1
            )
            self.cross_attn_hes2gp = nn.MultiheadAttention(
                embed_dim, num_heads=6, batch_first=True, dropout=0.1
            )
            self.norm_gp = nn.LayerNorm(embed_dim)
            self.norm_hes = nn.LayerNorm(embed_dim)
            self.out_proj = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, h_gp: torch.Tensor, h_hes: torch.Tensor,
                h_gp_seq=None, h_hes_seq=None,
                gp_key_padding_mask=None, hes_key_padding_mask=None) -> torch.Tensor:
        """
        Args:
            h_gp:  (bsz, embed_dim) - GP backbone last hidden state
            h_hes: (bsz, embed_dim) - HES backbone last hidden state
            h_gp_seq:  (bsz, seq_gp, embed_dim) - GP full sequence (for cross_attention)
            h_hes_seq: (bsz, seq_hes, embed_dim) - HES full sequence (for cross_attention)
            gp_key_padding_mask:  (bsz, seq_gp) - True where padded (for cross_attention)
            hes_key_padding_mask: (bsz, seq_hes) - True where padded (for cross_attention)
        Returns:
            h_fused: (bsz, embed_dim)
        """
        if self.fusion_type == "concat_linear":
            return self.proj(torch.cat([h_gp, h_hes], dim=-1))
        elif self.fusion_type == "gated":
            gate = self.gate(torch.cat([h_gp, h_hes], dim=-1))
            return gate * self.proj_gp(h_gp) + (1 - gate) * self.proj_hes(h_hes)
        elif self.fusion_type == "cross_attention":
            bsz = h_gp.shape[0]
            # Query: last token (bsz, 1, embed_dim)
            q_gp = h_gp.unsqueeze(1)
            q_hes = h_hes.unsqueeze(1)

            # GP attends to HES sequence
            if h_hes_seq is not None and hes_key_padding_mask is not None:
                # Check for patients with NO HES records (all masked)
                all_masked = hes_key_padding_mask.all(dim=1)  # (bsz,)
                # Temporarily unmask one position to avoid MHA error
                safe_hes_mask = hes_key_padding_mask.clone()
                safe_hes_mask[all_masked, 0] = False

                enriched_gp, _ = self.cross_attn_gp2hes(
                    query=q_gp, key=h_hes_seq, value=h_hes_seq,
                    key_padding_mask=safe_hes_mask,
                )
                enriched_gp = enriched_gp.squeeze(1)  # (bsz, embed_dim)
                # Zero out for patients with no HES
                enriched_gp[all_masked] = 0.0
                enriched_gp = self.norm_gp(h_gp + enriched_gp)
            else:
                enriched_gp = h_gp

            # HES attends to GP sequence
            if h_gp_seq is not None and gp_key_padding_mask is not None:
                enriched_hes, _ = self.cross_attn_hes2gp(
                    query=q_hes, key=h_gp_seq, value=h_gp_seq,
                    key_padding_mask=gp_key_padding_mask,
                )
                enriched_hes = enriched_hes.squeeze(1)  # (bsz, embed_dim)
                enriched_hes = self.norm_hes(h_hes + enriched_hes)
            else:
                enriched_hes = h_hes

            return self.out_proj(torch.cat([enriched_gp, enriched_hes], dim=-1))


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
        self.surv_layer = None  # lives on experiment, not model; needed for callback compat

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
