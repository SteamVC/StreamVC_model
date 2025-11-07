"""Residual Vector Quantizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ResidualVQConfig:
    num_quantizers: int = 8
    codebook_size: int = 1024
    dims: int = 8
    commitment_cost: float = 0.25
    progressive_steps: int = 2000  # Steps per quantizer activation
    use_cosine_sim: bool = False  # Phase A: Back to L2 distance
    use_ste_fix: bool = True  # Phase A: STE consistency fix
    # Phase 1-EMA: Exponential Moving Average update
    use_ema: bool = False  # Enable EMA-based codebook update (DAC-style)
    ema_decay: float = 0.99  # EMA decay rate (0.99 recommended)
    dead_threshold: int = 100  # Minimum usage count to avoid dead code reset


class ResidualVectorQuantizer(nn.Module):
    """SoundStreamで用いられるRVQのPyTorch実装。"""

    def __init__(self, config: ResidualVQConfig) -> None:
        super().__init__()
        self.config = config
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(config.codebook_size, config.dims)) for _ in range(config.num_quantizers)]
        )
        self.num_active_quantizers = config.num_quantizers  # Will be controlled during training

        # Phase A: Per-quantizer affine transform (scale/bias)
        if config.use_ste_fix:
            self.post_scale = nn.Parameter(torch.ones(config.num_quantizers))
            self.post_bias = nn.Parameter(torch.zeros(config.num_quantizers))
            # Final 1x1 conv for output scale adjustment and channel mixing
            self.final_conv = nn.Conv1d(config.dims, config.dims, kernel_size=1)
            # CRITICAL: Identity initialization to preserve scale
            nn.init.eye_(self.final_conv.weight.squeeze())
            nn.init.zeros_(self.final_conv.bias)

        # Phase 2A: Learnable scale for cosine similarity VQ (deprecated)
        if config.use_cosine_sim:
            self.scale = nn.Parameter(torch.ones(config.num_quantizers))

        # Phase 1-EMA: EMA buffers for codebook update
        if config.use_ema:
            # cluster_size: (num_quantizers, codebook_size) - tracks usage count per code
            self.register_buffer('cluster_size', torch.zeros(config.num_quantizers, config.codebook_size))
            # embed_avg: (num_quantizers, codebook_size, dims) - tracks sum of embeddings per code
            self.register_buffer('embed_avg', torch.zeros(config.num_quantizers, config.codebook_size, config.dims))
            # Initialize embed_avg with codebook values
            for q_idx in range(config.num_quantizers):
                self.embed_avg[q_idx].copy_(self.codebooks[q_idx].data)

    def set_num_active_quantizers(self, num: int) -> None:
        """Set number of active quantizers for progressive training."""
        self.num_active_quantizers = min(num, self.config.num_quantizers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # x: (B, T, C) - already normalized in decoder (mean=0, std=1)
        residual = x
        quantized_sum = torch.zeros_like(x)
        commitment_loss = 0.0
        codes: List[torch.Tensor] = []

        # Progressive RVQ: only use first N quantizers during training
        active_codebooks = self.codebooks[:self.num_active_quantizers]

        for q_idx, codebook in enumerate(active_codebooks):
            if self.config.use_cosine_sim:
                # Phase 2A: Cosine similarity VQ (deprecated)
                E = F.normalize(codebook, dim=-1)
                x_n = F.normalize(residual, dim=-1)
                sim = torch.einsum('btd,kd->btk', x_n, E)
                indices = torch.argmax(sim, dim=-1)
                codes.append(indices)
                embeds = F.embedding(indices, E) * self.scale[q_idx]
            else:
                # Phase A: L2 distance VQ in normalized space
                residual_flat = residual.reshape(-1, residual.shape[-1])  # (B*T, C)
                residual_sq = residual_flat.pow(2).sum(dim=-1, keepdim=True)  # (B*T, 1)
                codebook_sq = codebook.pow(2).sum(dim=-1, keepdim=True)  # (K, 1)
                dot_product = torch.matmul(residual_flat, codebook.t())  # (B*T, K)
                distances = residual_sq + codebook_sq.t() - 2 * dot_product  # (B*T, K)

                indices = torch.argmin(distances, dim=-1)  # (B*T,)
                indices = indices.reshape(residual.shape[0], residual.shape[1])  # (B, T)
                codes.append(indices)

                embeds = F.embedding(indices, codebook)  # (B, T, C)

                # Phase 1-EMA: Update codebook using EMA (if enabled and in training mode)
                if self.config.use_ema and self.training:
                    self.ema_update(q_idx, indices.reshape(-1), residual_flat)

            if self.config.use_ste_fix:
                # Phase A: STE in same coordinate space (normalized)
                embeds_st = residual + (embeds - residual).detach()
                # Per-quantizer affine transform
                embeds_scaled = embeds_st * self.post_scale[q_idx] + self.post_bias[q_idx]
                quantized_sum = quantized_sum + embeds_scaled
                residual = residual - embeds  # Next residual in normalized space
                commitment_loss = commitment_loss + (residual.detach() - embeds).pow(2).mean()
            else:
                # Original behavior
                quantized_sum = quantized_sum + embeds
                residual = residual - embeds
                commitment_loss = commitment_loss + (residual - embeds.detach()).pow(2).mean()

        if self.config.use_ste_fix:
            # Final 1x1 conv for scale adjustment and channel mixing
            # (B, T, C) -> (B, C, T) -> Conv1d -> (B, C, T) -> (B, T, C)
            quantized_final = self.final_conv(quantized_sum.transpose(1, 2)).transpose(1, 2)
            return quantized_final, commitment_loss * self.config.commitment_cost, codes
        else:
            # Original STE (coordinate mismatch)
            quantized = x + (quantized_sum - x).detach()
            return quantized, commitment_loss * self.config.commitment_cost, codes

    def ema_update(self, q_idx: int, indices: torch.Tensor, flat_input: torch.Tensor) -> None:
        """
        Update codebook using Exponential Moving Average (DAC-style).
        Memory-efficient version using scatter operations.

        Args:
            q_idx: Quantizer index (0-7)
            indices: (B*T,) Code indices selected for this quantizer
            flat_input: (B*T, C) Input vectors before quantization
        """
        if not self.config.use_ema or not self.training:
            return

        decay = self.config.ema_decay

        with torch.no_grad():
            # First apply decay to existing values
            self.cluster_size[q_idx].mul_(decay)
            self.embed_avg[q_idx].mul_(decay)

            # Count usage per code using bincount (much faster than loop)
            cluster_size_update = torch.bincount(
                indices,
                minlength=self.config.codebook_size
            ).float()

            # Compute sum of embeddings per code using index_add_
            embed_sum = torch.zeros_like(self.embed_avg[q_idx])
            # Use scatter_add for efficiency
            indices_expanded = indices.unsqueeze(1).expand(-1, flat_input.shape[1])
            embed_sum.scatter_add_(0, indices_expanded, flat_input)

            # EMA update
            self.cluster_size[q_idx].add_(cluster_size_update, alpha=1 - decay)
            self.embed_avg[q_idx].add_(embed_sum, alpha=1 - decay)

            # Update codebook
            n = self.cluster_size[q_idx].unsqueeze(1).clamp(min=1.0)  # Avoid division by zero
            updated_codebook = self.embed_avg[q_idx] / n
            self.codebooks[q_idx].data.copy_(updated_codebook)

    def reset_dead_codes(self, q_idx: int) -> int:
        """
        Reset codes with low usage count (dead codes).

        Args:
            q_idx: Quantizer index (0-7)

        Returns:
            Number of dead codes that were reset
        """
        if not self.config.use_ema or not self.training:
            return 0

        threshold = self.config.dead_threshold
        mask = self.cluster_size[q_idx] < threshold
        num_dead = mask.sum().item()

        if num_dead > 0:
            with torch.no_grad():
                # Reinitialize dead codes with small random values
                self.codebooks[q_idx].data[mask] = torch.randn(
                    num_dead, self.config.dims, device=self.codebooks[q_idx].device
                ) * 0.01
                # Reset EMA buffers for dead codes
                self.cluster_size[q_idx][mask] = 0
                self.embed_avg[q_idx][mask] = 0

        return num_dead


