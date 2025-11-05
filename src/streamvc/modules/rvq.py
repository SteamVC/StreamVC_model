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


class ResidualVectorQuantizer(nn.Module):
    """SoundStreamで用いられるRVQのPyTorch実装。"""

    def __init__(self, config: ResidualVQConfig) -> None:
        super().__init__()
        self.config = config
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(config.codebook_size, config.dims)) for _ in range(config.num_quantizers)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        residual = x
        quantized = torch.zeros_like(x)
        commitment_loss = 0.0
        codes: List[torch.Tensor] = []
        for codebook in self.codebooks:
            distances = (
                residual.unsqueeze(-2) - codebook.unsqueeze(0).unsqueeze(0)
            ).pow(2).sum(-1)
            indices = torch.argmin(distances, dim=-1)
            codes.append(indices)
            embeds = F.embedding(indices, codebook)
            quantized = quantized + embeds
            residual = residual - embeds
            commitment_loss = commitment_loss + (residual.detach() - embeds).pow(2).mean()
        quantized = x + (quantized - x).detach()
        return quantized, commitment_loss * self.config.commitment_cost, codes


