"""StreamVCの話者エンコーダ。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn

from .content_encoder import CausalConvBlock, _compute_padding


@dataclass
class SpeakerEncoderConfig:
    channels: int = 32
    latent_dim: int = 64
    num_layers: int = 6
    kernel_size: int = 5
    stride_schedule: Iterable[int] = (2, 2)
    dilation_growth: int = 2


class LearnablePooling(nn.Module):
    """単一クエリの注意重み付きプーリング。"""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(latent_dim))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, T, D)
        q = self.query.unsqueeze(0).unsqueeze(0)  # (1,1,D)
        scores = torch.matmul(feats, q.transpose(-1, -2)).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(feats * weights.unsqueeze(-1), dim=1)
        return pooled


class SpeakerEncoder(nn.Module):
    """参照音声をグローバル話者埋め込みへ集約。"""

    def __init__(self, config: SpeakerEncoderConfig) -> None:
        super().__init__()
        self.config = config
        layers: List[nn.Module] = []
        in_ch = 1
        dilation = 1
        stride_schedule = list(config.stride_schedule)
        for i in range(config.num_layers):
            stride = stride_schedule[i] if i < len(stride_schedule) else 1
            layers.append(
                CausalConvBlock(
                    in_channels=in_ch,
                    out_channels=config.channels,
                    kernel_size=config.kernel_size,
                    dilation=dilation,
                    stride=stride,
                )
            )
            in_ch = config.channels
            dilation *= config.dilation_growth
        self.blocks = nn.ModuleList(layers)
        self.proj = nn.Conv1d(in_ch, config.latent_dim, kernel_size=1)
        self.norm = nn.LayerNorm(config.latent_dim)
        self.pool = LearnablePooling(config.latent_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        feats = self.proj(x).transpose(1, 2)
        feats = self.norm(feats)
        embedding = self.pool(feats)
        return embedding


