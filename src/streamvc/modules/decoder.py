"""StreamVCデコーダの実装。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

import torch
from torch import nn

from .rvq import ResidualVQConfig, ResidualVectorQuantizer


def _causal_conv(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    padding = (conv.kernel_size[0] - 1) * conv.dilation[0]
    x = nn.functional.pad(x, (padding, 0))
    return conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.SiLU()
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = _causal_conv(x, self.conv)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = self.act(out).transpose(1, 2)
        out = self.proj(out)
        return residual + out


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int, factor: int, kernel_size: int) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            channels,
            channels,
            kernel_size=factor * 2,
            stride=factor,
            padding=factor // 2 + factor % 2,
        )
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = _causal_conv(x, self.conv)
        return x


@dataclass
class DecoderConfig:
    channels: int = 40
    latent_dim: int = 64
    upsample_factors: Iterable[int] = (2, 2, 2, 2)
    kernel_size: int = 7
    num_residual_blocks: int = 3
    rvq: ResidualVQConfig = field(default_factory=lambda: ResidualVQConfig())


class StreamVCDecoder(nn.Module):
    """Content+f0+speaker条件から波形を生成するデコーダ。"""

    def __init__(
        self,
        config: DecoderConfig,
        side_dim: int,
        speaker_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        input_dim = config.latent_dim + side_dim
        self.input_proj = nn.Linear(input_dim, config.channels)
        self.speaker_proj = nn.Linear(speaker_dim, config.channels)

        self.upsample_blocks = nn.ModuleList(
            [UpsampleBlock(config.channels, factor, config.kernel_size) for factor in config.upsample_factors]
        )

        residual_blocks: List[nn.Module] = []
        for i in range(config.num_residual_blocks):
            dilation = 2 ** i
            residual_blocks.append(ResidualBlock(config.channels, config.kernel_size, dilation))
        self.residual_blocks = nn.ModuleList(residual_blocks)
        self.post = nn.Sequential(
            nn.Conv1d(config.channels, config.channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(config.channels, config.channels, kernel_size=1),
        )

        self.out_proj = nn.Conv1d(config.channels, 1, kernel_size=1)
        self.rvq = ResidualVectorQuantizer(config.rvq)

    def forward(
        self,
        content_units: torch.Tensor,
        side_features: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:

        if side_features.shape[:2] != content_units.shape[:2]:
            raise ValueError("side features and content units must align in time")

        x = torch.cat([content_units, side_features], dim=-1)
        x = self.input_proj(x)
        speaker = self.speaker_proj(speaker_embedding).unsqueeze(1)
        x = x + speaker
        x = x.transpose(1, 2)
        for up in self.upsample_blocks:
            x = up(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.post(x)
        x = x.transpose(1, 2)
        quantized, rvq_loss, codes = self.rvq(x)
        audio = self.out_proj(quantized.transpose(1, 2)).squeeze(1)
        return audio, rvq_loss, codes


