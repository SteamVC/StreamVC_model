"""StreamVCのコンテントエンコーダ実装。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from torch import nn


def _compute_padding(kernel_size: int, dilation: int) -> int:
    return (kernel_size - 1) * dilation


class CausalConvBlock(nn.Module):
    """ストリーミング対応の因果Convブロック。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = _compute_padding(kernel_size, dilation)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1dの出力は左側に不要なパディングが含まれるため切り落とす
        out = self.conv(x)
        if self.conv.padding[0] > 0:
            trim = self.conv.padding[0]
            out = out[..., trim:]
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        return out.transpose(1, 2)


@dataclass
class ContentEncoderConfig:
    channels: int = 64
    latent_dim: int = 64
    num_layers: int = 6
    kernel_size: int = 7
    stride_schedule: Iterable[int] = (2, 2, 2)
    dilation_growth: int = 2
    dropout: float = 0.1


class ContentEncoder(nn.Module):
    """HuBERT疑似ラベルに合わせて学習される因果コンテントエンコーダ。"""

    def __init__(self, config: ContentEncoderConfig) -> None:
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
                    dropout=config.dropout,
                )
            )
            in_ch = config.channels
            dilation *= config.dilation_growth
        self.blocks = nn.ModuleList(layers)
        self.proj = nn.Conv1d(in_ch, config.latent_dim, kernel_size=1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """16kHzモノラル音声（B, T）をコンテンツ埋め込みへ。"""

        x = audio.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        latent = self.proj(x).transpose(1, 2)
        return latent


class ContentHead(nn.Module):
    """学習時のみ使用するHuBERTラベル向け分類ヘッド。"""

    def __init__(self, latent_dim: int, num_labels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.linear = nn.Linear(latent_dim, num_labels)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        logits = self.linear(self.norm(latent))
        return logits


