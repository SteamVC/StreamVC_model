"""Multi-Scale and Multi-Period Discriminators for StreamVC.

Based on:
- SoundStream (Zeghidour et al., 2021): Multi-Scale Discriminator
- HiFi-GAN (Kong et al., 2020): Multi-Period Discriminator
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminatorBlock(nn.Module):
    """Single discriminator block for one scale (SoundStream-style).

    Uses strided convolutions with grouped convolutions for efficiency.
    Weight normalization for training stability.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        # SoundStream architecture: Progressive channel expansion with striding
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(in_channels, 16, 15, stride=1, padding=7)),
            nn.utils.weight_norm(nn.Conv1d(16, 64, 41, stride=4, groups=4, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(64, 256, 41, stride=4, groups=16, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(256, 1024, 41, stride=4, groups=64, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(1024, 1024, 41, stride=4, groups=256, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(1024, 1024, 5, stride=1, padding=2)),
        ])
        self.conv_post = nn.utils.weight_norm(nn.Conv1d(1024, 1, 3, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with feature extraction.

        Args:
            x: (B, 1, T) input waveform

        Returns:
            output: (B, 1, T') discriminator score map
            features: List of intermediate feature maps for feature matching loss
        """
        features = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)

        x = self.conv_post(x)
        features.append(x)

        return x, features


class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator (SoundStream-style).

    Evaluates audio at multiple temporal resolutions:
    - Scale 1: Original resolution
    - Scale 2: 2x downsampled
    - Scale 3: 4x downsampled

    This allows capturing both fine-grained details and global structure.
    """

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorBlock(in_channels=1),  # Original scale
            DiscriminatorBlock(in_channels=1),  # 2x downsampled
            DiscriminatorBlock(in_channels=1),  # 4x downsampled
        ])
        # Average pooling for downsampling (no learnable parameters)
        self.poolings = nn.ModuleList([
            nn.Identity(),                      # No downsampling
            nn.AvgPool1d(4, 2, padding=2),      # 2x downsample
            nn.AvgPool1d(4, 2, padding=2),      # Additional 2x (total 4x)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Forward pass at multiple scales.

        Args:
            x: (B, 1, T) input waveform

        Returns:
            outputs: List of discriminator score maps for each scale
            features_list: List of feature map lists for each scale (for feature matching)
        """
        outputs = []
        features_list = []

        x_scaled = x
        for i, (pool, disc) in enumerate(zip(self.poolings, self.discriminators)):
            if i > 0:
                x_scaled = pool(x_scaled)

            out, feats = disc(x_scaled)
            outputs.append(out)
            features_list.append(feats)

        return outputs, features_list


class PeriodDiscriminator(nn.Module):
    """Single period discriminator (HiFi-GAN style).

    Reshapes 1D waveform into 2D by grouping samples with a fixed period.
    This helps capture periodic patterns like pitch harmonics.
    """

    def __init__(self, period: int):
        super().__init__()
        self.period = period
        # 2D convolutions on reshaped waveform: [B, 1, T//period, period]
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = nn.utils.weight_norm(nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with period-based reshaping.

        Args:
            x: (B, 1, T) waveform

        Returns:
            output: (B, 1, T'//period, 1) discriminator score map
            features: List of intermediate features for feature matching
        """
        features = []

        # Reshape to 2D: (B, 1, T) -> (B, 1, T//period, period)
        b, c, t = x.shape
        if t % self.period != 0:
            # Pad to multiple of period
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)

        x = self.conv_post(x)
        features.append(x)

        return x, features


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator (HiFi-GAN style).

    Uses multiple prime-number periods to capture different periodic structures:
    - Period 2: Even/odd sample patterns
    - Period 3, 5, 7, 11: Various harmonic structures

    Particularly effective for capturing pitch-related features.
    """

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(2),
            PeriodDiscriminator(3),
            PeriodDiscriminator(5),
            PeriodDiscriminator(7),
            PeriodDiscriminator(11),
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Forward pass with multiple periods.

        Args:
            x: (B, 1, T) waveform

        Returns:
            outputs: List of discriminator scores for each period
            features_list: List of feature lists for each period
        """
        outputs = []
        features_list = []

        for disc in self.discriminators:
            out, feats = disc(x)
            outputs.append(out)
            features_list.append(feats)

        return outputs, features_list


class CombinedDiscriminator(nn.Module):
    """Combined Multi-Scale + Multi-Period Discriminator.

    Combines the strengths of both architectures:
    - MSD: Captures multi-resolution temporal features
    - MPD: Captures periodic/harmonic structures

    Used together for best audio quality.
    """

    def __init__(self, use_msd: bool = True, use_mpd: bool = False):
        super().__init__()
        self.discriminators = nn.ModuleList()

        if use_msd:
            self.discriminators.append(MultiScaleDiscriminator())
        if use_mpd:
            self.discriminators.append(MultiPeriodDiscriminator())

        if len(self.discriminators) == 0:
            raise ValueError("At least one discriminator (MSD or MPD) must be enabled")

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Forward pass through all enabled discriminators.

        Args:
            x: (B, 1, T) waveform

        Returns:
            all_outputs: Combined list of discriminator scores from all discriminators
            all_features: Combined list of feature lists from all discriminators
        """
        all_outputs = []
        all_features = []

        for disc in self.discriminators:
            outputs, features = disc(x)
            all_outputs.extend(outputs)
            all_features.extend(features)

        return all_outputs, all_features
