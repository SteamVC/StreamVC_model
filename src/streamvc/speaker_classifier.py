"""Speaker Encoder + Classifier for Phase 1 Pretrain."""

from __future__ import annotations

import torch
from torch import nn

from .modules.speaker_encoder import SpeakerEncoder, SpeakerEncoderConfig


class SpeakerClassifier(nn.Module):
    """Speaker Encoder + Linear Classifier for speaker ID prediction.

    Phase 1: Train this module with CE loss for speaker classification.
    Phase 2: Use only the encoder part (frozen) in VC pipeline.
    """

    def __init__(
        self,
        num_speakers: int,
        channels: int = 32,
        latent_dim: int = 128,  # Slightly larger than VC (64) for better pretrain
        num_layers: int = 6,
        kernel_size: int = 5,
        stride_schedule: list[int] = None,
        dilation_growth: int = 2,
    ) -> None:
        super().__init__()

        stride_schedule = stride_schedule or [2, 2]

        # Speaker Encoder (same architecture as StreamVC)
        encoder_config = SpeakerEncoderConfig(
            channels=channels,
            latent_dim=latent_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            stride_schedule=stride_schedule,
            dilation_growth=dilation_growth,
        )
        self.encoder = SpeakerEncoder(encoder_config)

        # Classification head
        self.classifier = nn.Linear(latent_dim, num_speakers)

        self.num_speakers = num_speakers
        self.latent_dim = latent_dim

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio: (B, T) waveform

        Returns:
            embedding: (B, latent_dim) speaker embedding
            logits: (B, num_speakers) classification logits
        """
        # Extract speaker embedding
        embedding = self.encoder(audio)  # (B, latent_dim)

        # Classify
        logits = self.classifier(embedding)  # (B, num_speakers)

        return embedding, logits

    def extract_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding only (for Phase 2)."""
        return self.encoder(audio)
