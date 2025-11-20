"""StreamVC推論パイプライン。"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .config import StreamVCConfig
from .modules import (
    ContentEncoder,
    ContentHead,
    PitchEnergyExtractor,
    SpeakerEncoder,
    StreamVCDecoder,
)


class StreamVCPipeline(nn.Module):
    def __init__(self, config: StreamVCConfig, num_hubert_labels: int) -> None:
        super().__init__()
        self.config = config
        self.content_encoder = ContentEncoder(config.model.content_encoder)
        self.content_head = ContentHead(config.model.content_encoder.latent_dim, num_hubert_labels)
        self.speaker_encoder = SpeakerEncoder(config.model.speaker_encoder)
        # Use frame_ms for hop_ms to align with content encoder frame rate
        self.pitch_extractor = PitchEnergyExtractor(
            sample_rate=config.data.sample_rate,
            frame_ms=config.data.frame_ms,
            hop_ms=config.data.frame_ms,  # same as frame_ms for aligned processing
        )
        side_dim = 3  # whitened f0, voiced prob, energy
        self.decoder = StreamVCDecoder(
            config.model.decoder,
            side_dim=side_dim,
            speaker_dim=config.model.speaker_encoder.latent_dim,
        )
        self._speaker_cache: Optional[torch.Tensor] = None

    @torch.no_grad()
    def encode_speaker(self, reference_audio: torch.Tensor) -> torch.Tensor:
        embedding = self.speaker_encoder(reference_audio)
        self._speaker_cache = embedding
        return embedding

    @torch.no_grad()
    def reset_pitch_stats(self) -> None:
        self.pitch_extractor.reset_stats()

    def forward(
        self,
        source_audio: torch.Tensor,
        target_reference: Optional[torch.Tensor] = None,
        mode: str = "train",
    ) -> Dict[str, torch.Tensor]:
        if mode not in {"train", "infer"}:
            raise ValueError("mode must be 'train' or 'infer'")

        if mode == "train" or target_reference is not None:
            if target_reference is None:
                raise ValueError("target_reference must be provided during training")
            speaker_embedding = self.speaker_encoder(target_reference)
        else:
            if self._speaker_cache is None:
                raise RuntimeError("Speaker embedding not cached. Call encode_speaker first.")
            speaker_embedding = self._speaker_cache

        units = self.content_encoder(source_audio)
        pitch = self.pitch_extractor(source_audio, mode="train" if mode == "train" else "infer")

        side = self._build_side_features(pitch, units.shape[1])

        # Stop gradient from Decoder loss to Content Encoder
        # Content Encoder should only be trained by HuBERT CE loss
        units_for_decoder = units.detach() if mode == "train" else units
        audio, rvq_loss, codes, pre_norm_std = self.decoder(units_for_decoder, side, speaker_embedding)

        outputs: Dict[str, torch.Tensor] = {
            "audio": audio,
            "rvq_loss": rvq_loss,
            "pre_rvq_std": pre_norm_std,
        }
        if mode == "train":
            logits = self.content_head(units)
            outputs["content_logits"] = logits
            outputs["codes"] = torch.stack(codes)
        else:
            outputs["codes"] = torch.stack(codes)
        return outputs

    def _build_side_features(self, pitch_output, target_length: int) -> torch.Tensor:
        # Stack features along feature dimension (not batch)
        # f0_whiten, voiced_prob, energy each have shape (B, T)
        # We need to interpolate each to target_length first, then stack
        batch_size = pitch_output.f0_whiten.shape[0]
        device = pitch_output.f0_whiten.device

        # Interpolate each feature independently to target_length
        f0_interp = F.interpolate(
            pitch_output.f0_whiten.unsqueeze(1),
            size=target_length,
            mode="linear",
            align_corners=False
        ).squeeze(1)  # (B, target_length)

        voiced_interp = F.interpolate(
            pitch_output.voiced_prob.unsqueeze(1),
            size=target_length,
            mode="linear",
            align_corners=False
        ).squeeze(1)  # (B, target_length)

        energy_interp = F.interpolate(
            pitch_output.energy_whiten.unsqueeze(1),
            size=target_length,
            mode="linear",
            align_corners=False
        ).squeeze(1)  # (B, target_length)

        # Stack along feature dimension
        features = torch.stack([f0_interp, voiced_interp, energy_interp], dim=2)  # (B, target_length, 3)
        return features

