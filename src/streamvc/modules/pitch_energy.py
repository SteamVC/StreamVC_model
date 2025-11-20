"""f0とエネルギーのストリーミング抽出器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torchaudio.functional as AF


@dataclass
class PitchEnergyOutput:
    f0_hz: torch.Tensor
    f0_whiten: torch.Tensor
    voiced_prob: torch.Tensor
    energy: torch.Tensor
    energy_whiten: torch.Tensor


class PitchEnergyExtractor(nn.Module):
    """YINに基づくf0/エネルギ抽出と白色化。"""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: float = 20.0,
        hop_ms: Optional[float] = None,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * frame_ms / 1000)
        hop_ms = frame_ms if hop_ms is None else hop_ms
        self.hop_length = int(sample_rate * hop_ms / 1000)
        self.ema_decay = ema_decay
        # F0用の running stats
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var", torch.ones(1))
        self.register_buffer("num_updates", torch.zeros(1))
        # Energy用の running stats
        self.register_buffer("energy_running_mean", torch.zeros(1))
        self.register_buffer("energy_running_var", torch.ones(1))
        self.register_buffer("energy_num_updates", torch.zeros(1))

    @torch.no_grad()
    def reset_stats(self) -> None:
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.num_updates.zero_()
        self.energy_running_mean.zero_()
        self.energy_running_var.fill_(1.0)
        self.energy_num_updates.zero_()

    def forward(self, audio: torch.Tensor, mode: str = "train") -> PitchEnergyOutput:
        """audio: (B, T)"""

        if audio.dim() != 2:
            raise ValueError("audio must be (B, T)")

        # Batch process pitch detection
        frame_time_sec = self.hop_length / self.sample_rate
        f0 = AF.detect_pitch_frequency(
            audio,
            sample_rate=self.sample_rate,
            frame_time=frame_time_sec,
            win_length=30,  # median smoothing window (frames, not samples)
        )  # (B, num_frames)

        voiced_prob = (f0 > 0).float()

        # Batch process energy extraction
        # unfold doesn't work directly on batches, so we need to process each sample
        energies = []
        for wav in audio:
            frames = wav.unfold(0, self.frame_length, self.hop_length)
            energy = frames.pow(2).mean(dim=-1)
            energies.append(energy)
        energy = torch.nn.utils.rnn.pad_sequence(energies, batch_first=True)

        # Per-utterance whitening: normalize each sample independently
        # This removes speaker-dependent F0 range information
        whiten = torch.zeros_like(f0)
        for i in range(f0.shape[0]):
            valid_mask = f0[i] > 0
            if torch.any(valid_mask):
                # Compute mean/std only from voiced frames
                f0_voiced = f0[i][valid_mask]
                mean_i = f0_voiced.mean()
                std_i = f0_voiced.std() if f0_voiced.numel() > 1 else torch.tensor(1.0, device=f0.device)
                std_i = torch.where(std_i == 0, torch.tensor(1.0, device=f0.device), std_i)

                # Normalize this utterance
                whiten[i] = (f0[i] - mean_i) / std_i

                # Update running statistics (for inference mode reference, though not used in train mode)
                if mode == "train":
                    if self.num_updates.item() == 0:
                        self.running_mean.copy_(mean_i.unsqueeze(0))
                        self.running_var.copy_(std_i.pow(2).unsqueeze(0))
                        self.num_updates.fill_(1.0)
                    else:
                        self.running_mean.mul_(self.ema_decay).add_(mean_i.unsqueeze(0) * (1 - self.ema_decay))
                        self.running_var.mul_(self.ema_decay).add_(std_i.pow(2).unsqueeze(0) * (1 - self.ema_decay))
            else:
                # No voiced frames - leave as zeros
                whiten[i] = f0[i]

        # Energy whitening: per-utterance normalization
        # This removes speaker-dependent loudness information
        energy_whiten = torch.zeros_like(energy)
        for i in range(energy.shape[0]):
            energy_i = energy[i]
            mean_i = energy_i.mean()
            std_i = energy_i.std()
            std_i = torch.where(std_i == 0, torch.tensor(1.0, device=energy.device), std_i)

            # Normalize this utterance
            energy_whiten[i] = (energy_i - mean_i) / std_i

            # Update running statistics (for inference mode reference)
            if mode == "train":
                if self.energy_num_updates.item() == 0:
                    self.energy_running_mean.copy_(mean_i.unsqueeze(0))
                    self.energy_running_var.copy_(std_i.pow(2).unsqueeze(0))
                    self.energy_num_updates.fill_(1.0)
                else:
                    self.energy_running_mean.mul_(self.ema_decay).add_(mean_i.unsqueeze(0) * (1 - self.ema_decay))
                    self.energy_running_var.mul_(self.ema_decay).add_(std_i.pow(2).unsqueeze(0) * (1 - self.ema_decay))

        return PitchEnergyOutput(f0_hz=f0, f0_whiten=whiten, voiced_prob=voiced_prob, energy=energy, energy_whiten=energy_whiten)

