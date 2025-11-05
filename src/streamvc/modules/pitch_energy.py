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
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var", torch.ones(1))
        self.register_buffer("num_updates", torch.zeros(1))

    @torch.no_grad()
    def reset_stats(self) -> None:
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.num_updates.zero_()

    def forward(self, audio: torch.Tensor, mode: str = "train") -> PitchEnergyOutput:
        """audio: (B, T)"""

        if audio.dim() != 2:
            raise ValueError("audio must be (B, T)")
        pitches = []
        voiced_probs = []
        energies = []
        for wav in audio:
            # frame_time is in seconds for the API
            frame_time_sec = self.hop_length / self.sample_rate
            pitch = AF.detect_pitch_frequency(
                wav.unsqueeze(0),
                sample_rate=self.sample_rate,
                frame_time=frame_time_sec,
                win_length=30,  # median smoothing window (frames, not samples)
            ).squeeze(0)
            pitches.append(pitch)
            voiced = (pitch > 0).float()
            voiced_probs.append(voiced)
            frames = wav.unfold(0, self.frame_length, self.hop_length)
            energy = frames.pow(2).mean(dim=-1)
            energies.append(energy)
        f0 = torch.nn.utils.rnn.pad_sequence(pitches, batch_first=True)
        voiced_prob = torch.nn.utils.rnn.pad_sequence(voiced_probs, batch_first=True)
        energy = torch.nn.utils.rnn.pad_sequence(energies, batch_first=True)

        if mode == "train":
            valid_mask = f0 > 0
            mean_value = torch.mean(f0[valid_mask]) if torch.any(valid_mask) else torch.tensor(0.0, device=f0.device)
            std_value = torch.std(f0[valid_mask]) if torch.sum(valid_mask) > 1 else torch.tensor(1.0, device=f0.device)
            mean = mean_value
            std = std_value
        else:
            if self.num_updates.item() == 0:
                valid_mask = f0 > 0
                init_mean = torch.mean(f0[valid_mask]) if torch.any(valid_mask) else torch.tensor(0.0, device=f0.device)
                init_var = torch.var(f0[valid_mask]) if torch.sum(valid_mask) > 1 else torch.tensor(1.0, device=f0.device)
                self.running_mean.copy_(init_mean.unsqueeze(0))
                self.running_var.copy_(init_var.unsqueeze(0))
                self.num_updates.fill_(1.0)
            mean = self.running_mean.squeeze(0)
            std = torch.sqrt(self.running_var.squeeze(0))

        if mode == "train":
            std = torch.where(std == 0, torch.tensor(1.0, device=f0.device), std)
            whiten = (f0 - mean) / std
            # EMA更新
            valid = torch.any(f0 > 0)
            if valid:
                batch_mean = mean
                batch_var = std.pow(2)
                if self.num_updates.item() == 0:
                    self.running_mean.copy_(batch_mean.unsqueeze(0))
                    self.running_var.copy_(batch_var.unsqueeze(0))
                    self.num_updates.fill_(1.0)
                else:
                    self.running_mean.mul_(self.ema_decay).add_(batch_mean.unsqueeze(0) * (1 - self.ema_decay))
                    self.running_var.mul_(self.ema_decay).add_(batch_var.unsqueeze(0) * (1 - self.ema_decay))
        else:
            std = torch.where(std == 0, torch.tensor(1.0, device=f0.device), std)
            whiten = (f0 - mean) / std
            valid_mask = f0 > 0
            if torch.any(valid_mask):
                obs_mean = torch.mean(f0[valid_mask])
                obs_var = torch.var(f0[valid_mask]) if torch.sum(valid_mask) > 1 else self.running_var.squeeze(0)
                self.running_mean.mul_(self.ema_decay).add_(obs_mean.unsqueeze(0) * (1 - self.ema_decay))
                self.running_var.mul_(self.ema_decay).add_(obs_var.unsqueeze(0) * (1 - self.ema_decay))

        return PitchEnergyOutput(f0_hz=f0, f0_whiten=whiten, voiced_prob=voiced_prob, energy=energy)

