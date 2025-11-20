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

        # Energy whitening (per-utterance mean/var normalization)
        if mode == "train":
            energy_mean = energy.mean()
            energy_std = energy.std()
            energy_std = torch.where(energy_std == 0, torch.tensor(1.0, device=energy.device), energy_std)
            energy_whiten = (energy - energy_mean) / energy_std
            # EMA更新
            if self.energy_num_updates.item() == 0:
                self.energy_running_mean.copy_(energy_mean.unsqueeze(0))
                self.energy_running_var.copy_(energy_std.pow(2).unsqueeze(0))
                self.energy_num_updates.fill_(1.0)
            else:
                self.energy_running_mean.mul_(self.ema_decay).add_(energy_mean.unsqueeze(0) * (1 - self.ema_decay))
                self.energy_running_var.mul_(self.ema_decay).add_(energy_std.pow(2).unsqueeze(0) * (1 - self.ema_decay))
        else:
            if self.energy_num_updates.item() == 0:
                energy_mean = energy.mean()
                energy_var = energy.var()
                self.energy_running_mean.copy_(energy_mean.unsqueeze(0))
                self.energy_running_var.copy_(energy_var.unsqueeze(0))
                self.energy_num_updates.fill_(1.0)
            energy_mean = self.energy_running_mean.squeeze(0)
            energy_std = torch.sqrt(self.energy_running_var.squeeze(0))
            energy_std = torch.where(energy_std == 0, torch.tensor(1.0, device=energy.device), energy_std)
            energy_whiten = (energy - energy_mean) / energy_std
            # EMA更新
            obs_mean = energy.mean()
            obs_var = energy.var()
            self.energy_running_mean.mul_(self.ema_decay).add_(obs_mean.unsqueeze(0) * (1 - self.ema_decay))
            self.energy_running_var.mul_(self.ema_decay).add_(obs_var.unsqueeze(0) * (1 - self.ema_decay))

        return PitchEnergyOutput(f0_hz=f0, f0_whiten=whiten, voiced_prob=voiced_prob, energy=energy, energy_whiten=energy_whiten)

