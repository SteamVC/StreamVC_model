"""StreamVCの学習で使用する損失関数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    content_ce: float = 1.0
    stft: float = 1.0
    l1: float = 10.0
    adversarial: float = 0.0
    feature_matching: float = 0.0


def multi_resolution_stft_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    n_ffts: Sequence[int] = (2048, 1024, 512),
    hop_lengths: Sequence[int] = (512, 256, 128),
    win_lengths: Sequence[int] = (2048, 1024, 512),
) -> torch.Tensor:
    if not (len(n_ffts) == len(hop_lengths) == len(win_lengths)):
        raise ValueError("FFT/ hop/ win lists must have same length")
    loss = torch.tensor(0.0, device=prediction.device)
    for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths):
        win = torch.hann_window(win_length, device=prediction.device)
        pred_spec = torch.stft(
            prediction,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=win,
            return_complex=True,
        )
        target_spec = torch.stft(
            target,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=win,
            return_complex=True,
        )
        sc_loss = spectral_convergence(pred_spec, target_spec)
        log_mag = log_stft_magnitude(pred_spec, target_spec)
        loss = loss + sc_loss + log_mag
    return loss / len(n_ffts)


def spectral_convergence(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Use magnitude for complex tensors (MPS compatible)
    diff = torch.abs(target) - torch.abs(pred)
    num = torch.norm(diff)
    den = torch.norm(torch.abs(target))
    return num / (den + 1e-9)


def log_stft_magnitude(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(torch.log1p(torch.abs(target)), torch.log1p(torch.abs(pred)))


def content_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: (B, T_logits, C), labels: (B, T_labels)
    # Interpolate logits to match labels length
    if logits.shape[1] != labels.shape[1]:
        logits = F.interpolate(
            logits.transpose(1, 2),  # (B, C, T)
            size=labels.shape[1],
            mode='linear',
            align_corners=False
        ).transpose(1, 2)  # (B, T_labels, C)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


