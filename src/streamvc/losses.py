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
    rms: float = 0.1  # Phase A: Frame RMS supervision
    multiband_rms: float = 0.05  # Phase A: Multi-band RMS supervision


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


def frame_rms_loss(prediction: torch.Tensor, target: torch.Tensor, frame_length: int = 512) -> torch.Tensor:
    """
    Phase A-v2: Absolute RMS target to prevent scale collapse.

    Args:
        prediction: (B, T) predicted audio
        target: (B, T) target audio
        frame_length: frame size for RMS computation

    Returns:
        MSE loss forcing pred RMS to match target RMS (absolute scale)
    """
    # Compute batch-level RMS (absolute scale enforcement)
    pred_rms = torch.sqrt((prediction ** 2).mean() + 1e-8)
    target_rms = torch.sqrt((target ** 2).mean() + 1e-8)

    # Force pred_rms to match target_rms (not just relative shape)
    return F.mse_loss(pred_rms, target_rms)


def multiband_rms_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 16000,
    n_bands: int = 4,
    frame_length: int = 512,
) -> torch.Tensor:
    """
    Phase A: Multi-band RMS matching to prevent high-frequency collapse.

    Splits audio into frequency bands and computes RMS loss per band.
    This prevents scenarios like "mosquito tone" where high frequencies dominate.

    Args:
        prediction: (B, T) predicted audio
        target: (B, T) target audio
        sample_rate: audio sample rate
        n_bands: number of frequency bands
        frame_length: frame size for RMS computation

    Returns:
        Mean RMS loss across all bands
    """
    # Simple band splitting using mel-scale boundaries
    # Band boundaries (Hz): [0, 1000, 2000, 4000, 8000]
    band_boundaries = [0, 1000, 2000, 4000, sample_rate // 2][:n_bands + 1]

    losses = []
    for i in range(n_bands):
        low_freq = band_boundaries[i]
        high_freq = band_boundaries[i + 1]

        # Simple bandpass filtering using STFT masking
        # This is a simplified version; more sophisticated filtering could be used
        win = torch.hann_window(512, device=prediction.device)
        pred_stft = torch.stft(
            prediction,
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=win,
            return_complex=True,
        )
        target_stft = torch.stft(
            target,
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=win,
            return_complex=True,
        )

        # Frequency bin range for this band
        freq_bins = pred_stft.shape[1]
        low_bin = int(low_freq / (sample_rate / 2) * freq_bins)
        high_bin = int(high_freq / (sample_rate / 2) * freq_bins)

        # Mask to isolate band
        mask = torch.zeros_like(pred_stft)
        mask[:, low_bin:high_bin, :] = 1.0

        pred_band_stft = pred_stft * mask
        target_band_stft = target_stft * mask

        # Inverse STFT to get time-domain band signal
        pred_band = torch.istft(
            pred_band_stft,
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=win,
            length=prediction.shape[-1],
        )
        target_band = torch.istft(
            target_band_stft,
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=win,
            length=target.shape[-1],
        )

        # Compute RMS loss for this band
        band_rms_loss = frame_rms_loss(pred_band, target_band, frame_length)
        losses.append(band_rms_loss)

    return torch.stack(losses).mean()


def discriminator_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_fake_outputs: list[torch.Tensor],
) -> torch.Tensor:
    """
    LSGAN-style discriminator loss.

    Trains discriminator to distinguish real (target=1) from fake (target=0) audio.

    Args:
        disc_real_outputs: List of discriminator outputs for real audio (each scale/period)
        disc_fake_outputs: List of discriminator outputs for fake audio (each scale/period)

    Returns:
        Discriminator loss (MSE between predictions and targets)
    """
    loss = 0
    for dr, df in zip(disc_real_outputs, disc_fake_outputs):
        # Real should be close to 1
        loss_real = torch.mean((dr - 1) ** 2)
        # Fake should be close to 0
        loss_fake = torch.mean(df ** 2)
        loss += loss_real + loss_fake

    return loss / len(disc_real_outputs)


def generator_adversarial_loss(
    disc_fake_outputs: list[torch.Tensor],
) -> torch.Tensor:
    """
    LSGAN-style generator adversarial loss.

    Trains generator to fool discriminator (make fake audio look real).

    Args:
        disc_fake_outputs: List of discriminator outputs for generated audio

    Returns:
        Generator adversarial loss (MSE between predictions and target=1)
    """
    loss = 0
    for df in disc_fake_outputs:
        # Generator wants fake to be close to 1 (fool discriminator)
        loss += torch.mean((df - 1) ** 2)

    return loss / len(disc_fake_outputs)


def feature_matching_loss(
    real_features_list: list[list[torch.Tensor]],
    fake_features_list: list[list[torch.Tensor]],
) -> torch.Tensor:
    """
    Feature matching loss (SoundStream/HiFi-GAN style).

    Matches intermediate feature maps between real and fake audio.
    This helps generator learn perceptually similar features beyond just
    fooling the discriminator.

    Args:
        real_features_list: List of feature lists for each discriminator scale/period
        fake_features_list: List of feature lists for each discriminator scale/period

    Returns:
        L1 loss between real and fake features (averaged across all layers)
    """
    loss = 0
    num_features = 0

    for real_feats, fake_feats in zip(real_features_list, fake_features_list):
        for real_feat, fake_feat in zip(real_feats, fake_feats):
            # Feature matching uses detached real features (stop gradient)
            loss += F.l1_loss(fake_feat, real_feat.detach())
            num_features += 1

    # Normalize by total number of feature maps
    return loss / max(num_features, 1)


