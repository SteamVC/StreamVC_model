#!/usr/bin/env python3
"""VC出力の波形を可視化して崩壊パターンを診断"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


def visualize_waveforms(audio_dir: Path, output_path: Path):
    """波形を可視化"""

    # Find sample pairs
    self_recon_files = sorted(audio_dir.glob("self_recon_00_*.wav"))
    cross_speaker_files = sorted(audio_dir.glob("cross_speaker_00_*.wav"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("VC Output Waveform Diagnosis (Step 40000)", fontsize=16)

    # ================== Self-reconstruction ==================
    if len(self_recon_files) >= 2:
        src_file = audio_dir / "self_recon_00_source.wav"
        conv_file = audio_dir / "self_recon_00_converted.wav"

        if src_file.exists() and conv_file.exists():
            src_audio, sr = sf.read(src_file)
            conv_audio, _ = sf.read(conv_file)

            # Limit to first 1 second
            duration = min(sr, len(src_audio), len(conv_audio))
            src_audio = src_audio[:duration]
            conv_audio = conv_audio[:duration]

            time = np.arange(len(src_audio)) / sr

            # Source waveform
            axes[0, 0].plot(time, src_audio, linewidth=0.5, color='blue')
            axes[0, 0].set_title("Self-recon: Source (Original)")
            axes[0, 0].set_xlabel("Time (s)")
            axes[0, 0].set_ylabel("Amplitude")
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(-1, 1)

            # Converted waveform
            axes[0, 1].plot(time, conv_audio, linewidth=0.5, color='red')
            axes[0, 1].set_title("Self-recon: Converted (A→A)")
            axes[0, 1].set_xlabel("Time (s)")
            axes[0, 1].set_ylabel("Amplitude")
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(-1, 1)

            # Statistics
            src_rms = np.sqrt(np.mean(src_audio ** 2))
            conv_rms = np.sqrt(np.mean(conv_audio ** 2))
            src_peak = np.max(np.abs(src_audio))
            conv_peak = np.max(np.abs(conv_audio))

            axes[0, 0].text(
                0.02, 0.98,
                f"RMS: {src_rms:.4f}\nPeak: {src_peak:.4f}",
                transform=axes[0, 0].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

            axes[0, 1].text(
                0.02, 0.98,
                f"RMS: {conv_rms:.4f}\nPeak: {conv_peak:.4f}",
                transform=axes[0, 1].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

    # ================== Cross-speaker ==================
    if len(cross_speaker_files) >= 3:
        src_file = audio_dir / "cross_speaker_00_source_spk4859.wav"
        tgt_file = audio_dir / "cross_speaker_00_target_spk8063.wav"
        conv_file = audio_dir / "cross_speaker_00_converted.wav"

        # Use first available files
        src_files = list(audio_dir.glob("cross_speaker_00_source_*.wav"))
        tgt_files = list(audio_dir.glob("cross_speaker_00_target_*.wav"))
        conv_files = list(audio_dir.glob("cross_speaker_00_converted.wav"))

        if src_files and tgt_files and conv_files:
            src_file = src_files[0]
            tgt_file = tgt_files[0]
            conv_file = conv_files[0]

            src_audio, sr = sf.read(src_file)
            tgt_audio, _ = sf.read(tgt_file)
            conv_audio, _ = sf.read(conv_file)

            # Limit to first 1 second
            duration = min(sr, len(src_audio), len(conv_audio))
            src_audio = src_audio[:duration]
            conv_audio = conv_audio[:duration]

            time = np.arange(len(src_audio)) / sr

            # Source waveform
            axes[1, 0].plot(time, src_audio, linewidth=0.5, color='blue')
            axes[1, 0].set_title("Cross-speaker: Source (A)")
            axes[1, 0].set_xlabel("Time (s)")
            axes[1, 0].set_ylabel("Amplitude")
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim(-1, 1)

            # Converted waveform
            axes[1, 1].plot(time, conv_audio, linewidth=0.5, color='red')
            axes[1, 1].set_title("Cross-speaker: Converted (A→B)")
            axes[1, 1].set_xlabel("Time (s)")
            axes[1, 1].set_ylabel("Amplitude")
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(-1, 1)

            # Statistics
            src_rms = np.sqrt(np.mean(src_audio ** 2))
            conv_rms = np.sqrt(np.mean(conv_audio ** 2))
            src_peak = np.max(np.abs(src_audio))
            conv_peak = np.max(np.abs(conv_audio))

            axes[1, 0].text(
                0.02, 0.98,
                f"RMS: {src_rms:.4f}\nPeak: {src_peak:.4f}",
                transform=axes[1, 0].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

            axes[1, 1].text(
                0.02, 0.98,
                f"RMS: {conv_rms:.4f}\nPeak: {conv_peak:.4f}",
                transform=axes[1, 1].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved waveform visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VC出力波形の可視化")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/vc_waveform_diagnosis.png"))

    args = parser.parse_args()

    visualize_waveforms(args.audio_dir, args.output)


if __name__ == "__main__":
    main()
