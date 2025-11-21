#!/usr/bin/env python3
"""Speaker Latent Sensitivity Test

Decoder が speaker embedding z を実際に使用しているかを検証:
1. 固定した source audio について、複数の reference speaker から z を抽出
2. 各 z で VC を実行
3. 出力の mel spectrogram の変化を可視化・定量化

もし z を変えても出力がほぼ変わらない場合、
Decoder が speaker latent を無視していると判断できる。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from streamvc import StreamVCPipeline, load_config
from streamvc.data import StreamVCCacheDataset
from streamvc.data.dataset import _collate_fn


def mel_spectrogram(audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """Convert waveform to mel spectrogram."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
    )
    return mel_transform(audio)


def test_sensitivity(
    checkpoint_path: Path,
    config_path: Path,
    cache_dir: Path,
    num_sources: int = 3,
    num_references: int = 5,
    output_dir: Path = Path("sensitivity_test"),
):
    """Speaker latent sensitivity test を実行"""

    print("=" * 70)
    print("Speaker Latent Sensitivity Test")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Num sources: {num_sources}")
    print(f"Num references per source: {num_references}")
    print()

    # Output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    step = checkpoint.get("step", "unknown")
    print(f"Step: {step}\n")

    # Load model
    config = load_config(config_path)
    num_hubert_labels = checkpoint["model"]["content_head.linear.weight"].shape[0]
    pipeline = StreamVCPipeline(config, num_hubert_labels)
    pipeline.load_state_dict(checkpoint["model"], strict=False)
    pipeline.eval()

    # Load dataset
    dataset = StreamVCCacheDataset(cache_dir, dataset_name="libri_tts", split="valid")
    dataloader = DataLoader(dataset, batch_size=num_sources + num_references, collate_fn=_collate_fn, shuffle=True)

    batch = next(iter(dataloader))

    print("=== Testing Speaker Latent Usage ===\n")

    with torch.no_grad():
        for src_idx in range(num_sources):
            print(f"Source {src_idx + 1}/{num_sources}:")

            # Source audio
            source_audio = batch["source_audio"][src_idx:src_idx+1]  # (1, T)

            # Extract multiple reference speaker embeddings
            z_refs = []
            for ref_idx in range(num_references):
                ref_audio = batch["target_reference"][num_sources + ref_idx:num_sources + ref_idx+1]
                z_ref = pipeline.encode_speaker(ref_audio)  # (1, D)
                z_refs.append(z_ref)

            # Generate VC outputs for each z
            outputs = []
            mels = []

            for ref_idx, z_ref in enumerate(z_refs):
                # VC
                output = pipeline(source_audio, z_ref)["audio"]  # (1, T)
                outputs.append(output)

                # Mel spectrogram
                mel = mel_spectrogram(output.squeeze(0))  # (n_mels, T)
                mels.append(mel)

            # Stack mels for comparison
            mels_stacked = torch.stack(mels, dim=0)  # (num_refs, n_mels, T)

            # Compute pairwise L2 distances
            distances = []
            for i in range(num_references):
                for j in range(i+1, num_references):
                    dist = torch.norm(mels[i] - mels[j]).item()
                    distances.append(dist)

            mean_dist = np.mean(distances)
            std_dist = np.std(distances)

            print(f"  Mel pairwise L2 distance: {mean_dist:.4f} ± {std_dist:.4f}")

            #判定
            if mean_dist < 1.0:
                print(f"  ❌ WARNING: Very low variation! Decoder may be ignoring z")
            elif mean_dist < 5.0:
                print(f"  ⚠️  Low variation. Decoder uses z weakly")
            else:
                print(f"  ✅ OK: Decoder responds to z")

            # Save visualization
            fig, axes = plt.subplots(num_references, 1, figsize=(12, 2 * num_references))
            if num_references == 1:
                axes = [axes]

            for ref_idx, mel in enumerate(mels):
                ax = axes[ref_idx]
                ax.imshow(mel.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
                ax.set_title(f"Reference {ref_idx + 1}")
                ax.set_ylabel("Mel bins")
                if ref_idx == num_references - 1:
                    ax.set_xlabel("Time")

            fig.suptitle(f"Source {src_idx + 1} - Speaker Latent Sensitivity (Step {step})\nMean distance: {mean_dist:.2f}")
            fig.tight_layout()

            output_path = output_dir / f"source_{src_idx}_step_{step}.png"
            plt.savefig(output_path, dpi=150)
            plt.close()

            print(f"  Saved: {output_path}\n")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test speaker latent sensitivity")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/colab_gpu_training.yaml"),
        help="Config file path",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache"),
        help="Cache directory",
    )
    parser.add_argument(
        "--num-sources",
        type=int,
        default=3,
        help="Number of source audios to test",
    )
    parser.add_argument(
        "--num-references",
        type=int,
        default=5,
        help="Number of reference speakers per source",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sensitivity_test"),
        help="Output directory for visualizations",
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    test_sensitivity(
        args.checkpoint,
        args.config,
        args.cache_dir,
        args.num_sources,
        args.num_references,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
