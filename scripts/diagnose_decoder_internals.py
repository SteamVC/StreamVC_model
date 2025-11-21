#!/usr/bin/env python3
"""Decoder内部の健全性チェック（RVQ, Content Encoder）"""

import argparse
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from tqdm import tqdm

from streamvc import StreamVCPipeline, load_config


def diagnose_decoder_internals(
    checkpoint_path: Path,
    config_path: Path,
    cache_dir: Path,
    num_samples: int = 50,
    device: str = "cpu",
):
    """Decoder内部の診断"""

    print("\n" + "=" * 70)
    print("Decoder内部診断")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path.name}")
    print()

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = load_config(config_path)
    num_hubert_labels = checkpoint["model"]["content_head.linear.weight"].shape[0]
    pipeline = StreamVCPipeline(config, num_hubert_labels)
    pipeline.load_state_dict(checkpoint["model"], strict=False)
    pipeline.eval()
    pipeline.to(device)

    # Collect samples
    cache_path = cache_dir / "libri_tts" / "valid"
    cache_files = list(cache_path.glob("*.pt"))[:num_samples]

    print(f"Testing {len(cache_files)} samples...")
    print()

    # ==================== Content Encoder 診断 ====================
    print("=" * 70)
    print("Content Encoder 診断")
    print("=" * 70)

    content_means = []
    content_stds = []
    content_norms = []

    for cache_file in tqdm(cache_files, desc="Content"):
        cache = torch.load(cache_file)
        source_audio = cache["source_audio"].unsqueeze(0).to(device)

        with torch.no_grad():
            # Extract content features
            c = pipeline.content_encoder(source_audio)  # (B, C, T)

        content_means.append(c.mean().item())
        content_stds.append(c.std().item())
        content_norms.append(c.norm(dim=1).mean().item())

    print(f"Content features (n={len(cache_files)}):")
    print(f"  Mean: {np.mean(content_means):.4f} ± {np.std(content_means):.4f}")
    print(f"  Std:  {np.mean(content_stds):.4f} ± {np.std(content_stds):.4f}")
    print(f"  Norm: {np.mean(content_norms):.4f} ± {np.std(content_norms):.4f}")
    print()

    if abs(np.mean(content_means)) > 10 or np.mean(content_stds) < 0.01:
        print("⚠️  Content Encoder の出力が異常（スケールが極端）")
    elif np.mean(content_stds) < 0.1:
        print("⚠️  Content Encoder の出力が収束しすぎ（collapse の可能性）")
    else:
        print("✅ Content Encoder は正常な範囲")
    print()

    # ==================== RVQ Codebook 診断 ====================
    print("=" * 70)
    print("RVQ Codebook 診断")
    print("=" * 70)

    # Get active quantizers
    num_active = checkpoint.get("num_active_quantizers", 8)
    print(f"Active quantizers: {num_active} / 8")
    print()

    all_codes = {i: [] for i in range(num_active)}

    for cache_file in tqdm(cache_files, desc="RVQ codes"):
        cache = torch.load(cache_file)
        source_audio = cache["source_audio"].unsqueeze(0).to(device)

        with torch.no_grad():
            # Full forward to get RVQ codes
            outputs = pipeline(source_audio, source_audio, mode="train")

            if "codes" in outputs:
                codes_tensor = outputs["codes"]  # (num_q, B, T)

                for q in range(min(num_active, codes_tensor.shape[0])):
                    codes = codes_tensor[q, 0].cpu().numpy()
                    all_codes[q].extend(codes.tolist())

    # Analyze codebook utilization
    print("Codebook utilization per quantizer:")
    for q in range(num_active):
        if all_codes[q]:
            counter = Counter(all_codes[q])
            unique_codes = len(counter)
            total_codes = 1024  # codebook_size
            utilization = unique_codes / total_codes * 100

            # Top 10 most common codes
            top_codes = counter.most_common(10)
            top_usage = sum(count for _, count in top_codes) / len(all_codes[q]) * 100

            print(f"  Q{q}: {unique_codes}/{total_codes} ({utilization:.1f}%)")
            print(f"       Top-10 codes: {top_usage:.1f}% of usage")

            if utilization < 10:
                print(f"       ❌ Codebook collapse（< 10% 使用）")
            elif utilization < 30:
                print(f"       ⚠️  Codebook under-utilized（< 30%）")
            else:
                print(f"       ✅ Healthy utilization")

    print()

    # ==================== Decoder 出力診断 ====================
    print("=" * 70)
    print("Decoder 出力診断")
    print("=" * 70)

    output_means = []
    output_stds = []
    output_peaks = []
    output_rms = []

    for cache_file in tqdm(cache_files[:20], desc="Decoder output"):
        cache = torch.load(cache_file)
        source_audio = cache["source_audio"].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = pipeline(source_audio, source_audio, mode="train")
            audio_out = outputs["audio"]  # (B, 1, T)

        output_means.append(audio_out.mean().item())
        output_stds.append(audio_out.std().item())
        output_peaks.append(audio_out.abs().max().item())
        output_rms.append((audio_out ** 2).mean().sqrt().item())

    print(f"Decoder output statistics (n=20):")
    print(f"  Mean: {np.mean(output_means):.6f} ± {np.std(output_means):.6f}")
    print(f"  Std:  {np.mean(output_stds):.4f} ± {np.std(output_stds):.4f}")
    print(f"  Peak: {np.mean(output_peaks):.4f} ± {np.std(output_peaks):.4f}")
    print(f"  RMS:  {np.mean(output_rms):.4f} ± {np.std(output_rms):.4f}")
    print()

    if abs(np.mean(output_means)) > 0.1:
        print("⚠️  出力の DC bias が大きい")
    if np.mean(output_peaks) < 0.1:
        print("❌ 出力振幅が極端に小さい（スケール崩壊）")
    elif np.mean(output_peaks) > 2.0:
        print("⚠️  出力振幅が大きすぎる（clipping の可能性）")
    else:
        print("✅ 出力スケールは正常範囲")

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Decoder内部診断")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2a_l1only.yaml"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])

    args = parser.parse_args()

    diagnose_decoder_internals(
        args.checkpoint,
        args.config,
        args.cache_dir,
        args.num_samples,
        args.device,
    )


if __name__ == "__main__":
    main()
