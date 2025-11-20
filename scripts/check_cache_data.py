#!/usr/bin/env python3
"""キャッシュファイルの内容を確認"""

import torch
from pathlib import Path


def main():
    cache_dir = Path("data/cache/libri_tts/train")
    cache_files = list(cache_dir.glob("*.pt"))[:5]  # 最初の5ファイル

    print("=== Cache File Contents ===\n")

    for i, cache_file in enumerate(cache_files, 1):
        print(f"[{i}] {cache_file.name}")
        data = torch.load(cache_file, map_location="cpu")

        source_audio = data["source_audio"]
        target_wave = data["target_wave"]
        target_reference = data["target_reference"]
        hubert_labels = data["hubert_labels"]

        print(f"  source_audio shape: {source_audio.shape}")
        print(f"  target_wave shape: {target_wave.shape}")
        print(f"  target_reference shape: {target_reference.shape}")
        print(f"  hubert_labels shape: {hubert_labels.shape}")

        # source と target_wave の同一性
        are_identical = torch.allclose(source_audio, target_wave, atol=1e-6)
        print(f"  source == target_wave: {are_identical}")

        # source と target_reference の波形統計
        print(f"  source_audio - mean: {source_audio.mean().item():.6f}, std: {source_audio.std().item():.6f}")
        print(f"  target_reference - mean: {target_reference.mean().item():.6f}, std: {target_reference.std().item():.6f}")

        # 最初の16000サンプルで比較
        min_len = min(source_audio.shape[0], target_reference.shape[0])
        source_crop = source_audio[:min_len]
        ref_crop = target_reference[:min_len]

        l2_dist = torch.dist(source_crop, ref_crop).item()
        cos_sim = torch.nn.functional.cosine_similarity(
            source_crop.unsqueeze(0), ref_crop.unsqueeze(0), dim=-1
        ).item()
        print(f"  First {min_len} samples - L2: {l2_dist:.6f}, CosSim: {cos_sim:.6f}")

        print()

    print("=== Diagnosis ===")
    print("もし source == target_wave が False → 設定ミス")
    print("もし source vs reference の L2 が 0 に近い → 同じ音声がロードされている")
    print("もし全ての reference が同じ shape → 全て同じファイルの可能性")


if __name__ == "__main__":
    main()
