#!/usr/bin/env python3
"""F0/Energy whitening の話者情報漏れを確認"""

import torch
from pathlib import Path
from torch.utils.data import DataLoader

from streamvc import StreamVCPipeline, load_config
from streamvc.data import StreamVCCacheDataset
from streamvc.data.dataset import _collate_fn


def main():
    config = load_config(Path("configs/colab_gpu_training.yaml"))

    checkpoint_path = Path("runs/streamvc_phase1_ema/checkpoints/step_20000.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    num_hubert_labels = checkpoint["model"]["content_head.linear.weight"].shape[0]

    pipeline = StreamVCPipeline(config, num_hubert_labels)
    pipeline.load_state_dict(checkpoint["model"], strict=False)
    pipeline.eval()

    # データセット
    dataset = StreamVCCacheDataset(Path("data/cache"), dataset_name="libri_tts", split="train")
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=_collate_fn, shuffle=True)

    batch = next(iter(dataloader))
    source_audio = batch["source_audio"]

    print("=== Whitening Leak Check ===\n")

    # Pitch/Energy 抽出
    with torch.no_grad():
        pitch_energy = pipeline.pitch_extractor(source_audio, mode="train")

    f0_hz = pitch_energy.f0_hz  # (B, T)
    f0_whiten = pitch_energy.f0_whiten  # (B, T)
    energy = pitch_energy.energy  # (B, T)
    energy_whiten = pitch_energy.energy_whiten  # (B, T)

    print("=== F0 Statistics (per sample) ===")
    for i in range(source_audio.shape[0]):
        valid_mask = f0_hz[i] > 0
        if torch.any(valid_mask):
            f0_mean = f0_hz[i][valid_mask].mean().item()
            f0_std = f0_hz[i][valid_mask].std().item()

            whiten_mean = f0_whiten[i][valid_mask].mean().item()
            whiten_std = f0_whiten[i][valid_mask].std().item()

            print(f"Sample {i}:")
            print(f"  Raw F0 - mean: {f0_mean:.2f} Hz, std: {f0_std:.2f} Hz")
            print(f"  Whitened F0 - mean: {whiten_mean:.4f}, std: {whiten_std:.4f}")

    print("\n=== F0 Whitened: Batch-level Statistics ===")
    # バッチ全体での統計（話者差が残っているか確認）
    all_valid_f0 = []
    for i in range(f0_whiten.shape[0]):
        valid_mask = f0_hz[i] > 0
        if torch.any(valid_mask):
            all_valid_f0.append(f0_whiten[i][valid_mask])

    if all_valid_f0:
        all_f0_concat = torch.cat(all_valid_f0)
        print(f"全サンプル統合後:")
        print(f"  Mean: {all_f0_concat.mean().item():.6f}")
        print(f"  Std: {all_f0_concat.std().item():.6f}")

        # 各サンプルの平均値の分散（話者差の指標）
        sample_means = [f0_whiten[i][f0_hz[i] > 0].mean().item() for i in range(f0_whiten.shape[0]) if torch.any(f0_hz[i] > 0)]
        mean_of_means = torch.tensor(sample_means).mean().item()
        std_of_means = torch.tensor(sample_means).std().item()
        print(f"  各サンプル平均の分散: {std_of_means:.6f}")
        print(f"  → もし > 0.1 なら、話者差が残っている")

    print("\n=== Energy Statistics (per sample) ===")
    for i in range(source_audio.shape[0]):
        energy_mean = energy[i].mean().item()
        energy_std = energy[i].std().item()

        whiten_mean = energy_whiten[i].mean().item()
        whiten_std = energy_whiten[i].std().item()

        print(f"Sample {i}:")
        print(f"  Raw Energy - mean: {energy_mean:.6f}, std: {energy_std:.6f}")
        print(f"  Whitened Energy - mean: {whiten_mean:.4f}, std: {whiten_std:.4f}")

    print("\n=== Energy Whitened: Batch-level Statistics ===")
    energy_concat = energy_whiten.flatten()
    print(f"全サンプル統合後:")
    print(f"  Mean: {energy_concat.mean().item():.6f}")
    print(f"  Std: {energy_concat.std().item():.6f}")

    sample_energy_means = [energy_whiten[i].mean().item() for i in range(energy_whiten.shape[0])]
    std_of_energy_means = torch.tensor(sample_energy_means).std().item()
    print(f"  各サンプル平均の分散: {std_of_energy_means:.6f}")
    print(f"  → もし > 0.1 なら、話者差が残っている")

    print("\n=== Diagnosis ===")
    print("正しい per-utterance whitening なら:")
    print("  - 各サンプルの whitened mean ≈ 0, std ≈ 1")
    print("  - 各サンプル平均の分散 ≈ 0（全サンプルが平均0付近）")
    print("\n現在の実装（batch-level whitening）では:")
    print("  - 各サンプル平均がバラバラ → 話者情報が漏れている")


if __name__ == "__main__":
    main()
