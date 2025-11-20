#!/usr/bin/env python3
"""Speaker Encoder collapse診断スクリプト

学習中のチェックポイントで以下を確認：
1. Speaker latent統計（cos sim, std）
2. F0/Energy whitening統計
3. Content Encoder の timbre leak
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from streamvc import StreamVCPipeline, load_config
from streamvc.data import StreamVCCacheDataset
from streamvc.data.dataset import _collate_fn


def diagnose_checkpoint(checkpoint_path: Path, config_path: Path, cache_dir: Path):
    """チェックポイントの診断を実行"""
    print("=" * 70)
    print(f"診断対象: {checkpoint_path.name}")
    print("=" * 70)

    # チェックポイント読み込み
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    step = checkpoint.get("step", "unknown")
    print(f"Step: {step}\n")

    # モデル構築
    config = load_config(config_path)
    num_hubert_labels = checkpoint["model"]["content_head.linear.weight"].shape[0]
    pipeline = StreamVCPipeline(config, num_hubert_labels)
    pipeline.load_state_dict(checkpoint["model"], strict=False)
    pipeline.eval()

    # データ読み込み
    dataset = StreamVCCacheDataset(cache_dir, dataset_name="libri_tts", split="train")
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=_collate_fn, shuffle=True)
    batch = next(iter(dataloader))

    print("=== 1. Speaker Encoder 統計 ===")
    with torch.no_grad():
        # Speaker embeddings抽出
        speaker_embeddings = []
        for i in range(batch["source_audio"].shape[0]):
            emb = pipeline.encode_speaker(batch["target_reference"][i:i+1])
            speaker_embeddings.append(emb)
        speaker_embeddings = torch.cat(speaker_embeddings, dim=0)  # (B, D)

        # Pairwise cosine similarity
        cos_sims = []
        for i in range(speaker_embeddings.shape[0]):
            for j in range(i+1, speaker_embeddings.shape[0]):
                cos_sim = torch.nn.functional.cosine_similarity(
                    speaker_embeddings[i:i+1], speaker_embeddings[j:j+1], dim=-1
                ).item()
                cos_sims.append(cos_sim)

        mean_cos_sim = torch.tensor(cos_sims).mean().item()
        std_cos_sim = torch.tensor(cos_sims).std().item()
        print(f"  Mean pairwise cos similarity: {mean_cos_sim:.6f} ± {std_cos_sim:.6f}")

        # Dimension-wise statistics
        dim_stds = speaker_embeddings.std(dim=0)
        print(f"  Mean dimension std: {dim_stds.mean().item():.6f}")
        print(f"  Min dimension std: {dim_stds.min().item():.6f}")
        print(f"  Max dimension std: {dim_stds.max().item():.6f}")

        # Overall std
        overall_std = speaker_embeddings.std().item()
        print(f"  Overall std: {overall_std:.6f}")

        # 判定
        if mean_cos_sim > 0.99 and dim_stds.mean().item() < 0.01:
            print("  ❌ COLLAPSED: Speaker Encoder is collapsed")
        elif mean_cos_sim > 0.9:
            print("  ⚠️  WARNING: High similarity (potential collapse)")
        else:
            print("  ✅ OK: Speaker Encoder is working")

    print("\n=== 2. F0/Energy Whitening 統計 ===")
    with torch.no_grad():
        pitch_energy = pipeline.pitch_extractor(batch["source_audio"], mode="train")
        f0_whiten = pitch_energy.f0_whiten
        energy_whiten = pitch_energy.energy_whiten

        # Per-sample mean variance
        f0_sample_means = []
        for i in range(f0_whiten.shape[0]):
            valid_mask = pitch_energy.f0_hz[i] > 0
            if torch.any(valid_mask):
                f0_sample_means.append(f0_whiten[i][valid_mask].mean().item())

        energy_sample_means = [energy_whiten[i].mean().item() for i in range(energy_whiten.shape[0])]

        f0_mean_var = torch.tensor(f0_sample_means).var().item() if f0_sample_means else 0.0
        energy_mean_var = torch.tensor(energy_sample_means).var().item()

        print(f"  F0 sample mean variance: {f0_mean_var:.6f}")
        print(f"  Energy sample mean variance: {energy_mean_var:.6f}")

        # 判定
        if f0_mean_var < 0.01 and energy_mean_var < 0.01:
            print("  ✅ OK: Whitening is working correctly")
        else:
            print(f"  ⚠️  WARNING: Whitening may have leaks (target < 0.01)")

    print("\n=== 3. Content Encoder Timbre Leak ===")
    with torch.no_grad():
        content_features = pipeline.content_encoder(batch["source_audio"])  # (B, T, D)
        content_pooled = content_features.mean(dim=1)  # (B, D)

        # Pairwise cosine similarity
        content_cos_sims = []
        for i in range(content_pooled.shape[0]):
            for j in range(i+1, content_pooled.shape[0]):
                cos_sim = torch.nn.functional.cosine_similarity(
                    content_pooled[i:i+1], content_pooled[j:j+1], dim=-1
                ).item()
                content_cos_sims.append(cos_sim)

        mean_content_sim = torch.tensor(content_cos_sims).mean().item()
        std_content_sim = torch.tensor(content_cos_sims).std().item()

        print(f"  Mean pairwise cos similarity: {mean_content_sim:.6f} ± {std_content_sim:.6f}")

        # Dimension-wise statistics
        content_dim_stds = content_pooled.std(dim=0).mean().item()
        print(f"  Mean dimension std: {content_dim_stds:.6f}")

        # 判定
        if mean_content_sim > 0.9:
            print("  ✅ Content features are similar (content only)")
        elif mean_content_sim < 0.7:
            print("  ⚠️  WARNING: Content features may contain speaker info")
        else:
            print("  ⚠️  Borderline")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Diagnose Speaker Encoder collapse")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint file (e.g., runs/xxx/checkpoints/step_5000.pt)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/colab_gpu_training.yaml"),
        help="Config file path"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache"),
        help="Cache directory"
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    diagnose_checkpoint(args.checkpoint, args.config, args.cache_dir)


if __name__ == "__main__":
    main()
