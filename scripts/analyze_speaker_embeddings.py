#!/usr/bin/env python3
"""Speaker埋め込みの統計分析"""

import torch
import soundfile as sf
import numpy as np
from pathlib import Path

from streamvc import StreamVCPipeline, load_config


def analyze_checkpoint(checkpoint_path, config, audio_files):
    """チェックポイントの埋め込みを分析"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    num_hubert_labels = checkpoint["model"]["content_head.linear.weight"].shape[0]

    pipeline = StreamVCPipeline(config, num_hubert_labels)
    pipeline.load_state_dict(checkpoint["model"], strict=False)
    pipeline.eval()

    embeddings = []
    for audio_file in audio_files:
        audio, sr = sf.read(audio_file)
        audio = torch.from_numpy(audio).float().unsqueeze(0)

        with torch.no_grad():
            emb = pipeline.encode_speaker(audio)
        embeddings.append(emb.squeeze(0))

    embeddings = torch.stack(embeddings)  # (N, D)

    # 統計計算
    norms = embeddings.norm(dim=-1)
    mean_norm = norms.mean().item()
    std_norm = norms.std().item()

    per_dim_std = embeddings.std(dim=0)
    mean_per_dim_std = per_dim_std.mean().item()
    min_per_dim_std = per_dim_std.min().item()
    max_per_dim_std = per_dim_std.max().item()

    # コサイン類似度（全ペア）
    cos_sim_matrix = torch.nn.functional.cosine_similarity(
        embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1
    )
    # 対角成分を除外
    mask = ~torch.eye(len(embeddings), dtype=torch.bool)
    cos_sims = cos_sim_matrix[mask]
    mean_cos_sim = cos_sims.mean().item()
    std_cos_sim = cos_sims.std().item()

    return {
        "mean_norm": mean_norm,
        "std_norm": std_norm,
        "mean_per_dim_std": mean_per_dim_std,
        "min_per_dim_std": min_per_dim_std,
        "max_per_dim_std": max_per_dim_std,
        "mean_cos_sim": mean_cos_sim,
        "std_cos_sim": std_cos_sim,
    }


def main():
    config = load_config(Path("configs/colab_gpu_training.yaml"))

    # テスト音声（異なる話者を想定）
    audio_files = [
        "outputs/inference_test/source_original.wav",
        "outputs/inference_test/target_reference_original.wav",
    ]

    checkpoints = [
        "runs/streamvc_phase1_ema/checkpoints/step_5000.pt",
        "runs/streamvc_phase1_ema/checkpoints/step_10000.pt",
        "runs/streamvc_phase1_ema/checkpoints/step_15000.pt",
        "runs/streamvc_phase1_ema/checkpoints/step_20000.pt",
    ]

    print("=== Speaker Embedding Analysis Across Training ===\n")
    print(f"{'Step':>10s} {'Norm':>10s} {'PerDimStd':>10s} {'CosSim':>10s}")
    print("-" * 50)

    for ckpt_path in checkpoints:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"{ckpt_path.stem:>10s} {'(not found)':>30s}")
            continue

        stats = analyze_checkpoint(ckpt_path, config, audio_files)
        step = ckpt_path.stem.replace("step_", "")
        print(f"{step:>10s} {stats['mean_norm']:>10.4f} {stats['mean_per_dim_std']:>10.4f} {stats['mean_cos_sim']:>10.4f}")

    print("\n診断:")
    print("  CosSim が 1.0 に近い → 埋め込みが同一（学習していない）")
    print("  CosSim が時間とともに下がる → 話者を区別し始めている")
    print("  PerDimStd が小さすぎる → 次元がcollapse")


if __name__ == "__main__":
    main()
