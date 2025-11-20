#!/usr/bin/env python3
"""Speaker Encoderのテスト"""

import torch
import soundfile as sf
from pathlib import Path

from streamvc import StreamVCPipeline, load_config


def main():
    # チェックポイント読み込み
    checkpoint_path = Path("runs/streamvc_phase1_ema/checkpoints/step_20000.pt")
    config_path = Path("configs/colab_gpu_training.yaml")

    config = load_config(config_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    num_hubert_labels = checkpoint["model"]["content_head.linear.weight"].shape[0]

    pipeline = StreamVCPipeline(config, num_hubert_labels)
    pipeline.load_state_dict(checkpoint["model"], strict=False)
    pipeline.eval()

    # テスト音声
    source, sr = sf.read("outputs/inference_test/source_original.wav")
    source = torch.from_numpy(source).float().unsqueeze(0)
    target, sr = sf.read("outputs/inference_test/target_reference_original.wav")
    target = torch.from_numpy(target).float().unsqueeze(0)

    # 埋め込み抽出
    with torch.no_grad():
        emb_source = pipeline.encode_speaker(source)
        emb_target = pipeline.encode_speaker(target)

    # コサイン類似度
    cos_sim = torch.nn.functional.cosine_similarity(
        emb_source, emb_target, dim=-1
    ).item()

    print(f"Source embedding shape: {emb_source.shape}")
    print(f"Target embedding shape: {emb_target.shape}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"\nSource embedding stats:")
    print(f"  Mean: {emb_source.mean():.6f}, Std: {emb_source.std():.6f}")
    print(f"  Min: {emb_source.min():.6f}, Max: {emb_source.max():.6f}")
    print(f"\nTarget embedding stats:")
    print(f"  Mean: {emb_target.mean():.6f}, Std: {emb_target.std():.6f}")
    print(f"  Min: {emb_target.min():.6f}, Max: {emb_target.max():.6f}")
    print(f"\nL2 distance: {torch.dist(emb_source, emb_target).item():.6f}")

    if cos_sim > 0.95:
        print("⚠ 警告: 異なる話者なのに類似度が高すぎます（Speaker Encoderが学習できていない可能性）")
    else:
        print("✓ 正常: 異なる話者で適切な類似度です")


if __name__ == "__main__":
    main()
