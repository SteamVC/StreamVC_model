#!/usr/bin/env python3
"""Content Encoder features の timbre leak を確認"""

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

    # データセット - 同一内容で異なる話者のペアを見つけるのは難しいので、
    # 代わりに「異なる話者の content features がどれくらい異なるか」を見る
    dataset = StreamVCCacheDataset(Path("data/cache"), dataset_name="libri_tts", split="train")
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=_collate_fn, shuffle=True)

    batch = next(iter(dataloader))
    source_audio = batch["source_audio"]

    print("=== Content Features Timbre Leak Check ===\n")

    # Content Encoder で特徴抽出
    with torch.no_grad():
        content_features = pipeline.content_encoder(source_audio)  # (B, T, D)

    print(f"Content features shape: {content_features.shape}")

    # 各サンプルの global average pooling
    content_pooled = content_features.mean(dim=1)  # (B, D)

    print("\n=== Content Features: Per-sample Statistics ===")
    for i in range(content_pooled.shape[0]):
        mean = content_pooled[i].mean().item()
        std = content_pooled[i].std().item()
        norm = content_pooled[i].norm().item()
        print(f"Sample {i}: mean={mean:.4f}, std={std:.4f}, norm={norm:.4f}")

    # バッチ内の類似度
    print("\n=== Content Features: Pairwise Cosine Similarity ===")
    cos_sims = []
    for i in range(content_pooled.shape[0]):
        for j in range(i+1, content_pooled.shape[0]):
            cos_sim = torch.nn.functional.cosine_similarity(
                content_pooled[i:i+1], content_pooled[j:j+1], dim=-1
            ).item()
            cos_sims.append(cos_sim)
            if j <= i+2:  # 最初の数ペアだけ表示
                print(f"Sample {i} vs {j}: {cos_sim:.6f}")

    mean_cos_sim = torch.tensor(cos_sims).mean().item()
    std_cos_sim = torch.tensor(cos_sims).std().item()
    print(f"\nMean pairwise similarity: {mean_cos_sim:.6f} ± {std_cos_sim:.6f}")

    # 次元ごとの統計
    print("\n=== Content Features: Dimension-wise Statistics ===")
    dim_means = content_pooled.mean(dim=0)  # (D,)
    dim_stds = content_pooled.std(dim=0)    # (D,)

    print(f"Mean of dimension means: {dim_means.mean().item():.6f}")
    print(f"Mean of dimension stds: {dim_stds.mean().item():.6f}")
    print(f"Min dimension std: {dim_stds.min().item():.6f}")
    print(f"Max dimension std: {dim_stds.max().item():.6f}")

    # 全体の分散
    overall_std = content_pooled.std().item()
    print(f"\nOverall batch std: {overall_std:.6f}")

    print("\n=== Diagnosis ===")
    print("もし Mean pairwise similarity > 0.9:")
    print("  → Content features が話者間で非常に類似（内容情報のみ）")
    print("もし Mean pairwise similarity < 0.7:")
    print("  → Content features に話者差が混入している可能性")
    print("\nもし Mean of dimension stds > 0.1:")
    print("  → バッチ内（話者間）で特徴が大きく異なる")
    print("  → Timbre 情報が content に漏れている可能性")

    # 参考：同じ処理を Speaker Encoder でも実行
    print("\n\n=== [参考] Speaker Encoder の同じ統計 ===")
    with torch.no_grad():
        speaker_embeddings = []
        for i in range(source_audio.shape[0]):
            emb = pipeline.encode_speaker(source_audio[i:i+1])
            speaker_embeddings.append(emb)
        speaker_embeddings = torch.cat(speaker_embeddings, dim=0)  # (B, D)

    spk_cos_sims = []
    for i in range(speaker_embeddings.shape[0]):
        for j in range(i+1, speaker_embeddings.shape[0]):
            cos_sim = torch.nn.functional.cosine_similarity(
                speaker_embeddings[i:i+1], speaker_embeddings[j:j+1], dim=-1
            ).item()
            spk_cos_sims.append(cos_sim)

    mean_spk_cos_sim = torch.tensor(spk_cos_sims).mean().item()
    std_spk_cos_sim = torch.tensor(spk_cos_sims).std().item()
    print(f"Speaker embeddings mean pairwise similarity: {mean_spk_cos_sim:.6f} ± {std_spk_cos_sim:.6f}")

    spk_dim_stds = speaker_embeddings.std(dim=0)
    print(f"Speaker embeddings mean of dimension stds: {spk_dim_stds.mean().item():.6f}")

    print("\n→ Speaker Encoder が collapse している場合、similarity ≈ 1.0, dim std ≈ 0")
    print("→ Content Encoder の similarity が Speaker より低いなら、content に timbre が混入")


if __name__ == "__main__":
    main()
