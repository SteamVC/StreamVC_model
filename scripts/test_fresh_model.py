#!/usr/bin/env python3
"""初期化直後のモデルのSpeaker Encoder動作を確認"""

import torch
from pathlib import Path
from torch.utils.data import DataLoader

from streamvc import StreamVCPipeline, load_config
from streamvc.data import StreamVCCacheDataset
from streamvc.data.dataset import _collate_fn


def main():
    config = load_config(Path("configs/colab_gpu_training.yaml"))

    # 新しいモデルを初期化（チェックポイントなし）
    num_hubert_labels = 100  # デフォルト値
    pipeline = StreamVCPipeline(config, num_hubert_labels)
    pipeline.eval()

    # データセット
    dataset = StreamVCCacheDataset(Path("data/cache"), dataset_name="libri_tts", split="train")
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=_collate_fn, shuffle=True)

    # 1バッチだけ取得
    batch = next(iter(dataloader))
    source_audio = batch["source_audio"]
    target_reference = batch["target_reference"]

    print("=== Fresh Model (No Training) ===\n")

    # Speaker Encoderに各サンプルを通す
    with torch.no_grad():
        embeddings = []
        for i in range(source_audio.shape[0]):
            emb_source = pipeline.encode_speaker(source_audio[i:i+1])
            emb_ref = pipeline.encode_speaker(target_reference[i:i+1])

            embeddings.append((emb_source, emb_ref))

            print(f"Sample {i}:")
            print(f"  Source - mean: {emb_source.mean().item():.6f}, std: {emb_source.std().item():.6f}, norm: {emb_source.norm().item():.6f}")
            print(f"  Reference - mean: {emb_ref.mean().item():.6f}, std: {emb_ref.std().item():.6f}, norm: {emb_ref.norm().item():.6f}")

            cos_sim = torch.nn.functional.cosine_similarity(emb_source, emb_ref, dim=-1).item()
            l2_dist = torch.dist(emb_source, emb_ref).item()
            print(f"  CosSim: {cos_sim:.8f}, L2: {l2_dist:.8f}\n")

    # バッチ内の全ペアのCosine similarity
    print("=== Pairwise Similarities (Source embeddings) ===")
    source_embs = torch.cat([e[0] for e in embeddings], dim=0)  # (4, D)

    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            cos_sim = torch.nn.functional.cosine_similarity(
                source_embs[i:i+1], source_embs[j:j+1], dim=-1
            ).item()
            print(f"Sample {i} vs {j}: {cos_sim:.8f}")

    # 次元ごとの統計
    print("\n=== Batch-level Statistics ===")
    print(f"Std across batch (per dimension): {source_embs.std(dim=0).mean().item():.6f}")
    print(f"Std within embedding (per sample): {source_embs.std(dim=1).mean().item():.6f}")

    print("\n=== Conclusion ===")
    if source_embs.std(dim=0).mean().item() < 0.001:
        print("⚠ 初期化直後から全サンプルで同じベクトルを生成している")
        print("   → Speaker Encoderの設計に根本的な問題がある")
    else:
        print("✓ 初期化直後は異なるベクトルを生成できる")
        print("   → 学習の過程で同じベクトルに収束してしまった")


if __name__ == "__main__":
    main()
