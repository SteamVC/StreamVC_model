#!/usr/bin/env python3
"""学習ステップ中のSpeaker Encoder出力をデバッグ"""

import torch
from pathlib import Path
from torch.utils.data import DataLoader

from streamvc import StreamVCPipeline, load_config
from streamvc.data import StreamVCCacheDataset
from streamvc.data.dataset import _collate_fn


def main():
    config = load_config(Path("configs/colab_gpu_training.yaml"))

    # チェックポイント読み込み
    checkpoint_path = Path("runs/streamvc_phase1_ema/checkpoints/step_20000.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    num_hubert_labels = checkpoint["model"]["content_head.linear.weight"].shape[0]

    pipeline = StreamVCPipeline(config, num_hubert_labels)
    pipeline.load_state_dict(checkpoint["model"], strict=False)
    pipeline.train()

    # データセット
    dataset = StreamVCCacheDataset(Path("data/cache"), dataset_name="libri_tts", split="train")
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=_collate_fn, shuffle=True)

    # 1バッチだけ取得
    batch = next(iter(dataloader))

    source_audio = batch["source_audio"]
    target_reference = batch["target_reference"]

    print("=== Batch Info ===")
    print(f"Batch size: {source_audio.shape[0]}")
    print(f"Source audio shapes: {[s.shape for s in source_audio]}")
    print(f"Target reference shapes: {[r.shape for r in target_reference]}")

    # Speaker Encoderに各サンプルを通す
    print("\n=== Speaker Encoder Outputs ===")
    with torch.no_grad():
        embeddings = []
        for i in range(source_audio.shape[0]):
            # Sourceからの埋め込み
            emb_source = pipeline.encode_speaker(source_audio[i:i+1])
            # Referenceからの埋め込み
            emb_ref = pipeline.encode_speaker(target_reference[i:i+1])

            embeddings.append((emb_source, emb_ref))

            print(f"\nSample {i}:")
            print(f"  Source embedding - mean: {emb_source.mean().item():.6f}, std: {emb_source.std().item():.6f}, norm: {emb_source.norm().item():.6f}")
            print(f"  Reference embedding - mean: {emb_ref.mean().item():.6f}, std: {emb_ref.std().item():.6f}, norm: {emb_ref.norm().item():.6f}")

            cos_sim = torch.nn.functional.cosine_similarity(emb_source, emb_ref, dim=-1).item()
            l2_dist = torch.dist(emb_source, emb_ref).item()
            print(f"  CosSim: {cos_sim:.8f}, L2: {l2_dist:.8f}")

    # バッチ内の全ペアのCosine similarity
    print("\n=== Pairwise Similarities (Source embeddings) ===")
    source_embs = torch.cat([e[0] for e in embeddings], dim=0)  # (4, D)

    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            cos_sim = torch.nn.functional.cosine_similarity(
                source_embs[i:i+1], source_embs[j:j+1], dim=-1
            ).item()
            print(f"Sample {i} vs {j}: {cos_sim:.8f}")

    # 次元ごとの統計
    print("\n=== Batch-level Statistics ===")
    print(f"Source embeddings:")
    print(f"  Mean across batch: {source_embs.mean(dim=0).abs().mean().item():.6f}")
    print(f"  Std across batch (per dimension): {source_embs.std(dim=0).mean().item():.6f}")
    print(f"  Std within embedding (per sample): {source_embs.std(dim=1).mean().item():.6f}")

    print("\n=== Diagnosis ===")
    print("もし全ペアでCosSim≈1.0 → Speaker Encoderが入力を区別できていない")
    print("もしStd across batch≈0 → 全サンプルで同じベクトルを生成")
    print("もしStd within embedding≈0 → 次元がcollapse")


if __name__ == "__main__":
    main()
