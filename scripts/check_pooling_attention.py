#!/usr/bin/env python3
"""LearnablePoolingのアテンション重みを確認"""

import torch
import soundfile as sf
from pathlib import Path

from streamvc import StreamVCPipeline, load_config


def main():
    checkpoint_path = Path("runs/streamvc_phase1_ema/checkpoints/step_20000.pt")
    config_path = Path("configs/colab_gpu_training.yaml")

    config = load_config(config_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    num_hubert_labels = checkpoint["model"]["content_head.linear.weight"].shape[0]

    pipeline = StreamVCPipeline(config, num_hubert_labels)
    pipeline.load_state_dict(checkpoint["model"], strict=False)
    pipeline.eval()

    # Test audio
    source, sr = sf.read("outputs/inference_test/source_original.wav")
    source = torch.from_numpy(source).float().unsqueeze(0)
    target, sr = sf.read("outputs/inference_test/target_reference_original.wav")
    target = torch.from_numpy(target).float().unsqueeze(0)

    # Queryベクトルの確認
    print("=== Learnable Query Vector ===")
    query = pipeline.speaker_encoder.pool.query
    print(f"Query shape: {query.shape}")
    print(f"Query mean: {query.mean().item():.6f}")
    print(f"Query std: {query.std().item():.6f}")
    print(f"Query norm: {query.norm().item():.6f}")
    print(f"First 10 values: {query[:10].tolist()}")

    # アテンション重みを抽出するためのフック
    attention_weights = {}

    def hook_fn(module, input, output):
        feats = input[0]  # (B, T, D)
        q = module.query.unsqueeze(0).unsqueeze(0)  # (1,1,D)
        scores = torch.matmul(feats, q.transpose(-1, -2)).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        attention_weights['weights'] = weights.detach()
        attention_weights['feats'] = feats.detach()
        attention_weights['scores'] = scores.detach()

    hook = pipeline.speaker_encoder.pool.register_forward_hook(hook_fn)

    # Source
    print("\n=== Source Audio ===")
    with torch.no_grad():
        emb_source = pipeline.encode_speaker(source)

    weights_source = attention_weights['weights'][0]  # (T,)
    feats_source = attention_weights['feats'][0]  # (T, D)
    scores_source = attention_weights['scores'][0]  # (T,)

    print(f"Feature shape: {feats_source.shape}")
    print(f"Attention weights shape: {weights_source.shape}")
    print(f"Attention scores (raw) - mean: {scores_source.mean().item():.6f}, std: {scores_source.std().item():.6f}")
    print(f"Attention scores - min: {scores_source.min().item():.6f}, max: {scores_source.max().item():.6f}")
    print(f"Attention weights - min: {weights_source.min().item():.8f}, max: {weights_source.max().item():.8f}")
    print(f"Attention entropy: {-torch.sum(weights_source * torch.log(weights_source + 1e-10)).item():.6f}")
    print(f"Max entropy (uniform): {torch.log(torch.tensor(len(weights_source))).item():.6f}")

    # 均等分布との距離
    uniform = torch.ones_like(weights_source) / len(weights_source)
    kl_div = torch.sum(weights_source * torch.log((weights_source + 1e-10) / (uniform + 1e-10))).item()
    print(f"KL divergence from uniform: {kl_div:.6f}")

    # Target
    print("\n=== Target Audio ===")
    with torch.no_grad():
        emb_target = pipeline.encode_speaker(target)

    weights_target = attention_weights['weights'][0]
    feats_target = attention_weights['feats'][0]
    scores_target = attention_weights['scores'][0]

    print(f"Feature shape: {feats_target.shape}")
    print(f"Attention scores (raw) - mean: {scores_target.mean().item():.6f}, std: {scores_target.std().item():.6f}")
    print(f"Attention scores - min: {scores_target.min().item():.6f}, max: {scores_target.max().item():.6f}")
    print(f"Attention weights - min: {weights_target.min().item():.8f}, max: {weights_target.max().item():.8f}")
    print(f"Attention entropy: {-torch.sum(weights_target * torch.log(weights_target + 1e-10)).item():.6f}")

    uniform_target = torch.ones_like(weights_target) / len(weights_target)
    kl_div_target = torch.sum(weights_target * torch.log((weights_target + 1e-10) / (uniform_target + 1e-10))).item()
    print(f"KL divergence from uniform: {kl_div_target:.6f}")

    # 特徴の多様性
    print("\n=== Feature Diversity ===")
    print(f"Source features:")
    print(f"  Mean: {feats_source.mean().item():.6f}, Std: {feats_source.std().item():.6f}")
    print(f"  Per-timestep std (mean): {feats_source.std(dim=1).mean().item():.6f}")
    print(f"  Per-dimension std (mean): {feats_source.std(dim=0).mean().item():.6f}")

    print(f"\nTarget features:")
    print(f"  Mean: {feats_target.mean().item():.6f}, Std: {feats_target.std().item():.6f}")
    print(f"  Per-timestep std (mean): {feats_target.std(dim=1).mean().item():.6f}")
    print(f"  Per-dimension std (mean): {feats_target.std(dim=0).mean().item():.6f}")

    # 時系列間の類似度
    print(f"\nFeature cosine similarity (first vs last timestep):")
    print(f"  Source: {torch.nn.functional.cosine_similarity(feats_source[0], feats_source[-1], dim=0).item():.6f}")
    print(f"  Target: {torch.nn.functional.cosine_similarity(feats_target[0], feats_target[-1], dim=0).item():.6f}")

    hook.remove()

    print("\n=== Diagnosis ===")
    print("もしエントロピーが最大エントロピーに近い → 均等にプーリング（情報損失）")
    print("もし全timestepでcos sim≈1.0 → 特徴が時間方向に変化していない")
    print("もしPer-dimension std≈0 → 特徴がcollapse")


if __name__ == "__main__":
    main()
