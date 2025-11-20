#!/usr/bin/env python3
"""Speaker Encoderが入力に対して異なる出力を生成できるか確認"""

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

    # Test 1: 異なる音声ファイル
    print("=== Test 1: Different Audio Files ===")
    source, sr = sf.read("outputs/inference_test/source_original.wav")
    source = torch.from_numpy(source).float().unsqueeze(0)
    target, sr = sf.read("outputs/inference_test/target_reference_original.wav")
    target = torch.from_numpy(target).float().unsqueeze(0)

    with torch.no_grad():
        emb_source = pipeline.encode_speaker(source)
        emb_target = pipeline.encode_speaker(target)

    cos_sim = torch.nn.functional.cosine_similarity(
        emb_source, emb_target, dim=-1
    ).item()
    l2_dist = torch.dist(emb_source, emb_target).item()

    print(f"Source vs Target:")
    print(f"  Cosine similarity: {cos_sim:.8f}")
    print(f"  L2 distance: {l2_dist:.8f}")

    # Test 2: ランダムノイズ
    print("\n=== Test 2: Random Noise ===")
    noise1 = torch.randn(1, 16000)
    noise2 = torch.randn(1, 16000)

    with torch.no_grad():
        emb_noise1 = pipeline.encode_speaker(noise1)
        emb_noise2 = pipeline.encode_speaker(noise2)

    cos_sim_noise = torch.nn.functional.cosine_similarity(
        emb_noise1, emb_noise2, dim=-1
    ).item()
    l2_dist_noise = torch.dist(emb_noise1, emb_noise2).item()

    print(f"Noise1 vs Noise2:")
    print(f"  Cosine similarity: {cos_sim_noise:.8f}")
    print(f"  L2 distance: {l2_dist_noise:.8f}")

    # Test 3: ゼロ入力
    print("\n=== Test 3: Zero vs Non-zero Input ===")
    zero_input = torch.zeros(1, 16000)

    with torch.no_grad():
        emb_zero = pipeline.encode_speaker(zero_input)
        emb_source = pipeline.encode_speaker(source)

    cos_sim_zero = torch.nn.functional.cosine_similarity(
        emb_zero, emb_source, dim=-1
    ).item()
    l2_dist_zero = torch.dist(emb_zero, emb_source).item()

    print(f"Zero vs Source:")
    print(f"  Cosine similarity: {cos_sim_zero:.8f}")
    print(f"  L2 distance: {l2_dist_zero:.8f}")

    # Test 4: 中間層の活性化を確認
    print("\n=== Test 4: Intermediate Activations ===")

    def hook_fn(name):
        activations = {}

        def hook(module, input, output):
            activations[name] = output.detach()

        return hook, activations

    # 最初と最後のブロックにフックを設定
    hook1, act1 = hook_fn("block0")
    hook2, act2 = hook_fn("block5")
    hook3, act3 = hook_fn("pool")

    h1 = pipeline.speaker_encoder.blocks[0].register_forward_hook(hook1)
    h2 = pipeline.speaker_encoder.blocks[5].register_forward_hook(hook2)
    h3 = pipeline.speaker_encoder.pool.register_forward_hook(hook3)

    with torch.no_grad():
        _ = pipeline.encode_speaker(source)
        act_source = {
            "block0": act1["block0"].clone(),
            "block5": act2["block5"].clone(),
            "pool": act3["pool"].clone(),
        }

        _ = pipeline.encode_speaker(target)
        act_target = {
            "block0": act1["block0"].clone(),
            "block5": act2["block5"].clone(),
            "pool": act3["pool"].clone(),
        }

    h1.remove()
    h2.remove()
    h3.remove()

    print(f"Block 0 (Source vs Target):")
    print(f"  Shape: {act_source['block0'].shape}")
    print(f"  L2 distance: {torch.dist(act_source['block0'], act_target['block0']).item():.8f}")
    print(f"  Source std: {act_source['block0'].std().item():.6f}")

    print(f"\nBlock 5 (Source vs Target):")
    print(f"  Shape: {act_source['block5'].shape}")
    print(f"  L2 distance: {torch.dist(act_source['block5'], act_target['block5']).item():.8f}")
    print(f"  Source std: {act_source['block5'].std().item():.6f}")

    print(f"\nPooling output (Source vs Target):")
    print(f"  Shape: {act_source['pool'].shape}")
    print(f"  L2 distance: {torch.dist(act_source['pool'], act_target['pool']).item():.8f}")
    print(f"  Source std: {act_source['pool'].std().item():.6f}")

    print("\n=== Diagnosis ===")
    print("もし全てのテストでCosSim≈1.0 → Speaker Encoderが入力を無視している")
    print("もし中間層でL2距離>0だが最終出力で≈0 → Pooling層で情報が消失")
    print("もし中間層からL2距離≈0 → 初期層で既に情報が失われている")


if __name__ == "__main__":
    main()
