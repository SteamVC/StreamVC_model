#!/usr/bin/env python3
"""話者変換の検証: Speaker Encoderで話者埋め込みの類似度を測定"""

import argparse
from pathlib import Path

import torch
import soundfile as sf

from streamvc import StreamVCPipeline, load_config


def compute_speaker_similarity(
    checkpoint_path: Path,
    config_path: Path,
    audio1_path: Path,
    audio2_path: Path,
    device: str = "cpu",
):
    """2つの音声ファイルの話者埋め込み類似度を計算"""

    print("=" * 70)
    print("話者埋め込み類似度検証")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Audio 1: {audio1_path.name}")
    print(f"Audio 2: {audio2_path.name}")
    print()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    step = checkpoint.get("step", "unknown")
    print(f"Step: {step}")

    # Load model
    config = load_config(config_path)
    num_hubert_labels = checkpoint["model"]["content_head.linear.weight"].shape[0]
    pipeline = StreamVCPipeline(config, num_hubert_labels)
    pipeline.load_state_dict(checkpoint["model"], strict=False)
    pipeline.eval()
    pipeline.to(device)

    # Load audio files using soundfile
    audio1_np, sr1 = sf.read(audio1_path)
    audio2_np, sr2 = sf.read(audio2_path)

    # Convert to torch tensor
    audio1 = torch.from_numpy(audio1_np).float()
    audio2 = torch.from_numpy(audio2_np).float()

    # Ensure shape is (channels, samples)
    if audio1.ndim == 1:
        audio1 = audio1.unsqueeze(0)
    elif audio1.ndim == 2:
        audio1 = audio1.T

    if audio2.ndim == 1:
        audio2 = audio2.unsqueeze(0)
    elif audio2.ndim == 2:
        audio2 = audio2.T

    # Resample if needed
    if sr1 != 16000:
        import torchaudio.transforms as T
        audio1 = T.Resample(sr1, 16000)(audio1)
    if sr2 != 16000:
        import torchaudio.transforms as T
        audio2 = T.Resample(sr2, 16000)(audio2)

    # Convert to mono if stereo
    if audio1.shape[0] > 1:
        audio1 = audio1.mean(dim=0, keepdim=True)
    if audio2.shape[0] > 1:
        audio2 = audio2.mean(dim=0, keepdim=True)

    print(f"\nAudio 1 shape: {audio1.shape}")
    print(f"Audio 2 shape: {audio2.shape}")

    # Extract speaker embeddings
    with torch.no_grad():
        audio1 = audio1.to(device)
        audio2 = audio2.to(device)

        emb1 = pipeline.encode_speaker(audio1)  # (1, latent_dim)
        emb2 = pipeline.encode_speaker(audio2)  # (1, latent_dim)

    # Compute cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1).item()

    # Compute L2 distance
    l2_dist = torch.norm(emb1 - emb2).item()

    print("\n" + "=" * 70)
    print("結果")
    print("=" * 70)
    print(f"Cosine Similarity: {cos_sim:.6f}")
    print(f"L2 Distance: {l2_dist:.6f}")
    print()

    # Interpretation
    if cos_sim > 0.8:
        print("✅ 非常に類似 - 同じ話者と判定される可能性が高い")
    elif cos_sim > 0.5:
        print("⚠️  やや類似 - 話者変換が部分的に成功")
    elif cos_sim > 0.2:
        print("⚠️  低い類似度 - 話者変換の効果は限定的")
    else:
        print("❌ ほとんど類似していない - 異なる話者")

    print("=" * 70)

    return cos_sim, l2_dist


def main():
    parser = argparse.ArgumentParser(description="話者変換の検証")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint file path",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/colab_gpu_training.yaml"),
        help="Config file path",
    )
    parser.add_argument(
        "--audio1",
        type=Path,
        required=True,
        help="First audio file (e.g., converted audio)",
    )
    parser.add_argument(
        "--audio2",
        type=Path,
        required=True,
        help="Second audio file (e.g., target reference)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to use",
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    if not args.audio1.exists():
        print(f"❌ Audio 1 not found: {args.audio1}")
        return

    if not args.audio2.exists():
        print(f"❌ Audio 2 not found: {args.audio2}")
        return

    compute_speaker_similarity(
        args.checkpoint,
        args.config,
        args.audio1,
        args.audio2,
        args.device,
    )


if __name__ == "__main__":
    main()
