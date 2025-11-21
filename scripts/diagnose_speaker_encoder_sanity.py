#!/usr/bin/env python3
"""Speaker Encoder の健全性チェック（生音声 + VC出力）"""

import argparse
from pathlib import Path
import random

import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

from streamvc import StreamVCPipeline, load_config


def load_audio(path: Path) -> torch.Tensor:
    """Load audio and convert to tensor"""
    audio_np, sr = sf.read(path)
    audio = torch.from_numpy(audio_np).float()

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    elif audio.ndim == 2:
        audio = audio.T

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != 16000:
        import torchaudio.transforms as T
        audio = T.Resample(sr, 16000)(audio)

    return audio


def test_raw_audio_discrimination(
    checkpoint_path: Path,
    config_path: Path,
    cache_dir: Path,
    num_pairs: int = 50,
    device: str = "cpu",
):
    """セット①-1: 生音声での same/diff speaker 分布"""

    print("\n" + "=" * 70)
    print("セット①-1: Speaker Encoder - 生音声での健全性チェック")
    print("=" * 70)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = load_config(config_path)
    num_hubert_labels = checkpoint["model"]["content_head.linear.weight"].shape[0]
    pipeline = StreamVCPipeline(config, num_hubert_labels)
    pipeline.load_state_dict(checkpoint["model"], strict=False)
    pipeline.eval()
    pipeline.to(device)

    # Collect audio files
    cache_path = cache_dir / "libri_tts" / "valid"
    cache_files = list(cache_path.glob("*.pt"))

    if len(cache_files) < num_pairs * 2:
        print(f"⚠️  Not enough cache files: {len(cache_files)}")
        num_pairs = len(cache_files) // 2

    # Group by speaker
    speaker_files = {}
    for f in cache_files:
        speaker_id = f.stem.split("_")[0]
        speaker_files.setdefault(speaker_id, []).append(f)

    # Filter speakers with at least 2 files
    speaker_files = {k: v for k, v in speaker_files.items() if len(v) >= 2}
    speakers = list(speaker_files.keys())

    print(f"Speakers: {len(speakers)}")
    print(f"Testing {num_pairs} pairs...")

    # Test same-speaker pairs
    same_sims = []
    same_dists = []

    print("\nSame-speaker pairs:")
    for _ in tqdm(range(num_pairs)):
        spk = random.choice(speakers)
        files = random.sample(speaker_files[spk], 2)

        cache1 = torch.load(files[0])
        cache2 = torch.load(files[1])

        audio1 = cache1["source_audio"].unsqueeze(0).to(device)
        audio2 = cache2["source_audio"].unsqueeze(0).to(device)

        with torch.no_grad():
            emb1 = pipeline.encode_speaker(audio1)
            emb2 = pipeline.encode_speaker(audio2)

        cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1).item()
        l2_dist = torch.norm(emb1 - emb2).item()

        same_sims.append(cos_sim)
        same_dists.append(l2_dist)

    # Test different-speaker pairs
    diff_sims = []
    diff_dists = []

    print("\nDifferent-speaker pairs:")
    for _ in tqdm(range(num_pairs)):
        spk1, spk2 = random.sample(speakers, 2)

        file1 = random.choice(speaker_files[spk1])
        file2 = random.choice(speaker_files[spk2])

        cache1 = torch.load(file1)
        cache2 = torch.load(file2)

        audio1 = cache1["source_audio"].unsqueeze(0).to(device)
        audio2 = cache2["source_audio"].unsqueeze(0).to(device)

        with torch.no_grad():
            emb1 = pipeline.encode_speaker(audio1)
            emb2 = pipeline.encode_speaker(audio2)

        cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1).item()
        l2_dist = torch.norm(emb1 - emb2).item()

        diff_sims.append(cos_sim)
        diff_dists.append(l2_dist)

    # Report
    print("\n" + "=" * 70)
    print("結果")
    print("=" * 70)
    print(f"Same-speaker pairs (n={num_pairs}):")
    print(f"  Cos sim: {np.mean(same_sims):.4f} ± {np.std(same_sims):.4f}")
    print(f"  L2 dist: {np.mean(same_dists):.4f} ± {np.std(same_dists):.4f}")
    print()
    print(f"Different-speaker pairs (n={num_pairs}):")
    print(f"  Cos sim: {np.mean(diff_sims):.4f} ± {np.std(diff_sims):.4f}")
    print(f"  L2 dist: {np.mean(diff_dists):.4f} ± {np.std(diff_dists):.4f}")
    print()

    # Judgment
    same_mean = np.mean(same_sims)
    diff_mean = np.mean(diff_sims)
    separation = same_mean - diff_mean

    print(f"Separation: {separation:.4f}")
    if separation > 0.4:
        print("✅ Speaker Encoder は生音声で健全に動作")
    elif separation > 0.2:
        print("⚠️  分離はあるが弱い - pretrainが不十分か、話者数が少ない可能性")
    else:
        print("❌ ほとんど分離していない - Speaker Encoder に問題あり")

    print("=" * 70)

    return {
        "same_sims": same_sims,
        "same_dists": same_dists,
        "diff_sims": diff_sims,
        "diff_dists": diff_dists,
    }


def test_vc_output_discrimination(
    checkpoint_path: Path,
    config_path: Path,
    cache_dir: Path,
    num_samples: int = 20,
    device: str = "cpu",
):
    """セット①-2: VC出力（self-recon）での Speaker Encoder の反応"""

    print("\n" + "=" * 70)
    print("セット①-2: VC出力（self-recon）での Speaker Encoder 反応")
    print("=" * 70)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = load_config(config_path)
    num_hubert_labels = checkpoint["model"]["content_head.linear.weight"].shape[0]
    pipeline = StreamVCPipeline(config, num_hubert_labels)
    pipeline.load_state_dict(checkpoint["model"], strict=False)
    pipeline.eval()
    pipeline.to(device)

    # Collect samples
    cache_path = cache_dir / "libri_tts" / "valid"
    cache_files = list(cache_path.glob("*.pt"))[:num_samples]

    print(f"Testing {len(cache_files)} self-reconstruction samples...")

    sims = []
    dists = []

    for cache_file in tqdm(cache_files):
        cache = torch.load(cache_file)

        source_audio = cache["source_audio"].unsqueeze(0).to(device)

        with torch.no_grad():
            # Original embedding
            z_orig = pipeline.encode_speaker(source_audio)

            # VC: A→A (self-reconstruction)
            vc_output = pipeline(source_audio, z_orig)["audio"]

            # VC output embedding
            z_vc = pipeline.encode_speaker(vc_output)

        cos_sim = torch.nn.functional.cosine_similarity(z_orig, z_vc, dim=1).item()
        l2_dist = torch.norm(z_orig - z_vc).item()

        sims.append(cos_sim)
        dists.append(l2_dist)

    # Report
    print("\n" + "=" * 70)
    print("結果: Self-reconstruction (A→A)")
    print("=" * 70)
    print(f"Original vs VC output (n={len(cache_files)}):")
    print(f"  Cos sim: {np.mean(sims):.4f} ± {np.std(sims):.4f}")
    print(f"  L2 dist: {np.mean(dists):.4f} ± {np.std(dists):.4f}")
    print()

    mean_sim = np.mean(sims)
    if mean_sim > 0.5:
        print("✅ VC出力も話者として認識される - 音質は許容範囲")
    elif mean_sim > 0.2:
        print("⚠️  やや低い - VC出力の音質劣化が影響している可能性")
    else:
        print("❌ VC出力が別ドメインになっている - 音質/ノイズが酷すぎる")

    print("=" * 70)

    return {"sims": sims, "dists": dists}


def main():
    parser = argparse.ArgumentParser(description="Speaker Encoder健全性チェック")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/colab_gpu_training.yaml"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--num-pairs", type=int, default=50, help="セット①-1のペア数")
    parser.add_argument("--num-samples", type=int, default=20, help="セット①-2のサンプル数")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])

    args = parser.parse_args()

    # セット①-1
    test_raw_audio_discrimination(
        args.checkpoint,
        args.config,
        args.cache_dir,
        args.num_pairs,
        args.device,
    )

    # セット①-2
    test_vc_output_discrimination(
        args.checkpoint,
        args.config,
        args.cache_dir,
        args.num_samples,
        args.device,
    )


if __name__ == "__main__":
    main()
