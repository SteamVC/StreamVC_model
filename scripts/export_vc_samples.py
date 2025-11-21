#!/usr/bin/env python3
"""VC出力サンプルをエクスポートして音質確認"""

import argparse
from pathlib import Path
import random

import torch
import soundfile as sf

from streamvc import StreamVCPipeline, load_config


def export_vc_samples(
    checkpoint_path: Path,
    config_path: Path,
    cache_dir: Path,
    output_dir: Path,
    num_samples: int = 5,
    device: str = "cpu",
):
    """VC出力サンプルを保存"""

    print("\n" + "=" * 70)
    print("VC出力サンプルのエクスポート")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Group by speaker
    speaker_files = {}
    for f in cache_files:
        speaker_id = f.stem.split("_")[0]
        speaker_files.setdefault(speaker_id, []).append(f)

    # Filter speakers with at least 2 files
    speaker_files = {k: v for k, v in speaker_files.items() if len(v) >= 2}
    speakers = list(speaker_files.keys())

    print(f"Speakers: {len(speakers)}")
    print(f"Generating {num_samples} samples...")
    print()

    # ================== Self-reconstruction samples ==================
    print("=" * 70)
    print("Self-reconstruction (A→A)")
    print("=" * 70)

    for i in range(num_samples):
        spk = random.choice(speakers)
        src_file = random.choice(speaker_files[spk])

        cache = torch.load(src_file)
        src_audio = cache["source_audio"].unsqueeze(0).to(device)

        with torch.no_grad():
            z_src = pipeline.encode_speaker(src_audio)
            conv_audio = pipeline(src_audio, z_src)["audio"]

        # Save
        src_path = output_dir / f"self_recon_{i:02d}_source.wav"
        conv_path = output_dir / f"self_recon_{i:02d}_converted.wav"

        sf.write(src_path, src_audio.squeeze(0).cpu().numpy().T, 16000)
        sf.write(conv_path, conv_audio.squeeze(0).cpu().numpy().T, 16000)

        print(f"  [{i+1}/{num_samples}] Speaker {spk}")
        print(f"    Source: {src_path.name}")
        print(f"    Converted: {conv_path.name}")

    print()

    # ================== Cross-speaker samples ==================
    print("=" * 70)
    print("Cross-speaker (A→B)")
    print("=" * 70)

    for i in range(num_samples):
        spk_a, spk_b = random.sample(speakers, 2)

        src_file = random.choice(speaker_files[spk_a])
        tgt_file = random.choice(speaker_files[spk_b])

        cache_src = torch.load(src_file)
        cache_tgt = torch.load(tgt_file)

        src_audio = cache_src["source_audio"].unsqueeze(0).to(device)
        tgt_audio = cache_tgt["source_audio"].unsqueeze(0).to(device)

        with torch.no_grad():
            z_tgt = pipeline.encode_speaker(tgt_audio)
            conv_audio = pipeline(src_audio, z_tgt)["audio"]

        # Save
        src_path = output_dir / f"cross_speaker_{i:02d}_source_spk{spk_a}.wav"
        tgt_path = output_dir / f"cross_speaker_{i:02d}_target_spk{spk_b}.wav"
        conv_path = output_dir / f"cross_speaker_{i:02d}_converted.wav"

        sf.write(src_path, src_audio.squeeze(0).cpu().numpy().T, 16000)
        sf.write(tgt_path, tgt_audio.squeeze(0).cpu().numpy().T, 16000)
        sf.write(conv_path, conv_audio.squeeze(0).cpu().numpy().T, 16000)

        print(f"  [{i+1}/{num_samples}] {spk_a} → {spk_b}")
        print(f"    Source: {src_path.name}")
        print(f"    Target: {tgt_path.name}")
        print(f"    Converted: {conv_path.name}")

    print()
    print("=" * 70)
    print(f"✓ Saved {num_samples * 2} VC samples to {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="VC出力サンプルエクスポート")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/colab_gpu_training.yaml"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vc_diagnosis"))
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])

    args = parser.parse_args()

    export_vc_samples(
        args.checkpoint,
        args.config,
        args.cache_dir,
        args.output_dir,
        args.num_samples,
        args.device,
    )


if __name__ == "__main__":
    main()
