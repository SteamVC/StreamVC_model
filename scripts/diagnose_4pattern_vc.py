#!/usr/bin/env python3
"""セット②：4パターンVC評価（A→A, B→B, A→B, B→A）"""

import argparse
from pathlib import Path
import random

import torch
import numpy as np
from tqdm import tqdm

from streamvc import StreamVCPipeline, load_config


def evaluate_4pattern_vc(
    checkpoint_path: Path,
    config_path: Path,
    cache_dir: Path,
    num_pairs: int = 20,
    device: str = "cpu",
):
    """4パターンのVC評価"""

    print("\n" + "=" * 70)
    print("セット②：4パターンVC評価")
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
    print()

    # ================== Pattern 1: A→A (self-reconstruction) ==================
    print("=" * 70)
    print("Pattern 1: A→A (self-reconstruction)")
    print("=" * 70)

    aa_conv_vs_src = []
    aa_conv_vs_tgt = []

    for _ in tqdm(range(num_pairs), desc="A→A"):
        spk = random.choice(speakers)
        src_file = random.choice(speaker_files[spk])

        cache = torch.load(src_file)
        src_audio = cache["source_audio"].unsqueeze(0).to(device)

        with torch.no_grad():
            z_src = pipeline.encode_speaker(src_audio)
            conv_audio = pipeline(src_audio, z_src)["audio"]

            z_conv = pipeline.encode_speaker(conv_audio)

        # A→A なので、tgt = src
        cos_conv_src = torch.nn.functional.cosine_similarity(z_conv, z_src, dim=1).item()
        cos_conv_tgt = cos_conv_src  # 同一

        aa_conv_vs_src.append(cos_conv_src)
        aa_conv_vs_tgt.append(cos_conv_tgt)

    print(f"  Conv vs Src: {np.mean(aa_conv_vs_src):.4f} ± {np.std(aa_conv_vs_src):.4f}")
    print(f"  Conv vs Tgt: {np.mean(aa_conv_vs_tgt):.4f} ± {np.std(aa_conv_vs_tgt):.4f}")
    print()

    # ================== Pattern 2: B→B (self-reconstruction) ==================
    print("=" * 70)
    print("Pattern 2: B→B (self-reconstruction)")
    print("=" * 70)

    bb_conv_vs_src = []
    bb_conv_vs_tgt = []

    for _ in tqdm(range(num_pairs), desc="B→B"):
        spk = random.choice(speakers)
        src_file = random.choice(speaker_files[spk])

        cache = torch.load(src_file)
        src_audio = cache["source_audio"].unsqueeze(0).to(device)

        with torch.no_grad():
            z_src = pipeline.encode_speaker(src_audio)
            conv_audio = pipeline(src_audio, z_src)["audio"]

            z_conv = pipeline.encode_speaker(conv_audio)

        cos_conv_src = torch.nn.functional.cosine_similarity(z_conv, z_src, dim=1).item()
        cos_conv_tgt = cos_conv_src

        bb_conv_vs_src.append(cos_conv_src)
        bb_conv_vs_tgt.append(cos_conv_tgt)

    print(f"  Conv vs Src: {np.mean(bb_conv_vs_src):.4f} ± {np.std(bb_conv_vs_src):.4f}")
    print(f"  Conv vs Tgt: {np.mean(bb_conv_vs_tgt):.4f} ± {np.std(bb_conv_vs_tgt):.4f}")
    print()

    # ================== Pattern 3: A→B (cross-speaker) ==================
    print("=" * 70)
    print("Pattern 3: A→B (cross-speaker conversion)")
    print("=" * 70)

    ab_conv_vs_src = []
    ab_conv_vs_tgt = []

    for _ in tqdm(range(num_pairs), desc="A→B"):
        spk_a, spk_b = random.sample(speakers, 2)

        src_file = random.choice(speaker_files[spk_a])
        tgt_file = random.choice(speaker_files[spk_b])

        cache_src = torch.load(src_file)
        cache_tgt = torch.load(tgt_file)

        src_audio = cache_src["source_audio"].unsqueeze(0).to(device)
        tgt_audio = cache_tgt["source_audio"].unsqueeze(0).to(device)

        with torch.no_grad():
            z_src = pipeline.encode_speaker(src_audio)
            z_tgt = pipeline.encode_speaker(tgt_audio)

            # A→B: src content + tgt speaker
            conv_audio = pipeline(src_audio, z_tgt)["audio"]
            z_conv = pipeline.encode_speaker(conv_audio)

        cos_conv_src = torch.nn.functional.cosine_similarity(z_conv, z_src, dim=1).item()
        cos_conv_tgt = torch.nn.functional.cosine_similarity(z_conv, z_tgt, dim=1).item()

        ab_conv_vs_src.append(cos_conv_src)
        ab_conv_vs_tgt.append(cos_conv_tgt)

    print(f"  Conv vs Src: {np.mean(ab_conv_vs_src):.4f} ± {np.std(ab_conv_vs_src):.4f}")
    print(f"  Conv vs Tgt: {np.mean(ab_conv_vs_tgt):.4f} ± {np.std(ab_conv_vs_tgt):.4f}")
    print()

    # ================== Pattern 4: B→A (cross-speaker) ==================
    print("=" * 70)
    print("Pattern 4: B→A (cross-speaker conversion)")
    print("=" * 70)

    ba_conv_vs_src = []
    ba_conv_vs_tgt = []

    for _ in tqdm(range(num_pairs), desc="B→A"):
        spk_a, spk_b = random.sample(speakers, 2)

        src_file = random.choice(speaker_files[spk_b])
        tgt_file = random.choice(speaker_files[spk_a])

        cache_src = torch.load(src_file)
        cache_tgt = torch.load(tgt_file)

        src_audio = cache_src["source_audio"].unsqueeze(0).to(device)
        tgt_audio = cache_tgt["source_audio"].unsqueeze(0).to(device)

        with torch.no_grad():
            z_src = pipeline.encode_speaker(src_audio)
            z_tgt = pipeline.encode_speaker(tgt_audio)

            # B→A: src content + tgt speaker
            conv_audio = pipeline(src_audio, z_tgt)["audio"]
            z_conv = pipeline.encode_speaker(conv_audio)

        cos_conv_src = torch.nn.functional.cosine_similarity(z_conv, z_src, dim=1).item()
        cos_conv_tgt = torch.nn.functional.cosine_similarity(z_conv, z_tgt, dim=1).item()

        ba_conv_vs_src.append(cos_conv_src)
        ba_conv_vs_tgt.append(cos_conv_tgt)

    print(f"  Conv vs Src: {np.mean(ba_conv_vs_src):.4f} ± {np.std(ba_conv_vs_src):.4f}")
    print(f"  Conv vs Tgt: {np.mean(ba_conv_vs_tgt):.4f} ± {np.std(ba_conv_vs_tgt):.4f}")
    print()

    # ================== Summary ==================
    print("=" * 70)
    print("まとめ")
    print("=" * 70)
    print()

    print("Self-reconstruction (A→A, B→B):")
    print(f"  Conv vs Src/Tgt: {np.mean(aa_conv_vs_src + bb_conv_vs_src):.4f}")
    print()

    print("Cross-speaker (A→B, B→A):")
    print(f"  Conv vs Src: {np.mean(ab_conv_vs_src + ba_conv_vs_src):.4f}")
    print(f"  Conv vs Tgt: {np.mean(ab_conv_vs_tgt + ba_conv_vs_tgt):.4f}")
    print()

    # 期待される挙動
    print("期待される挙動:")
    print("  - Self-recon: Conv vs Src/Tgt > 0.5 (同一話者として認識)")
    print("  - Cross-speaker: Conv vs Tgt > 0.5 (target話者に近い)")
    print("  - Cross-speaker: Conv vs Src < 0.3 (source話者から離れる)")
    print()

    # 判定
    self_mean = np.mean(aa_conv_vs_src + bb_conv_vs_src)
    cross_tgt_mean = np.mean(ab_conv_vs_tgt + ba_conv_vs_tgt)
    cross_src_mean = np.mean(ab_conv_vs_src + ba_conv_vs_src)

    print("=" * 70)
    print("診断結果")
    print("=" * 70)

    if self_mean < 0.2:
        print("❌ Self-reconstruction失敗: VC出力がドメイン外（セット①-2と一致）")
        print("   → 音質劣化 or Decoder崩壊")
    elif self_mean < 0.5:
        print("⚠️  Self-reconstruction弱い: VC出力の音質に問題あり")
    else:
        print("✅ Self-reconstruction成功: VC出力は認識可能")

    print()

    if cross_tgt_mean < 0.2:
        print("❌ Cross-speaker失敗: targetに近づいていない")
        if cross_src_mean > 0.5:
            print("   → Decoder が z を無視して source のまま出力")
        else:
            print("   → VC出力が話者として認識不能（音質崩壊）")
    elif cross_tgt_mean < 0.5:
        print("⚠️  Cross-speaker弱い: 部分的にしか変換されていない")
        print(f"   Conv vs Src: {cross_src_mean:.4f}")
        print(f"   Conv vs Tgt: {cross_tgt_mean:.4f}")
    else:
        print("✅ Cross-speaker成功: targetに変換されている")

    print("=" * 70)

    return {
        "aa": {"conv_vs_src": aa_conv_vs_src, "conv_vs_tgt": aa_conv_vs_tgt},
        "bb": {"conv_vs_src": bb_conv_vs_src, "conv_vs_tgt": bb_conv_vs_tgt},
        "ab": {"conv_vs_src": ab_conv_vs_src, "conv_vs_tgt": ab_conv_vs_tgt},
        "ba": {"conv_vs_src": ba_conv_vs_src, "conv_vs_tgt": ba_conv_vs_tgt},
    }


def main():
    parser = argparse.ArgumentParser(description="4パターンVC評価")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/colab_gpu_training.yaml"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--num-pairs", type=int, default=20, help="各パターンのペア数")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])

    args = parser.parse_args()

    evaluate_4pattern_vc(
        args.checkpoint,
        args.config,
        args.cache_dir,
        args.num_pairs,
        args.device,
    )


if __name__ == "__main__":
    main()
