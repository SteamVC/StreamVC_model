#!/usr/bin/env python
"""StreamVC推論スクリプト - チェックポイントから音声を生成"""

import argparse
import time
from pathlib import Path

import torch
import soundfile as sf

from streamvc import StreamVCPipeline, load_config


def load_checkpoint(checkpoint_path: Path, config, device):
    """チェックポイントを読み込む"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get num_hubert_labels from checkpoint or infer from model
    if "num_hubert_labels" in checkpoint:
        num_hubert_labels = checkpoint["num_hubert_labels"]
    else:
        # Infer from content_head weight shape
        if "model" in checkpoint and "content_head.linear.weight" in checkpoint["model"]:
            num_hubert_labels = checkpoint["model"]["content_head.linear.weight"].shape[0]
            print(f"  推論: num_hubert_labels={num_hubert_labels} (content_head.linear.weightのshapeから)")
        else:
            # Default from HuBERT-Base
            num_hubert_labels = 504
            print(f"  デフォルト: num_hubert_labels={num_hubert_labels}")

    pipeline = StreamVCPipeline(config, num_hubert_labels=num_hubert_labels)
    pipeline.load_state_dict(checkpoint["model"])

    # Restore active quantizers for progressive RVQ
    if "num_active_quantizers" in checkpoint:
        num_active = checkpoint["num_active_quantizers"]
        pipeline.decoder.rvq.set_num_active_quantizers(num_active)
        print(f"  Progressive RVQ: {num_active} / {pipeline.decoder.rvq.config.num_quantizers} quantizers active")

    pipeline.to(device)
    pipeline.eval()

    return pipeline, checkpoint.get("step", 0)


def process_audio(audio_path: Path, sample_rate: int, device):
    """音声ファイルを読み込んで前処理"""
    waveform, sr = sf.read(audio_path)

    # Convert to torch tensor
    # sf.read returns (samples,) for mono or (samples, channels) for stereo
    waveform = torch.FloatTensor(waveform)

    # Handle stereo -> mono
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=1)  # Average channels

    # Now waveform is (samples,)
    # Add channel dimension: (samples,) -> (1, samples)
    waveform = waveform.unsqueeze(0)

    # Resample if needed
    if sr != sample_rate:
        import torchaudio.transforms as T
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Move to device (waveform is now (1, samples))
    waveform = waveform.to(device)

    return waveform


def infer_single(
    pipeline: StreamVCPipeline,
    source_path: Path,
    target_ref_path: Path,
    output_path: Path,
    sample_rate: int,
    device: torch.device,
    measure_rtf: bool = True,
):
    """単一音声の変換を実行"""

    # Load audio
    print(f"  変換元を読み込み中: {source_path.name}")
    source_audio = process_audio(source_path, sample_rate, device)

    print(f"  ターゲット参照を読み込み中: {target_ref_path.name}")
    target_ref_audio = process_audio(target_ref_path, sample_rate, device)

    # Encode speaker
    print("  話者埋め込みをエンコード中...")
    pipeline.encode_speaker(target_ref_audio)

    # Reset pitch stats right before inference so statistics are initialized from source audio
    print("  ピッチ統計をリセット中...")
    pipeline.reset_pitch_stats()

    # Inference with timing
    print("  音声変換を実行中...")
    if measure_rtf:
        start_time = time.time()

    with torch.no_grad():
        outputs = pipeline(source_audio, mode="infer")

    if measure_rtf:
        elapsed = time.time() - start_time
        audio_duration = source_audio.shape[-1] / sample_rate
        rtf = elapsed / audio_duration
    else:
        rtf = None

    # Save output
    # generated is (batch, samples) = (1, N)
    generated = outputs["audio"].cpu().squeeze(0).numpy()  # (1, N) -> (N,)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, generated, sample_rate)
    print(f"  ✓ 保存完了: {output_path}")

    return rtf, outputs


def main():
    parser = argparse.ArgumentParser(description="StreamVC推論")
    parser.add_argument("--checkpoint", type=Path, required=True, help="チェックポイントファイル (.pt)")
    parser.add_argument("--config", type=Path, required=True, help="設定ファイル (.yaml)")
    parser.add_argument("--source", type=Path, required=True, help="変換元音声")
    parser.add_argument("--target-ref", type=Path, required=True, help="ターゲット話者の参照音声")
    parser.add_argument("--output", type=Path, required=True, help="出力音声ファイル")
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"], help="デバイス")
    parser.add_argument("--no-rtf", action="store_true", help="RTF測定を無効化")

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    config = load_config(args.config)
    sample_rate = config.data.sample_rate

    print("=" * 80)
    print("StreamVC 推論")
    print("=" * 80)
    print(f"チェックポイント: {args.checkpoint}")
    print(f"設定ファイル: {args.config}")
    print(f"デバイス: {device}")
    print()

    # Load model
    print("モデルを読み込み中...")
    pipeline, step = load_checkpoint(args.checkpoint, config, device)
    print(f"✓ ステップ {step} のモデルを読み込みました")
    print()

    # Inference
    print("音声変換を開始...")
    print("-" * 80)

    rtf, outputs = infer_single(
        pipeline=pipeline,
        source_path=args.source,
        target_ref_path=args.target_ref,
        output_path=args.output,
        sample_rate=sample_rate,
        device=device,
        measure_rtf=not args.no_rtf,
    )

    print("-" * 80)
    print("\n✓✓ 変換完了")

    if rtf is not None:
        print(f"\nRTF (Real-Time Factor): {rtf:.3f}x")
        if rtf < 1.0:
            print(f"  → リアルタイムより {1/rtf:.1f}x 高速")
        else:
            print(f"  → リアルタイムより {rtf:.1f}x 遅い")

    print(f"\nRVQ Loss: {outputs['rvq_loss'].item():.6f}")
    print(f"Codes shape: {outputs['codes'].shape}")


if __name__ == "__main__":
    main()
