#!/usr/bin/env python3
"""勾配フロー確認スクリプト"""

import torch
from pathlib import Path

from streamvc import StreamVCPipeline, load_config


def main():
    config = load_config(Path("configs/colab_gpu_training.yaml"))

    # モデル初期化
    pipeline = StreamVCPipeline(config, num_hubert_labels=100)
    pipeline.train()

    # ダミーデータ作成
    batch_size = 2
    audio_length = 16000  # 1秒

    source_audio = torch.randn(batch_size, audio_length)
    target_reference = torch.randn(batch_size, audio_length)

    # Forward
    outputs = pipeline(
        source_audio=source_audio,
        target_reference=target_reference,
        mode="train"
    )

    # 簡単な損失計算
    loss = outputs["audio"].abs().mean() + outputs["rvq_loss"]

    # Backward
    loss.backward()

    # Speaker Encoderの勾配ノルムを確認
    print("\n=== Speaker Encoder Gradients ===")
    total_norm = 0.0
    for name, param in pipeline.speaker_encoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_norm += grad_norm ** 2
            if grad_norm > 1e-8:
                print(f"{name:50s} grad_norm={grad_norm:.6f}")
            else:
                print(f"{name:50s} grad_norm={grad_norm:.2e} (ほぼゼロ)")
        else:
            print(f"{name:50s} NO GRADIENT")

    total_norm = total_norm ** 0.5
    print(f"\nSpeaker Encoder total gradient norm: {total_norm:.6f}")

    # Content Encoderの勾配（detachされているはず）
    print("\n=== Content Encoder Gradients (should be zero due to detach) ===")
    for name, param in pipeline.content_encoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-8:
                print(f"{name:50s} grad_norm={grad_norm:.6f} ⚠WARNING")
            else:
                print(f"{name:50s} grad_norm={grad_norm:.2e} (正常)")
        else:
            print(f"{name:50s} NO GRADIENT (正常)")

    # Decoderの勾配
    print("\n=== Decoder Gradients ===")
    decoder_norm = 0.0
    for name, param in pipeline.decoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            decoder_norm += grad_norm ** 2
    decoder_norm = decoder_norm ** 0.5
    print(f"Decoder total gradient norm: {decoder_norm:.6f}")

    print("\n=== Summary ===")
    if total_norm < 1e-6:
        print("⚠ Speaker Encoder勾配がほぼゼロ → 実装に問題あり")
    elif total_norm < 0.01:
        print("⚠ Speaker Encoder勾配が小さすぎる → 学習困難")
    else:
        print("✓ Speaker Encoder勾配は正常に流れている")


if __name__ == "__main__":
    main()
