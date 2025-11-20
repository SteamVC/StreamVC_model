#!/usr/bin/env python3
"""Speaker Encoderの重みが実際に更新されているか確認"""

import torch
from pathlib import Path


def main():
    checkpoints = [
        "runs/streamvc_phase1_ema/checkpoints/step_5000.pt",
        "runs/streamvc_phase1_ema/checkpoints/step_20000.pt",
    ]

    # Load checkpoints
    ckpt_5k = torch.load(checkpoints[0], map_location="cpu")
    ckpt_20k = torch.load(checkpoints[1], map_location="cpu")

    # Extract speaker_encoder keys
    spk_keys = [k for k in ckpt_5k["model"].keys() if "speaker_encoder" in k]

    print("=== Speaker Encoder Weight Changes (5k → 20k) ===\n")
    print(f"{'Parameter':60s} {'Δ (abs mean)':>15s} {'Relative %':>12s}")
    print("-" * 90)

    for key in sorted(spk_keys):
        w1 = ckpt_5k["model"][key]
        w2 = ckpt_20k["model"][key]

        diff_abs_mean = (w2 - w1).abs().mean().item()
        w1_abs_mean = w1.abs().mean().item()

        if w1_abs_mean > 1e-8:
            relative_change = (diff_abs_mean / w1_abs_mean) * 100
        else:
            relative_change = 0.0

        print(f"{key:60s} {diff_abs_mean:>15.8f} {relative_change:>11.4f}%")

    print("\n=== Diagnosis ===")
    print("もし Δ がすべて 1e-6 以下 → Speaker Encoder の重みがほぼ更新されていない")
    print("もし Relative % が 0.01% 未満 → 更新が極端に遅い（learning rate問題の可能性）")
    print("もし Relative % が 1-10% 程度 → 正常に更新されている")


if __name__ == "__main__":
    main()
