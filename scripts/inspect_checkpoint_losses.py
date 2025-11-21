#!/usr/bin/env python3
"""Checkpointから損失履歴を抽出"""

import argparse
from pathlib import Path

import torch


def inspect_checkpoint(checkpoint_path: Path):
    """Checkpoint内の損失情報を表示"""

    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path.name}")
    print("=" * 70)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    print(f"\nStep: {checkpoint.get('step', 'unknown')}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    print()

    # Check available keys
    print("Available keys:")
    for key in sorted(checkpoint.keys()):
        if key != "model":
            print(f"  - {key}")
    print()

    # Check optimizer state
    if "optimizer" in checkpoint:
        print("Optimizer state found")
    if "discriminator_optimizer" in checkpoint:
        print("Discriminator optimizer state found")
    print()

    # Model architecture info
    if "model" in checkpoint:
        model_keys = list(checkpoint["model"].keys())
        print(f"Model parameters: {len(model_keys)}")
        print("\nKey components:")

        components = set()
        for key in model_keys:
            component = key.split(".")[0]
            components.add(component)

        for comp in sorted(components):
            comp_keys = [k for k in model_keys if k.startswith(comp + ".")]
            print(f"  - {comp}: {len(comp_keys)} params")

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Checkpoint情報確認")
    parser.add_argument("--checkpoint", type=Path, required=True)

    args = parser.parse_args()

    inspect_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
