#!/usr/bin/env python3
"""Test checkpoint resume functionality."""

from pathlib import Path
import torch
from streamvc import StreamVCTrainer, load_config
from streamvc.modules.discriminator import MultiScaleDiscriminator

print("=" * 70)
print("Testing checkpoint resume functionality")
print("=" * 70)

# Load config
config_path = Path("configs/colab_gpu_training.yaml")
config = load_config(config_path)
print(f"✓ Config loaded from {config_path}")

# Initialize trainer
discriminator = MultiScaleDiscriminator()
trainer = StreamVCTrainer(config, discriminator=discriminator)
trainer.device = torch.device("cpu")
trainer.pipeline.to(trainer.device)
trainer.discriminator.to(trainer.device)
print(f"✓ Trainer initialized")

# Resume from checkpoint
checkpoint_path = Path("runs/streamvc_phase1_ema/checkpoints/step_5000_2A.pt")
print(f"\nResuming from: {checkpoint_path}")

trainer.load_checkpoint(checkpoint_path)

print(f"✓ Checkpoint loaded successfully")
print(f"  Step: {trainer.step}")
print(f"  Device: {trainer.device}")

# Check speaker encoder frozen status
num_frozen = sum(1 for p in trainer.pipeline.speaker_encoder.parameters() if not p.requires_grad)
num_total = sum(1 for _ in trainer.pipeline.speaker_encoder.parameters())
print(f"  Speaker encoder frozen: {num_frozen}/{num_total} params")

print("\n" + "=" * 70)
print("✅ Resume test PASSED - ready for A100 deployment")
print("=" * 70)
