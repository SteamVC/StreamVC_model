"""StreamVC学習スクリプト。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from streamvc import StreamVCTrainer, load_config
from streamvc.data import build_dataloader
from streamvc.modules.discriminator import MultiScaleDiscriminator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train StreamVC")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--device", type=str, default="cuda", help="cuda, mps, or cpu")
    parser.add_argument("--eval", action="store_true", help="build eval loader")
    parser.add_argument(
        "--gdrive-backup",
        type=Path,
        default=None,
        help="Google Drive directory for checkpoint backups (e.g., /content/drive/MyDrive/streamVC/checkpoints)",
    )
    parser.add_argument(
        "--speaker-pretrain",
        type=Path,
        default=None,
        help="Path to pretrained speaker encoder checkpoint (Phase 1)",
    )
    parser.add_argument(
        "--freeze-speaker",
        action="store_true",
        help="Freeze speaker encoder weights (recommended for Phase 2-A)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from checkpoint (e.g., runs/streamvc_phase1_ema/checkpoints/step_5000_2A.pt)",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.experiment.seed)

    train_loader = build_dataloader(config, split="train")
    eval_loader = build_dataloader(config, split="valid", shuffle=False) if args.eval else None

    # Initialize discriminator if GAN training is enabled
    discriminator = None
    if config.training.losses.get("adversarial_weight", 0.0) > 0:
        discriminator = MultiScaleDiscriminator()
        print(f"Initialized Multi-Scale Discriminator (3 scales)")

    trainer = StreamVCTrainer(config, discriminator=discriminator)
    if args.device in ("cpu", "mps"):
        trainer.device = torch.device(args.device)
        trainer.pipeline.to(trainer.device)
        if trainer.discriminator is not None:
            trainer.discriminator.to(trainer.device)

    # Resume from checkpoint if specified
    if args.resume is not None:
        print(f"\n{'='*70}")
        print(f"Resuming training from checkpoint:")
        print(f"  {args.resume}")
        trainer.load_checkpoint(args.resume)
        print(f"✓ Resumed from step {trainer.step}")
        print(f"{'='*70}\n")

    # Load pretrained speaker encoder if specified (Phase 2)
    # NOTE: If both --resume and --speaker-pretrain are used,
    # speaker-pretrain will overwrite the encoder from the resumed checkpoint
    if args.speaker_pretrain is not None:
        print(f"\n{'='*70}")
        print(f"Loading pretrained speaker encoder from:")
        print(f"  {args.speaker_pretrain}")

        ckpt = torch.load(args.speaker_pretrain, map_location="cpu")

        # Extract encoder weights from SpeakerClassifier checkpoint
        encoder_state = {}
        for key, value in ckpt["model"].items():
            if key.startswith("encoder."):
                # Remove "encoder." prefix
                new_key = key.replace("encoder.", "")
                encoder_state[new_key] = value

        # Load into pipeline's speaker encoder
        trainer.pipeline.speaker_encoder.load_state_dict(encoder_state)

        print(f"✓ Loaded speaker encoder")
        print(f"  Pretrained on: {ckpt['num_speakers']} speakers")
        print(f"  Val accuracy: {ckpt['val_acc']:.2%}")
        print(f"  Latent dim: {ckpt['latent_dim']}")

        # Freeze if requested
        if args.freeze_speaker:
            for param in trainer.pipeline.speaker_encoder.parameters():
                param.requires_grad = False
            print(f"✓ Speaker encoder FROZEN (Phase 2-A mode)")
        else:
            print(f"⚠️  Speaker encoder will be fine-tuned (Phase 2-B/C mode)")

        print(f"{'='*70}\n")

    # Enable Google Drive backup if specified
    if args.gdrive_backup is not None:
        trainer.set_gdrive_backup(args.gdrive_backup)

    trainer.fit(train_loader, eval_loader)


if __name__ == "__main__":
    main()


