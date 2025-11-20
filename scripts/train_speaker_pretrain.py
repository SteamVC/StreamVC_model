#!/usr/bin/env python3
"""Phase 1: Speaker Encoder Pretrain with speaker classification.

Train Speaker Encoder to distinguish speaker IDs using CE loss.
The trained encoder will be used (frozen) in Phase 2 VC training.
"""

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from streamvc.data.speaker_dataset import SpeakerDataset, collate_speaker_batch
from streamvc.speaker_classifier import SpeakerClassifier


def train_epoch(model, loader, optimizer, device, epoch, writer, step):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        audio = batch["audio"].to(device)
        labels = batch["speaker_label"].to(device)

        optimizer.zero_grad()

        # Forward
        embeddings, logits = model(audio)

        # CE loss
        loss = nn.functional.cross_entropy(logits, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        # Log
        if step % 50 == 0:
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/accuracy", (preds == labels).float().mean().item(), step)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100*total_correct/total_samples:.2f}%"})
        step += 1

    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc, step


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(loader, desc="Validation"):
        audio = batch["audio"].to(device)
        labels = batch["speaker_label"].to(device)

        embeddings, logits = model(audio)
        loss = nn.functional.cross_entropy(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Speaker Encoder Pretrain")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Path to cache directory")
    parser.add_argument("--dataset-name", type=str, default="libri_tts", help="Dataset name in cache")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/speaker_pretrain"), help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=128, help="Speaker embedding dimension")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    args = parser.parse_args()

    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(str(args.output_dir / "logs"))

    # Load dataset
    train_dataset = SpeakerDataset(
        cache_dir=args.cache_dir,
        dataset_name=args.dataset_name,
        split="train",
    )
    valid_dataset = SpeakerDataset(
        cache_dir=args.cache_dir,
        dataset_name=args.dataset_name,
        split="valid",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_speaker_batch,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_speaker_batch,
        pin_memory=True,
    )

    # Model
    model = SpeakerClassifier(
        num_speakers=train_dataset.num_speakers,
        latent_dim=args.latent_dim,
    ).to(args.device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\n{'='*70}")
    print(f"Phase 1: Speaker Encoder Pretrain")
    print(f"{'='*70}")
    print(f"Speakers: {train_dataset.num_speakers}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Device: {args.device}")
    print(f"{'='*70}\n")

    best_val_acc = 0.0
    step = 0

    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_loss, train_acc, step = train_epoch(model, train_loader, optimizer, args.device, epoch, writer, step)

        # Validate
        val_loss, val_acc = validate(model, valid_loader, args.device)

        # Log epoch metrics
        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/train_accuracy", train_acc, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)
        writer.add_scalar("epoch/val_accuracy", val_acc, epoch)

        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {100*train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {100*val_acc:.2f}%")

        # Save checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_acc": val_acc,
                "num_speakers": train_dataset.num_speakers,
                "latent_dim": args.latent_dim,
            }, ckpt_path)
            print(f"  ✓ Best model saved (val_acc={100*val_acc:.2f}%)")

        # Save periodic checkpoint
        if epoch % 10 == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_acc": val_acc,
                "num_speakers": train_dataset.num_speakers,
                "latent_dim": args.latent_dim,
            }, ckpt_path)

    writer.close()
    print(f"\n{'='*70}")
    print(f"Training complete! Best val accuracy: {100*best_val_acc:.2f}%")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
