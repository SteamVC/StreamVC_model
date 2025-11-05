#!/usr/bin/env python
"""簡易学習テスト - ログ出力確認用"""
import sys
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)  # Line buffering

print("=== Starting Simple Training Test ===", flush=True)

from pathlib import Path
import torch
from streamvc import StreamVCTrainer, load_config
from streamvc.data import build_dataloader

print("Loading config...", flush=True)
config = load_config(Path("configs/test_libritts.yaml"))

print("Building dataloader...", flush=True)
train_loader = build_dataloader(config, split="train")

print("Initializing trainer...", flush=True)
trainer = StreamVCTrainer(config)
trainer.device = torch.device("cpu")
trainer.pipeline.to(trainer.device)

print("Starting training loop...", flush=True)
for i, batch in enumerate(train_loader):
    print(f"Step {i+1}/10: Processing batch...", flush=True)

    trainer.optimizer.zero_grad()
    metrics = trainer.train_step(batch)
    metrics["loss"].backward()
    trainer.optimizer.step()

    loss = metrics["loss"].item()
    print(f"  Loss: {loss:.4f}", flush=True)

    if i >= 9:  # 10 steps only
        break

print("=== Training Test Complete ===", flush=True)
