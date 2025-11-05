#!/usr/bin/env python
"""デバッグ用: フォワードパスの各ステップの時間を計測"""
import time
import torch
from streamvc import StreamVCTrainer, load_config
from streamvc.data import build_dataloader
from pathlib import Path

config = load_config(Path("configs/test_libritts.yaml"))
train_loader = build_dataloader(config, split="train", batch_size=4)
trainer = StreamVCTrainer(config)
trainer.device = torch.device("cpu")
trainer.pipeline.to(trainer.device)

batch = next(iter(train_loader))
source = batch["source_audio"].to(trainer.device)
target_reference = batch["target_reference"].to(trainer.device)

print(f"Source shape: {source.shape}")
print(f"Reference shape: {target_reference.shape}")

with torch.no_grad():
    # Content encoder
    t0 = time.time()
    units = trainer.pipeline.content_encoder(source)
    t1 = time.time()
    print(f"Content encoder: {t1-t0:.2f}s, output shape: {units.shape}")

    # Pitch extractor
    t0 = time.time()
    pitch = trainer.pipeline.pitch_extractor(source, mode="train")
    t1 = time.time()
    print(f"Pitch extractor: {t1-t0:.2f}s")
    print(f"  f0 shape: {pitch.f0_hz.shape}")
    print(f"  voiced shape: {pitch.voiced_prob.shape}")
    print(f"  energy shape: {pitch.energy.shape}")

    # Speaker encoder
    t0 = time.time()
    speaker_embedding = trainer.pipeline.speaker_encoder(target_reference)
    t1 = time.time()
    print(f"Speaker encoder: {t1-t0:.2f}s, output shape: {speaker_embedding.shape}")

    # Build side features
    t0 = time.time()
    side = trainer.pipeline._build_side_features(pitch, units.shape[1])
    t1 = time.time()
    print(f"Build side features: {t1-t0:.2f}s, output shape: {side.shape}")

    # Decoder
    t0 = time.time()
    audio, rvq_loss, codes = trainer.pipeline.decoder(units, side, speaker_embedding)
    t1 = time.time()
    print(f"Decoder: {t1-t0:.2f}s, output shape: {audio.shape}")

print("Complete!")
