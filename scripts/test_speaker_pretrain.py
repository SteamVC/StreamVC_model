#!/usr/bin/env python3
"""Quick test for Phase 1 Speaker Pretrain implementation."""

import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from streamvc.speaker_classifier import SpeakerClassifier
from streamvc.data.speaker_dataset import SpeakerDataset

def test_speaker_classifier():
    print("Testing SpeakerClassifier...")

    # Create model
    model = SpeakerClassifier(
        num_speakers=247,  # LibriTTS train-clean-100
        latent_dim=128,
    )

    # Test forward
    batch_size = 4
    audio_length = 16000 * 3  # 3 seconds
    audio = torch.randn(batch_size, audio_length)

    embeddings, logits = model(audio)

    print(f"  Audio shape: {audio.shape}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Logits shape: {logits.shape}")

    assert embeddings.shape == (batch_size, 128), f"Expected (4, 128), got {embeddings.shape}"
    assert logits.shape == (batch_size, 247), f"Expected (4, 247), got {logits.shape}"

    print("  ✓ SpeakerClassifier forward pass OK")

    # Test extract_embedding
    emb_only = model.extract_embedding(audio)
    assert emb_only.shape == embeddings.shape
    assert torch.allclose(emb_only, embeddings)
    print("  ✓ extract_embedding OK")

    # Test parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,}")

    print("\n✅ SpeakerClassifier tests passed!\n")

def test_speaker_dataset():
    print("Testing SpeakerDataset (cache-based)...")

    cache_dir = Path("data/cache")
    if not cache_dir.exists():
        print("  ⚠️  Cache not found, skipping dataset test")
        return

    try:
        dataset = SpeakerDataset(
            cache_dir=cache_dir,
            dataset_name="libri_tts",
            split="train",
        )

        print(f"  Dataset size: {len(dataset)}")
        print(f"  Num speakers: {dataset.num_speakers}")

        # Test single sample
        sample = dataset[0]
        print(f"  Audio shape: {sample['audio'].shape}")
        print(f"  Speaker label: {sample['speaker_label']}")

        assert sample['audio'].shape[0] == 16000 * 3, "Expected 3 sec audio"
        assert 0 <= sample['speaker_label'] < dataset.num_speakers

        print("  ✓ SpeakerDataset OK")

    except FileNotFoundError as e:
        print(f"  ⚠️  {e}")
        print("  Cache not ready yet, skipping dataset test")

if __name__ == "__main__":
    test_speaker_classifier()
    test_speaker_dataset()
    print("\n✅ All tests passed!\n")
