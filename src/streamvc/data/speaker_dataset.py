"""Speaker classification dataset for Phase 1 pretrain."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SpeakerDataset(Dataset):
    """Dataset for speaker classification (Phase 1 pretrain).

    Uses cached features from StreamVC preprocessing.
    Returns (audio, speaker_label) pairs.
    """

    def __init__(
        self,
        cache_dir: Path,
        dataset_name: str = "libri_tts",
        sample_rate: int = 16000,
        sample_length_sec: float = 3.0,  # Longer than VC (1.28s) for better speaker ID
        split: str = "train",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.dataset_name = dataset_name
        self.sample_rate = sample_rate
        self.sample_length = int(sample_rate * sample_length_sec)
        self.split = split

        # Load cached files
        cache_path = self.cache_dir / dataset_name / split
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache directory not found: {cache_path}")

        self.cache_files = sorted(cache_path.glob("*.pt"))
        if not self.cache_files:
            raise RuntimeError(f"No cache files found in {cache_path}")

        # Extract speaker IDs from filenames
        # Format: {speaker_id}_{speaker_id}_{book_id}_{...}.pt
        speaker_set = set()
        self.file_to_speaker = {}

        for cache_file in self.cache_files:
            # Extract speaker_id from filename (first part before first underscore)
            speaker_id = cache_file.stem.split("_")[0]
            self.file_to_speaker[cache_file] = speaker_id
            speaker_set.add(speaker_id)

        # Create speaker ID mapping
        self.speaker_to_id = {}
        for spk_id, spk_str in enumerate(sorted(speaker_set)):
            self.speaker_to_id[spk_str] = spk_id

        self.num_speakers = len(self.speaker_to_id)

        print(f"SpeakerDataset ({split}): {len(self.cache_files)} samples, {self.num_speakers} speakers")

    def __len__(self) -> int:
        return len(self.cache_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cache_file = self.cache_files[idx]

        # Load cached data
        sample = torch.load(cache_file)
        wav = sample["source_audio"]  # (T,) at 16kHz, 1.28 sec = 20480 samples

        # Need 3.0 sec = 48000 samples
        # Strategy: repeat and crop
        target_length = self.sample_length

        if wav.numel() >= target_length:
            # Random crop
            start = torch.randint(0, wav.numel() - target_length + 1, (1,)).item()
            wav = wav[start : start + target_length]
        else:
            # Repeat until we have enough, then crop
            num_repeats = (target_length // wav.numel()) + 1
            wav = wav.repeat(num_repeats)
            start = torch.randint(0, wav.numel() - target_length + 1, (1,)).item()
            wav = wav[start : start + target_length]

        # Get speaker label
        speaker_str = self.file_to_speaker[cache_file]
        speaker_label = self.speaker_to_id[speaker_str]

        return {
            "audio": wav,
            "speaker_label": torch.tensor(speaker_label, dtype=torch.long),
        }


def collate_speaker_batch(batch):
    """Collate function for speaker dataset."""
    audio = torch.stack([item["audio"] for item in batch])  # (B, T)
    labels = torch.stack([item["speaker_label"] for item in batch])  # (B,)
    return {"audio": audio, "speaker_label": labels}
