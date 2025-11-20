"""Speaker classification dataset for Phase 1 pretrain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.utils.data import Dataset


class SpeakerDataset(Dataset):
    """Dataset for speaker classification (Phase 1 pretrain).

    Returns (audio, speaker_label) pairs.
    """

    def __init__(
        self,
        metadata_path: Path,
        data_root: Path,
        sample_rate: int = 16000,
        sample_length_sec: float = 3.0,  # Longer than VC (1.28s) for better speaker ID
        split: str = "train",
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.data_root = Path(data_root)
        self.sample_rate = sample_rate
        self.sample_length = int(sample_rate * sample_length_sec)
        self.split = split

        # Load metadata
        self.samples = []
        self.speaker_to_id = {}
        speaker_set = set()

        with self.metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("split") == split:
                    self.samples.append(entry)
                    speaker_set.add(entry["speaker_id"])

        # Create speaker ID mapping
        for spk_id, spk_str in enumerate(sorted(speaker_set)):
            self.speaker_to_id[spk_str] = spk_id

        self.num_speakers = len(self.speaker_to_id)

        if not self.samples:
            raise RuntimeError(f"No samples found for split={split} in {metadata_path}")

        print(f"SpeakerDataset ({split}): {len(self.samples)} samples, {self.num_speakers} speakers")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.samples[idx]

        # Load audio
        audio_path = self.data_root / entry["source"]
        wav, sr = sf.read(str(audio_path))
        wav = torch.from_numpy(wav).float()

        # Mono
        if len(wav.shape) > 1:
            wav = wav.mean(dim=1)

        # Resample if needed
        if sr != self.sample_rate:
            wav = AF.resample(wav.unsqueeze(0), sr, self.sample_rate).squeeze(0)

        # Random crop or pad
        if wav.numel() > self.sample_length:
            start = torch.randint(0, wav.numel() - self.sample_length + 1, (1,)).item()
            wav = wav[start : start + self.sample_length]
        elif wav.numel() < self.sample_length:
            pad = self.sample_length - wav.numel()
            wav = F.pad(wav, (0, pad))

        # Get speaker label
        speaker_str = entry["speaker_id"]
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
