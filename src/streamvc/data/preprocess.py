"""メタデータからキャッシュを生成する。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF


def _load_wave(path: Path, sample_rate: int) -> torch.Tensor:
    wav, sr = sf.read(str(path))
    wav = torch.from_numpy(wav).float()
    if len(wav.shape) > 1:  # Stereo to mono
        wav = wav.mean(dim=1)
    if sr != sample_rate:
        wav = AF.resample(wav.unsqueeze(0), sr, sample_rate).squeeze(0)
    return wav


def _crop(wav: torch.Tensor, num_samples: int) -> torch.Tensor:
    if wav.numel() <= num_samples:
        pad = num_samples - wav.numel()
        return torch.nn.functional.pad(wav, (0, pad))
    start = 0
    end = start + num_samples
    return wav[start:end]


def preprocess_metadata(
    metadata_path: Path,
    cache_dir: Path,
    dataset_root: Path,
    dataset_name: str,
    sample_rate: int,
    sample_length_sec: float,
    reference_length_sec: float,
    kmeans_path: Path,
    device: str = "cuda",
    split_filter: Optional[str] = None,
) -> None:
    bundle = torchaudio.pipelines.HUBERT_BASE
    hubert = bundle.get_model().to(device)
    hubert.eval()
    kmeans = joblib.load(kmeans_path)

    sample_len = int(sample_rate * sample_length_sec)
    ref_len = int(sample_rate * reference_length_sec)

    output_root = cache_dir / dataset_name
    output_root.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            split = entry.get("split", "train")
            if split_filter and split != split_filter:
                continue
            sample_id = entry.get("id")
            if sample_id is None:
                raise ValueError("metadata must include 'id'")
            source_path = dataset_root / entry["source"]
            ref_path = dataset_root / entry.get("reference", entry["source"])

            wav = _load_wave(source_path, sample_rate)
            source = _crop(wav, sample_len)

            reference_wav = _load_wave(ref_path, sample_rate)
            reference = _crop(reference_wav, ref_len)

            with torch.no_grad():
                feats, _ = hubert.extract_features(source.unsqueeze(0).to(device))
                hidden = feats[-1].squeeze(0).cpu().numpy()
            labels = torch.from_numpy(kmeans.predict(hidden))

            sample = {
                "source_audio": source,
                "target_wave": source.clone(),
                "target_reference": reference,
                "hubert_labels": labels,
            }

            split_dir = output_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            out_path = split_dir / f"{sample_id}.pt"
            torch.save(sample, out_path)

