"""キャッシュ済み特徴を読み込むDataset。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from ..config import StreamVCConfig


class StreamVCCacheDataset(Dataset):
    """`cache_root/dataset_name/split/*.pt` を読み込む。"""

    def __init__(self, cache_root: Path, dataset_name: str, split: str = "train") -> None:
        self.dataset_name = dataset_name
        self.cache_root = Path(cache_root)
        self.split = split
        pattern = self.cache_root / dataset_name / split
        if not pattern.exists():
            raise FileNotFoundError(f"Cache directory not found: {pattern}")
        self.files = sorted(pattern.glob("*.pt"))
        if not self.files:
            raise RuntimeError(f"No cache files under {pattern}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = torch.load(self.files[idx])
        required = {"source_audio", "target_wave", "target_reference", "hubert_labels"}
        missing = required.difference(sample.keys())
        if missing:
            raise KeyError(f"Missing keys {missing} in {self.files[idx]}")
        return sample


def _pad_stack(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    lengths = [t.shape[-1] for t in tensors]
    max_len = max(lengths)
    padded = [F.pad(t, (0, max_len - t.shape[-1])) for t in tensors]
    return torch.stack(padded)


def _collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    source = _pad_stack([item["source_audio"] for item in batch])
    target_wave = _pad_stack([item["target_wave"] for item in batch])
    reference = _pad_stack([item["target_reference"] for item in batch])
    labels = _pad_stack([item["hubert_labels"].to(torch.long) for item in batch])
    return {
        "source_audio": source,
        "target_wave": target_wave,
        "target_reference": reference,
        "hubert_labels": labels,
    }


def build_dataloader(
    config: StreamVCConfig,
    split: str = "train",
    batch_size: int | None = None,
    shuffle: bool = True,
) -> DataLoader:
    cache_root = Path(config.data.cache_dir)
    datasets: List[StreamVCCacheDataset] = []
    sample_weights: List[float] = []
    for dataset_cfg in config.data.datasets:
        dataset = StreamVCCacheDataset(cache_root, dataset_cfg["name"], split)
        datasets.append(dataset)
        weight = float(dataset_cfg.get("weight", 1.0))
        sample_weights.extend([weight] * len(dataset))

    merged = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    batch = batch_size or config.training.batch_size

    if shuffle:
        weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
        sampler = WeightedRandomSampler(weights_tensor, num_samples=len(sample_weights), replacement=True)
        loader = DataLoader(merged, batch_size=batch, sampler=sampler, collate_fn=_collate_fn)
    else:
        loader = DataLoader(merged, batch_size=batch, shuffle=False, collate_fn=_collate_fn)
    return loader

