"""YAML設定ローダ。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from .modules.content_encoder import ContentEncoderConfig
from .modules.speaker_encoder import SpeakerEncoderConfig
from .modules.decoder import DecoderConfig
from .modules.rvq import ResidualVQConfig


@dataclass
class ExperimentConfig:
    name: str
    seed: int


@dataclass
class DataConfig:
    sample_rate: int
    frame_ms: float
    lookahead_frames: int
    cache_dir: str
    datasets: Any
    sample_length_sec: float
    reference_length_sec: float
    hubert_cache: str
    yin_cache: str


@dataclass
class TrainingConfig:
    batch_size: int
    num_steps: int
    optimizer: Dict[str, Any]
    losses: Dict[str, float]
    log_interval: int
    eval_interval: int
    ckpt_interval: int
    output_dir: str
    scheduler: Dict[str, Any] = None
    gan_warmup_steps: int = 2000
    gan_rampup_steps: int = 2000
    gradient_clip_norm: float = 1.0
    use_amp: bool = False


@dataclass
class InferenceConfig:
    chunk_ms: float
    quantize: bool
    export: Dict[str, Any]


@dataclass
class ModelConfig:
    num_hubert_labels: int
    content_encoder: ContentEncoderConfig
    speaker_encoder: SpeakerEncoderConfig
    decoder: DecoderConfig


@dataclass
class StreamVCConfig:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig


def load_config(path: str | Path) -> StreamVCConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    experiment = ExperimentConfig(**data["experiment"])
    data_conf = DataConfig(**data["data"])

    # Parse decoder config with RVQ
    decoder_data = data["model"]["decoder"].copy()
    if "rvq" in decoder_data:
        decoder_data["rvq"] = ResidualVQConfig(**decoder_data["rvq"])

    model_conf = ModelConfig(
        num_hubert_labels=data["model"].get("num_hubert_labels", 100),
        content_encoder=ContentEncoderConfig(**data["model"]["content_encoder"]),
        speaker_encoder=SpeakerEncoderConfig(**data["model"]["speaker_encoder"]),
        decoder=DecoderConfig(**decoder_data),
    )
    training = TrainingConfig(**data["training"])
    inference = InferenceConfig(**data["inference"])
    return StreamVCConfig(
        experiment=experiment,
        data=data_conf,
        model=model_conf,
        training=training,
        inference=inference,
    )

