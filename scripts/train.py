"""StreamVC学習スクリプト。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from streamvc import StreamVCTrainer, load_config
from streamvc.data import build_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train StreamVC")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--eval", action="store_true", help="build eval loader")
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

    trainer = StreamVCTrainer(config)
    if args.device == "cpu":
        trainer.device = torch.device("cpu")
        trainer.pipeline.to(trainer.device)
        if trainer.discriminator is not None:
            trainer.discriminator.to(trainer.device)

    trainer.fit(train_loader, eval_loader)


if __name__ == "__main__":
    main()


