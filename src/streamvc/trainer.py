"""StreamVCトレーナ。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from .config import StreamVCConfig
from .pipeline import StreamVCPipeline
from .losses import LossWeights, content_loss, multi_resolution_stft_loss


class StreamVCTrainer:
    def __init__(self, config: StreamVCConfig, num_hubert_labels: Optional[int] = None, discriminator: Optional[nn.Module] = None) -> None:
        self.config = config
        num_labels = num_hubert_labels or config.model.num_hubert_labels
        self.pipeline = StreamVCPipeline(config, num_labels)
        self.loss_weights = LossWeights(
            content_ce=config.training.losses.get("content_ce_weight", 1.0),
            stft=config.training.losses.get("stft_weight", 1.0),
            l1=config.training.losses.get("l1_weight", 10.0),
            adversarial=config.training.losses.get("adversarial_weight", 0.0),
            feature_matching=config.training.losses.get("feature_matching_weight", 0.0),
        )
        self.discriminator = discriminator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline.to(self.device)
        if self.discriminator is not None:
            self.discriminator.to(self.device)
        self.optimizer = self._build_optimizer()
        self.step = 0
        self.writer: Optional[SummaryWriter] = None

    def _build_optimizer(self) -> Optimizer:
        opt_cfg = self.config.training.optimizer
        params = self.pipeline.parameters()
        opt_type = opt_cfg["type"].lower()
        if opt_type == "adamw":
            return torch.optim.AdamW(params, lr=opt_cfg["lr"], betas=tuple(opt_cfg.get("betas", (0.9, 0.99))), weight_decay=opt_cfg.get("weight_decay", 0.0))
        if opt_type == "adam":
            return torch.optim.Adam(params, lr=opt_cfg["lr"], betas=tuple(opt_cfg.get("betas", (0.9, 0.999))))
        raise ValueError(f"Unsupported optimizer type: {opt_cfg['type']}")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        source = batch["source_audio"].to(self.device)
        target_reference = batch["target_reference"].to(self.device)
        hubert_labels = batch["hubert_labels"].to(self.device)
        target_wave = batch["target_wave"].to(self.device)
        outputs = self.pipeline(source, target_reference, mode="train")
        generated = outputs["audio"]
        ce = content_loss(outputs["content_logits"], hubert_labels)
        l1 = torch.nn.functional.l1_loss(generated, target_wave)
        stft = multi_resolution_stft_loss(generated, target_wave)
        rvq = outputs["rvq_loss"]
        total = (
            self.loss_weights.content_ce * ce
            + self.loss_weights.l1 * l1
            + self.loss_weights.stft * stft
            + rvq
        )
        adv_loss = torch.tensor(0.0, device=generated.device)
        if self.discriminator is not None and self.loss_weights.adversarial > 0:
            pred_fake = self.discriminator(generated.unsqueeze(1))
            adv_loss = torch.mean((pred_fake - 1) ** 2)
            total = total + self.loss_weights.adversarial * adv_loss
        return {
            "loss": total,
            "loss_content": ce.detach(),
            "loss_l1": l1.detach(),
            "loss_stft": stft.detach(),
            "loss_rvq": rvq.detach(),
            "loss_adv": adv_loss.detach(),
        }

    def fit(self, train_loader, eval_loader=None) -> None:
        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(output_dir / "logs"))
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        self.pipeline.train()
        for batch in train_loader:
            self.optimizer.zero_grad()
            metrics = self.train_step(batch)
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.pipeline.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.writer is not None and self.step % self.config.training.log_interval == 0:
                self._log_metrics(metrics)
            self.step += 1
            if self.step % self.config.training.eval_interval == 0 and eval_loader is not None:
                self.evaluate(eval_loader)
            if self.step % self.config.training.ckpt_interval == 0:
                self.save_checkpoint(ckpt_dir / f"step_{self.step}.pt")
            if self.step >= self.config.training.num_steps:
                break
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, torch.Tensor]:
        self.pipeline.eval()
        losses = {}
        for batch in loader:
            metrics = self.train_step(batch)
            for key, value in metrics.items():
                losses.setdefault(key, []).append(value.detach().cpu())
        summary = {k: torch.stack(v).mean() for k, v in losses.items()}
        if self.writer is not None:
            for key, value in summary.items():
                self.writer.add_scalar(f"eval/{key}", value.item(), self.step)
        self.pipeline.train()
        return summary

    def _log_metrics(self, metrics: Dict[str, torch.Tensor]) -> None:
        if self.writer is None:
            return
        for key, value in metrics.items():
            if torch.is_tensor(value):
                self.writer.add_scalar(f"train/{key}", value.item(), self.step)

    def save_checkpoint(self, path: Path) -> None:
        state = {
            "model": self.pipeline.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
        }
        if self.discriminator is not None:
            state["discriminator"] = self.discriminator.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.pipeline.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = state.get("step", 0)
        if self.discriminator is not None and "discriminator" in state:
            self.discriminator.load_state_dict(state["discriminator"])

