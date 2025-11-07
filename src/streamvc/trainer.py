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
from .losses import LossWeights, content_loss, multi_resolution_stft_loss, frame_rms_loss, multiband_rms_loss


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
            rms=config.training.losses.get("rms_weight", 0.1),
            multiband_rms=config.training.losses.get("multiband_rms_weight", 0.05),
        )
        self.discriminator = discriminator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline.to(self.device)
        if self.discriminator is not None:
            self.discriminator.to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.step = 0
        self.writer: Optional[SummaryWriter] = None

    def _build_optimizer(self) -> Optimizer:
        opt_cfg = self.config.training.optimizer
        opt_type = opt_cfg["type"].lower()

        # Phase A: Separate parameter groups for out_proj and final_conv (no weight decay)
        protected_params = []
        other_params = []

        for name, param in self.pipeline.named_parameters():
            if 'decoder.out_proj' in name or 'decoder.rvq.final_conv' in name:
                protected_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {'params': other_params, 'weight_decay': opt_cfg.get("weight_decay", 0.0)},
            {'params': protected_params, 'weight_decay': 0.0},  # No decay for output layers
        ]

        if opt_type == "adamw":
            return torch.optim.AdamW(param_groups, lr=opt_cfg["lr"], betas=tuple(opt_cfg.get("betas", (0.9, 0.99))))
        if opt_type == "adam":
            return torch.optim.Adam(param_groups, lr=opt_cfg["lr"], betas=tuple(opt_cfg.get("betas", (0.9, 0.999))))
        raise ValueError(f"Unsupported optimizer type: {opt_cfg['type']}")

    def _build_scheduler(self):
        """Build learning rate scheduler with warmup support."""
        scheduler_cfg = self.config.training.scheduler
        if not scheduler_cfg or scheduler_cfg.get("type", "none") == "none":
            return None

        sched_type = scheduler_cfg["type"].lower()
        warmup_steps = scheduler_cfg.get("warmup_steps", 0)

        if sched_type == "cosine":
            # Cosine annealing with warmup
            total_steps = self.config.training.num_steps
            eta_min = scheduler_cfg.get("eta_min", 1e-6)
            base_lr = self.config.training.optimizer["lr"]

            # Create warmup + cosine scheduler
            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    # Linear warmup: 0 → 1
                    return max(float(current_step) / float(max(1, warmup_steps)), 1e-8)
                # Cosine annealing after warmup: 1 → eta_min/base_lr
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())
                return max(eta_min / base_lr, cosine_decay)

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        elif sched_type == "linear":
            # Linear decay with warmup
            total_steps = self.config.training.num_steps

            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        else:
            raise ValueError(f"Unsupported scheduler type: {sched_type}")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        source = batch["source_audio"].to(self.device)
        target_reference = batch["target_reference"].to(self.device)
        hubert_labels = batch["hubert_labels"].to(self.device)
        target_wave = batch["target_wave"].to(self.device)
        outputs = self.pipeline(source, target_reference, mode="train")
        generated = outputs["audio"]
        # Match generated audio length to target
        if generated.shape[1] != target_wave.shape[1]:
            if generated.shape[1] > target_wave.shape[1]:
                generated = generated[:, :target_wave.shape[1]]
            else:
                padding = target_wave.shape[1] - generated.shape[1]
                generated = torch.nn.functional.pad(generated, (0, padding))
        ce = content_loss(outputs["content_logits"], hubert_labels)
        l1 = torch.nn.functional.l1_loss(generated, target_wave)
        stft = multi_resolution_stft_loss(generated, target_wave)
        rvq = outputs["rvq_loss"]

        # Phase A: RMS supervision
        rms = frame_rms_loss(generated, target_wave) if self.loss_weights.rms > 0 else torch.tensor(0.0, device=generated.device)
        multiband_rms = multiband_rms_loss(generated, target_wave) if self.loss_weights.multiband_rms > 0 else torch.tensor(0.0, device=generated.device)

        total = (
            self.loss_weights.content_ce * ce
            + self.loss_weights.l1 * l1
            + self.loss_weights.stft * stft
            + rvq
            + self.loss_weights.rms * rms
            + self.loss_weights.multiband_rms * multiband_rms
        )
        adv_loss = torch.tensor(0.0, device=generated.device)
        if self.discriminator is not None and self.loss_weights.adversarial > 0:
            pred_fake = self.discriminator(generated.unsqueeze(1))
            adv_loss = torch.mean((pred_fake - 1) ** 2)
            total = total + self.loss_weights.adversarial * adv_loss

        # Compute RVQ diagnostics (perplexity, code usage)
        rvq_metrics = {}
        if "codes" in outputs:
            codes = outputs["codes"]  # List of (B, T) tensors, one per quantizer
            for i, code_tensor in enumerate(codes):
                # Compute perplexity: exp(entropy)
                # Count code usage
                unique_codes = torch.unique(code_tensor)
                usage_ratio = len(unique_codes) / self.pipeline.decoder.rvq.config.codebook_size

                # Compute perplexity (lower is worse, ideally close to codebook_size)
                # Perplexity = exp(entropy) where entropy = -sum(p_k * log(p_k))
                code_counts = torch.bincount(code_tensor.flatten(), minlength=self.pipeline.decoder.rvq.config.codebook_size)
                code_probs = code_counts.float() / code_counts.sum()
                # Avoid log(0)
                code_probs = code_probs[code_probs > 0]
                entropy = -(code_probs * torch.log(code_probs)).sum()
                perplexity = torch.exp(entropy)

                rvq_metrics[f"rvq_perplexity_q{i}"] = perplexity.detach()
                rvq_metrics[f"rvq_usage_q{i}"] = usage_ratio

        # Phase A: Track out_proj and final_conv norms
        out_proj_weight_norm = torch.norm(self.pipeline.decoder.out_proj.weight).detach()
        out_proj_bias_norm = torch.norm(self.pipeline.decoder.out_proj.bias).detach()

        final_conv_norm = torch.tensor(0.0, device=generated.device)
        if hasattr(self.pipeline.decoder.rvq, 'final_conv'):
            final_conv_norm = torch.norm(self.pipeline.decoder.rvq.final_conv.weight).detach()

        # Audio RMS statistics
        audio_rms_mean = torch.sqrt((generated ** 2).mean() + 1e-8).detach()

        return {
            "loss": total,
            "loss_content": ce.detach(),
            "loss_l1": l1.detach(),
            "loss_stft": stft.detach(),
            "loss_rvq": rvq.detach(),
            "loss_adv": adv_loss.detach(),
            "loss_rms": rms.detach(),
            "loss_multiband_rms": multiband_rms.detach(),
            "pre_rvq_std": outputs["pre_rvq_std"].detach(),
            "out_proj_weight_norm": out_proj_weight_norm,
            "out_proj_bias_norm": out_proj_bias_norm,
            "final_conv_norm": final_conv_norm,
            "audio_rms_mean": audio_rms_mean,
            **rvq_metrics,
        }

    def fit(self, train_loader, eval_loader=None) -> None:
        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(output_dir / "logs"))
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        self.pipeline.train()

        # Progressive RVQ configuration
        progressive_steps = self.pipeline.decoder.rvq.config.progressive_steps

        for batch in train_loader:
            # Update active quantizers based on training step
            if progressive_steps > 0:
                num_active = min(1 + self.step // progressive_steps, self.pipeline.decoder.rvq.config.num_quantizers)
                self.pipeline.decoder.rvq.set_num_active_quantizers(num_active)

            self.optimizer.zero_grad()
            metrics = self.train_step(batch)
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.pipeline.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.writer is not None and self.step % self.config.training.log_interval == 0:
                self._log_metrics(metrics)
                # Log learning rate and active quantizers
                if self.scheduler is not None:
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.step)
                if progressive_steps > 0:
                    self.writer.add_scalar("train/num_active_quantizers", self.pipeline.decoder.rvq.num_active_quantizers, self.step)
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
            elif isinstance(value, (int, float)):
                self.writer.add_scalar(f"train/{key}", value, self.step)

    def save_checkpoint(self, path: Path) -> None:
        state = {
            "model": self.pipeline.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        if self.discriminator is not None:
            state["discriminator"] = self.discriminator.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.pipeline.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = state.get("step", 0)
        if self.scheduler is not None and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        if self.discriminator is not None and "discriminator" in state:
            self.discriminator.load_state_dict(state["discriminator"])

