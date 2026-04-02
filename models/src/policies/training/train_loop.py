"""Training loop for image-conditioned robot flow policies."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import torch

from policies.algorithms.flow_matching import FlowMatchingAlgorithm
from policies.config import TrainConfig
from policies.training.evaluator import RobotFlowEvaluator
from policies.training.logger import ExperimentLogger
from policies.training.monitor import StatusMonitor
from policies.training.policy_wrapper import RobotFlowPolicyWrapper
from policies.training.trainer import CUDAPrefetcher
from policies.training.input_adapters import BaseInputAdapter


class RobotFlowTrainer:
    """
    End-to-end training: data fetch -> loss -> backward -> step -> log/eval.
    """

    def __init__(
        self,
        *,
        policy: RobotFlowPolicyWrapper,
        input_adapter: BaseInputAdapter,
        flow_algo: FlowMatchingAlgorithm,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        train_loader: Any,
        train_cfg: TrainConfig,
        evaluator: RobotFlowEvaluator,
        logger: ExperimentLogger,
        device: torch.device,
        out_dir: Path,
        run_config: Any = None,
    ):
        self.policy = policy
        self.input_adapter = input_adapter
        self.flow_algo = flow_algo
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.train_cfg = train_cfg
        self.evaluator = evaluator
        self.logger = logger
        self.device = device
        self.out_dir = out_dir
        self.run_config = run_config
        self.monitor = StatusMonitor(window=50)
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(device.type == "cuda" and bool(train_cfg.amp))
        )

    def _next_batch(self, prefetcher: CUDAPrefetcher, prefetch_iter: Any) -> tuple[dict[str, Any], Any]:
        try:
            batch = next(prefetch_iter)
        except StopIteration:
            prefetcher = CUDAPrefetcher(self.train_loader, self.policy.processor, self.device)
            prefetch_iter = iter(prefetcher)
            batch = next(prefetch_iter)
        return batch, prefetch_iter

    def _compute_grad_norm(self) -> float:
        total_sq = 0.0
        for p in self.policy.model.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            total_sq += float(torch.sum(g * g).item())
        return float(total_sq ** 0.5)

    def _save_checkpoint(self, *, step: int, tag: str, loss: float | None = None) -> Path:
        ckpt_dir = self.out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"{tag}.pt"
        self.policy.save_checkpoint(
            path,
            run_config=self.run_config,
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=(self.scheduler.state_dict() if self.scheduler is not None else None),
            step=int(step),
            loss=(None if loss is None else float(loss)),
        )
        return path

    def train(self) -> None:
        """Runs the full training loop for ``self.train_cfg.num_steps`` steps."""
        self.policy.model.train()
        prefetcher = CUDAPrefetcher(self.train_loader, self.policy.processor, self.device)
        prefetch_iter = iter(prefetcher)

        for warmup_idx in range(int(self.train_cfg.num_data_warmup)):
            batch, prefetch_iter = self._next_batch(prefetcher, prefetch_iter)
            if warmup_idx == 0:
                print("Warmup started...")
            _ = batch
        if int(self.train_cfg.num_data_warmup) > 0:
            print(f"Warmup complete: {int(self.train_cfg.num_data_warmup)} steps")

        for step in range(1, int(self.train_cfg.num_steps) + 1):
            t_fetch0 = time.perf_counter()
            batch, prefetch_iter = self._next_batch(prefetcher, prefetch_iter)
            fetch_ms = (time.perf_counter() - t_fetch0) * 1000.0
            train_kwargs = self.input_adapter.pack_train(batch)
            actions = train_kwargs["actions"]

            self.optimizer.zero_grad(set_to_none=True)

            t_step0 = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda" and bool(self.train_cfg.amp))):
                loss_tensor = self.policy.model.compute_loss(flow_algo=self.flow_algo, **train_kwargs)
            self.scaler.scale(loss_tensor).backward()

            if self.train_cfg.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), float(self.train_cfg.grad_clip_norm))

            grad_norm = self._compute_grad_norm()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            step_ms = (time.perf_counter() - t_step0) * 1000.0

            self.monitor.update(
                loss=float(loss_tensor.item()),
                fetch_ms=fetch_ms,
                step_ms=step_ms,
                grad_norm=grad_norm,
            )

            if step == 1 or (self.train_cfg.log_every > 0 and step % int(self.train_cfg.log_every) == 0):
                status = self.monitor.summary()
                status["train/loss"] = float(loss_tensor.item())
                status["train/lr"] = float(self.optimizer.param_groups[0]["lr"])
                progress = float(step) / max(1, int(self.train_cfg.num_steps))
                eta_sec = (status["train/elapsed_sec"] / max(progress, 1e-9)) - status["train/elapsed_sec"]
                status["train/eta_sec"] = float(max(0.0, eta_sec))
                if self.device.type == "cuda":
                    status["train/gpu_mem_gb"] = float(torch.cuda.max_memory_allocated(self.device) / 1e9)
                self.logger.log_metrics(status, step=step)
                print(
                    f"[step {step:>7d}/{self.train_cfg.num_steps}] "
                    f"loss={status['train/loss']:.6f} "
                    f"loss_win={status['train/loss_window']:.6f} "
                    f"fetch={status['train/fetch_ms_window']:.1f}ms "
                    f"step={status['train/step_ms_window']:.1f}ms "
                    f"sps={status['train/steps_per_sec_window']:.2f} "
                    f"grad={status['train/grad_norm_window']:.3f} "
                    f"eta={status['train/eta_sec']/60.0:.1f}m"
                )

            if self.train_cfg.eval_every > 0 and step % int(self.train_cfg.eval_every) == 0:
                eval_metrics = self.evaluator.evaluate(step=step)
                self.logger.log_metrics(eval_metrics, step=step)
                print(
                    f"[eval step {step:>7d}] "
                    f"val_loss={eval_metrics['val/loss']:.6f} "
                    f"val_mse_norm={eval_metrics['val/mse_norm']:.6f} "
                    f"val_mse_raw={eval_metrics['val/mse_raw']:.6f} "
                    f"val_mae_norm={eval_metrics['val/mae_norm']:.6f} "
                    f"val_mae_raw={eval_metrics['val/mae_raw']:.6f}"
                )

            if int(self.train_cfg.save_every) > 0 and step % int(self.train_cfg.save_every) == 0:
                _ = self._save_checkpoint(step=step, tag=f"ckpt_step_{step:07d}", loss=float(loss_tensor.item()))

        _ = self._save_checkpoint(step=int(self.train_cfg.num_steps), tag="ckpt_final")
