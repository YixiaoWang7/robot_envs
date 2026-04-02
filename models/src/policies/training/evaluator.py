"""Periodic validation evaluator for flow-matching robot policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from policies.algorithms.flow_matching import FlowMatchingAlgorithm
from policies.config import TrainConfig
from policies.training.logger import ExperimentLogger
from policies.training.policy_wrapper import RobotFlowPolicyWrapper
from policies.training.input_adapters import BaseInputAdapter


class RobotFlowEvaluator:
    """
    Computes validation metrics: loss, MSE/MAE (normalized and raw), and
    optional best-of-K MSE.
    """

    def __init__(
        self,
        *,
        policy: RobotFlowPolicyWrapper,
        input_adapter: BaseInputAdapter,
        flow_algo: FlowMatchingAlgorithm,
        val_loader: Any,
        out_dir: Path,
        logger: ExperimentLogger,
        device: torch.device,
        cfg: TrainConfig,
    ):
        self.policy = policy
        self.input_adapter = input_adapter
        self.flow_algo = flow_algo
        self.val_loader = val_loader
        self.out_dir = out_dir
        self.logger = logger
        self.device = device
        self.cfg = cfg

    @torch.no_grad()
    def evaluate(self, *, step: int) -> dict[str, float]:
        self.policy.model.eval()
        losses: list[float] = []
        mses_norm: list[float] = []
        maes_norm: list[float] = []
        mses_raw: list[float] = []
        maes_raw: list[float] = []
        best_k: list[float] = []
        best_k_raw: list[float] = []

        val_iter = iter(self.val_loader)
        for _ in range(int(self.cfg.eval_num_batches)):
            try:
                raw_batch = next(val_iter)
            except StopIteration:
                break
            batch = self.policy.process_raw_batch(raw_batch, device=self.device)
            train_kwargs = self.input_adapter.pack_train(batch)
            infer_kwargs = self.input_adapter.pack_infer(batch)
            actions = train_kwargs["actions"]

            loss_tensor = self.policy.model.compute_loss(flow_algo=self.flow_algo, **train_kwargs)
            actions_pred = self.policy.model.generate_actions(
                flow_algo=self.flow_algo,
                batch_size=int(actions.shape[0]),
                **infer_kwargs,
            )

            diff_norm = actions_pred - actions
            losses.append(float(loss_tensor.item()))
            mses_norm.append(float((diff_norm.pow(2).mean()).item()))
            maes_norm.append(float((diff_norm.abs().mean()).item()))

            actions_raw = self.policy.denormalize_actions(actions)
            actions_pred_raw = self.policy.denormalize_actions(actions_pred)
            diff_raw = actions_pred_raw - actions_raw
            mses_raw.append(float((diff_raw.pow(2).mean()).item()))
            maes_raw.append(float((diff_raw.abs().mean()).item()))

            num_samples = int(self.cfg.eval_best_of_k)
            if num_samples > 1:
                samples: list[torch.Tensor] = [actions_pred]
                for _ in range(num_samples - 1):
                    sample_pred = self.policy.model.generate_actions(
                        flow_algo=self.flow_algo,
                        batch_size=int(actions.shape[0]),
                        **infer_kwargs,
                    )
                    samples.append(sample_pred)
                stacked = torch.stack(samples, dim=1)
                mse_per_sample = ((stacked - actions.unsqueeze(1)) ** 2).mean(dim=(2, 3))
                best_k.append(float(mse_per_sample.min(dim=1).values.mean().item()))

                stacked_raw = self.policy.denormalize_actions(stacked)
                actions_raw_u = actions_raw.unsqueeze(1)
                mse_per_sample_raw = ((stacked_raw - actions_raw_u) ** 2).mean(dim=(2, 3))
                best_k_raw.append(float(mse_per_sample_raw.min(dim=1).values.mean().item()))

        metrics = {
            "val/loss": float(np.mean(losses)) if losses else float("nan"),
            "val/mse_norm": float(np.mean(mses_norm)) if mses_norm else float("nan"),
            "val/mae_norm": float(np.mean(maes_norm)) if maes_norm else float("nan"),
            "val/mse_raw": float(np.mean(mses_raw)) if mses_raw else float("nan"),
            "val/mae_raw": float(np.mean(maes_raw)) if maes_raw else float("nan"),
        }
        if best_k:
            metrics["val/best_of_k_mse_norm"] = float(np.mean(best_k))
        if best_k_raw:
            metrics["val/best_of_k_mse_raw"] = float(np.mean(best_k_raw))
        self.policy.model.train()
        return metrics
