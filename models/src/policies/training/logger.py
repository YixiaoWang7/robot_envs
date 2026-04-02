"""Optional Weights & Biases experiment logger."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ExperimentLogger:
    """
    Thin W&B wrapper.  When disabled all log methods are no-ops, so callers
    never need to guard on ``enabled``.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        out_dir: Path,
        wandb_project: str,
        wandb_entity: str,
        wandb_run_name: str,
        run_config: dict[str, Any],
    ):
        self.enabled: bool = bool(enabled)
        self._wandb: Any = None
        self._run: Any = None
        self.out_dir: Path = out_dir

        if self.enabled:
            try:
                import wandb  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "W&B logging is enabled but `wandb` is not installed. "
                    "Install it (e.g. `pip install wandb`) or disable W&B."
                ) from exc
            self._wandb = wandb
            self._run = wandb.init(
                project=str(wandb_project),
                entity=(None if not wandb_entity else str(wandb_entity)),
                name=(None if not wandb_run_name else str(wandb_run_name)),
                config=run_config,
                dir=str(out_dir),
            )

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def log_images(self, images: dict[str, Path], *, step: int) -> None:
        if not (self.enabled and self._wandb is not None):
            return
        payload = {k: self._wandb.Image(str(v)) for k, v in images.items()}
        self._wandb.log(payload, step=step)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
