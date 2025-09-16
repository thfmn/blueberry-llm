from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from experiments.compute import ComputeBudget
from experiments.train_eval import run_single_experiment
from llm import MoEModelConfig, set_seed

from .config import ExperimentPaths, ExperimentRequest, WandbSettings
from .wandb_utils import wandb_run


class RunLogger:
    """Aggregate per-stage metrics and forward them to wandb if available."""

    def __init__(self, wandb_session: Optional[Any] = None) -> None:
        self._wandb = wandb_session
        self.train_history: List[Dict[str, Any]] = []
        self.eval_history: List[Dict[str, Any]] = []
        self.summary_events: List[Dict[str, Any]] = []

    def __call__(self, stage: str, payload: Dict[str, Any]) -> None:
        if stage == "train":
            self.train_history.append(payload)
        elif stage == "eval":
            self.eval_history.append(payload)
        else:
            self.summary_events.append(payload)
        self._log_to_wandb(stage, payload)

    def _log_to_wandb(self, stage: str, payload: Dict[str, Any]) -> None:
        if self._wandb is None:
            return
        numeric = {k: v for k, v in payload.items() if isinstance(v, (int, float))}
        if not numeric:
            return
        step = payload.get("step") if isinstance(payload.get("step"), (int, float)) else None
        prefix = "train/" if stage == "train" else f"{stage}/"
        log_payload = {f"{prefix}{k}": v for k, v in numeric.items() if k != "step"}
        if not log_payload:
            return
        self._wandb.log(log_payload, step=step)


class SingleRunResult:
    def __init__(
        self,
        *,
        status: str,
        metrics: Dict[str, Any],
        started_at: datetime,
        completed_at: Optional[datetime],
        duration: float,
        train_history: List[Dict[str, Any]],
        eval_history: List[Dict[str, Any]],
        summary_events: List[Dict[str, Any]],
        wandb_info: Optional[Dict[str, Any]],
        checkpoint_path: Optional[Path],
        error: Optional[str] = None,
    ) -> None:
        self.status = status
        self.metrics = metrics
        self.started_at = started_at
        self.completed_at = completed_at
        self.duration = duration
        self.train_history = train_history
        self.eval_history = eval_history
        self.summary_events = summary_events
        self.wandb_info = wandb_info
        self.checkpoint_path = checkpoint_path
        self.error = error


def _extract_wandb_info(run: Optional[Any]) -> Optional[Dict[str, Any]]:
    if run is None:
        return None
    info: Dict[str, Any] = {
        "project": getattr(run, "project", None),
        "run_id": getattr(run, "id", None),
        "name": getattr(run, "name", None),
        "url": getattr(run, "url", None),
    }
    return {k: v for k, v in info.items() if v is not None}


def run_seed(
    request: ExperimentRequest,
    *,
    seed_value: int,
    wandb_settings: Optional[WandbSettings],
    paths: ExperimentPaths,
    run_name: str,
    tags: List[str],
    notes: Optional[str],
    save_checkpoint: bool,
) -> SingleRunResult:
    config: MoEModelConfig = request.clone_config()
    set_seed(seed_value)
    start_time = datetime.utcnow()
    wall_start = time.time()
    wandb_config = asdict(config)

    checkpoint_path: Optional[Path] = None
    metrics: Dict[str, Any] = {}
    status = "failed"
    error_message: Optional[str] = None
    completed: Optional[datetime] = None

    with wandb_run(
        wandb_settings,
        run_name=run_name,
        config={**wandb_config, "seed": seed_value},
        tags=[*tags, f"seed:{seed_value}", f"experts:{config.num_experts}"],
        notes=notes,
    ) as wb:
        logger = RunLogger(wb)
        try:
            model, metrics = run_single_experiment(config, log_hook=logger)
            budget = ComputeBudget(
                d_model=config.d_model,
                top_k=config.expert_top_k,
                d_ff_expert=config.d_ff,
                num_experts=config.num_experts,
            )
            metrics.update(budget.as_dict())
            status = "completed"
            completed = datetime.utcnow()
        except Exception as exc:  # pragma: no cover - surfaced to CLI
            error_message = str(exc)
            metrics["error"] = error_message
            status = "failed"
        finally:
            if completed is None:
                completed = datetime.utcnow()
        duration = time.time() - wall_start

        if status == "completed" and save_checkpoint:
            checkpoint_path = paths.artifacts_dir / "final_model.pt"
            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "metrics": metrics,
                "created_at": datetime.utcnow().isoformat(),
            }
            torch.save(checkpoint_payload, checkpoint_path)

    wandb_info = _extract_wandb_info(locals().get("wb"))

    return SingleRunResult(
        status=status,
        metrics=metrics,
        started_at=start_time,
        completed_at=completed,
        duration=duration,
        train_history=logger.train_history,
        eval_history=logger.eval_history,
        summary_events=logger.summary_events,
        wandb_info=wandb_info,
        checkpoint_path=checkpoint_path,
        error=error_message,
    )
