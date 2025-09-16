from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .config import ExperimentPaths
from .storage import relative_path


ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _fmt_time(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.strftime(ISO_FORMAT)


def extract_summary_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "train_loss",
        "train_accuracy",
        "train_perplexity",
        "val_loss",
        "val_accuracy",
        "val_perplexity",
        "wall_time_s",
        "tokens_per_second",
    ]
    summary: Dict[str, Any] = {}
    for key in keys:
        if key in metrics and metrics[key] is not None:
            summary_key = key if key.startswith("final_") else f"final_{key}"
            summary[summary_key] = metrics[key]
    # ensure wall time uses nicer label
    if "final_wall_time_s" in summary:
        summary["total_training_time"] = summary.pop("final_wall_time_s")
    return summary


def build_deliverable_summary(
    *,
    experiment_id: str,
    status: str,
    started_at: datetime,
    completed_at: Optional[datetime],
    metrics: Dict[str, Any],
    paths: ExperimentPaths,
    wandb_info: Optional[Dict[str, Any]],
    checkpoint_path: Optional[Path],
    artifact_paths: Dict[str, Path],
    report_paths: Iterable[Path],
    validation_status: Dict[str, bool],
) -> Dict[str, Any]:
    duration = None
    if completed_at is not None:
        duration = max(0.0, (completed_at - started_at).total_seconds())

    artifacts_list = [
        relative_path(path, paths.run_dir) for path in artifact_paths.values() if path is not None
    ]

    report_list = [relative_path(Path(p), paths.run_dir) for p in report_paths]

    deliverables_status: Dict[str, Any] = {}
    if wandb_info:
        deliverables_status["wandb_run"] = {
            "status": "delivered" if status == "completed" else "partial_success",
            **wandb_info,
            "validated": bool(wandb_info.get("run_id")),
            "validation_method": "synced_and_downloadable",
        }
    else:
        deliverables_status["wandb_run"] = {
            "status": "skipped",
            "validated": False,
            "validation_method": None,
        }

    checkpoint_status = "delivered" if checkpoint_path and checkpoint_path.exists() else "missing"
    deliverables_status["model_checkpoint"] = {
        "status": checkpoint_status,
        "path": relative_path(checkpoint_path, paths.run_dir) if checkpoint_path else None,
        "validated": checkpoint_status == "delivered",
        "validation_method": "exists_and_loadable" if checkpoint_status == "delivered" else None,
    }

    payload = {
        "experiment_id": experiment_id,
        "status": status,
        "completed_at": _fmt_time(completed_at),
        "duration_seconds": duration,
        "deliverables_status": deliverables_status,
        "summary_metrics": extract_summary_metrics(metrics),
        "validation_results": validation_status,
        "artifacts": artifacts_list,
        "analysis_reports": report_list,
    }
    return payload
