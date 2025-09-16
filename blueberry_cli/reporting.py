from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

from .config import ExperimentPaths
from .storage import write_json


def _numeric_series(records: Iterable[Dict[str, float]], key: str) -> List[float]:
    values: List[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def render_summary(
    console: Console,
    *,
    experiment_id: str,
    status: str,
    metrics: Dict[str, float],
    duration: float,
    artifacts: Dict[str, Optional[Path]],
) -> None:
    table = Table(title=f"Experiment {experiment_id}")
    table.add_column("Key", justify="left")
    table.add_column("Value", justify="right")
    table.add_row("status", status)
    table.add_row("duration_s", f"{duration:.2f}" if duration else "-")
    for key in ("val_loss", "val_accuracy", "val_perplexity", "tokens_per_second"):
        if key in metrics:
            table.add_row(key, f"{metrics[key]:.4f}" if isinstance(metrics[key], (int, float)) else str(metrics[key]))
    for label, path in artifacts.items():
        if path is not None:
            table.add_row(label, str(path))
    console.print(table)


def save_training_curve(train_history: List[Dict[str, float]], out_path: Path) -> Optional[Path]:
    steps = _numeric_series(train_history, "step")
    losses = _numeric_series(train_history, "loss")
    if not steps or not losses:
        return None
    plt.figure(figsize=(6, 4))
    plt.plot(steps, losses, label="train loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return out_path


def write_metrics(metrics: Dict[str, float], out_path: Path) -> Path:
    write_json(out_path, metrics)
    return out_path


def write_reports(paths: ExperimentPaths, metrics: Dict[str, float], notes: Optional[str]) -> List[Path]:
    reports: List[Path] = []
    performance = paths.reports_dir / "performance_analysis.md"
    lines = ["# Performance Analysis", "", f"- Final validation loss: {metrics.get('val_loss', 'n/a')}", f"- Final validation accuracy: {metrics.get('val_accuracy', 'n/a')}", f"- Tokens per second: {metrics.get('tokens_per_second', 'n/a')}"]
    if notes:
        lines.append(f"- Notes: {notes}")
    performance.write_text("\n".join(lines), encoding="utf-8")
    reports.append(performance)

    ablation = paths.reports_dir / "ablation_study.md"
    ablation.write_text(
        """# Ablation Study

Currently no ablations recorded for this run. Add comparisons by invoking the CLI with different grids or seeds and storing the outputs under the same experiment ID.
""",
        encoding="utf-8",
    )
    reports.append(ablation)

    baseline = paths.reports_dir / "baseline_comparison.md"
    baseline.write_text(
        """# Baseline Comparison

Baseline metrics can be logged via `blueberry-cli grid` to produce dense vs. MoE comparisons. Update this report with findings once baselines are established.
""",
        encoding="utf-8",
    )
    reports.append(baseline)
    return reports
