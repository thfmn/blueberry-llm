from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from llm import MoEModelConfig

DEFAULT_RUNS_DIR = Path("experiments/runs")


def generate_experiment_id(prefix: str = "exp") -> str:
    """Generate a timestamped experiment identifier."""
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{ts}"


@dataclass
class WandbSettings:
    project: Optional[str]
    entity: Optional[str] = None
    group: Optional[str] = None
    mode: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def is_enabled(self) -> bool:
        return bool(self.project)


@dataclass
class ExperimentPaths:
    run_dir: Path
    artifacts_dir: Path
    reports_dir: Path

    @classmethod
    def from_base(cls, base: Path) -> "ExperimentPaths":
        return cls(
            run_dir=base,
            artifacts_dir=base / "artifacts",
            reports_dir=base / "reports",
        )

    def ensure(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentRequest:
    experiment_id: str
    base_config: MoEModelConfig
    output_root: Path
    wandb: Optional[WandbSettings]
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def build_run_directory(
        self,
        *,
        replicate: Optional[int] = None,
        variant: Optional[str] = None,
    ) -> ExperimentPaths:
        run_dir = self.output_root / self.experiment_id
        if variant is not None:
            run_dir = run_dir / variant
        if replicate is not None:
            run_dir = run_dir / f"seed_{replicate}"
        paths = ExperimentPaths.from_base(run_dir)
        paths.ensure()
        return paths

    def clone_config(self, **updates) -> MoEModelConfig:
        return replace(self.base_config, **updates)
