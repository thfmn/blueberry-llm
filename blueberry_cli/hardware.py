from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import torch

try:  # Optional dependency for richer stats
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional
    psutil = None


@dataclass
class GPUSpec:
    index: int
    name: str
    memory_gb: float


@dataclass
class HardwareSnapshot:
    gpus: List[GPUSpec]
    num_cpus: Optional[int]
    system_memory_gb: Optional[float]
    cuda_available: bool

    @property
    def num_gpus(self) -> int:
        return len(self.gpus)


def collect_hardware_snapshot() -> HardwareSnapshot:
    gpus: List[GPUSpec] = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpus.append(
                GPUSpec(
                    index=idx,
                    name=props.name,
                    memory_gb=props.total_memory / (1024**3),
                )
            )

    num_cpus = os.cpu_count()
    system_memory_gb: Optional[float] = None
    if psutil is not None:
        try:
            system_memory_gb = psutil.virtual_memory().total / (1024**3)
        except Exception:  # pragma: no cover - defensive
            system_memory_gb = None

    return HardwareSnapshot(
        gpus=gpus,
        num_cpus=num_cpus,
        system_memory_gb=system_memory_gb,
        cuda_available=torch.cuda.is_available(),
    )
