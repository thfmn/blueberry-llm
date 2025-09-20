from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import torch

from .hardware import GPUSpec, HardwareSnapshot, collect_hardware_snapshot

try:  # Python 3.8 fallback
    from importlib import metadata as importlib_metadata  # type: ignore
except ImportError:  # pragma: no cover - py38 fallback
    import importlib_metadata  # type: ignore


_PACKAGE_CANDIDATES = ["typer", "rich", "wandb", "matplotlib", "seaborn"]


def _gpu_spec_to_dict(gpu: GPUSpec) -> Dict[str, Any]:
    return {
        "index": gpu.index,
        "name": gpu.name,
        "memory_gb": round(gpu.memory_gb, 2),
    }


def hardware_snapshot_to_dict(snapshot: HardwareSnapshot) -> Dict[str, Any]:
    return {
        "num_gpus": snapshot.num_gpus,
        "cuda_available": snapshot.cuda_available,
        "gpus": [_gpu_spec_to_dict(g) for g in snapshot.gpus],
        "num_cpus": snapshot.num_cpus,
        "system_memory_gb": snapshot.system_memory_gb,
    }


def _git_metadata() -> Dict[str, Optional[str]]:
    def _run(cmd: list[str]) -> Optional[str]:
        try:
            return (
                subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
        except Exception:
            return None

    status_output = _run(["git", "status", "--short"])

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "commit_short": _run(["git", "rev-parse", "--short", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status_output),
    }


def _package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for pkg in _PACKAGE_CANDIDATES:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except Exception:
            versions[pkg] = None
    return versions


def gather_environment_metadata(
    *,
    include_snapshot: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    snapshot = collect_hardware_snapshot() if include_snapshot else None

    metadata: Dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": sys.version,
        "platform": platform.platform(),
        "platform_release": platform.release(),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "WANDB_MODE": os.environ.get("WANDB_MODE"),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "packages": _package_versions(),
        "git": _git_metadata(),
    }

    if snapshot is not None:
        metadata["hardware"] = hardware_snapshot_to_dict(snapshot)

    if extra:
        metadata.update(extra)

    return metadata
