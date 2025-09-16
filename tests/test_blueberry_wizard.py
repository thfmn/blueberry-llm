from __future__ import annotations

from pathlib import Path
from typing import List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import auto_config
import typer
from rich.console import Console
from typer.testing import CliRunner

from blueberry_cli.modeling import ParameterEstimates
from blueberry_cli.wizard import RunRequest, run_wizard
from blueberry_cli.hardware import HardwareSnapshot, GPUSpec
from llm import MoEModelConfig


class StubConfigurator:
    def __init__(self) -> None:
        self.config = auto_config.AutoConfig(
            num_gpus=0,
            gpu_memory_gb=0.0,
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            num_experts=2,
            batch_size=4,
            gradient_accumulation_steps=1,
            max_steps=10,
            learning_rate=0.001,
            max_seq_len=64,
            use_distributed=False,
            use_amp=False,
        )

    def get_model_config(self) -> MoEModelConfig:
        cfg = MoEModelConfig(
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            batch_size=4,
            max_steps=10,
            max_seq_len=64,
            num_experts=2,
            expert_top_k=1,
            gradient_accumulation_steps=1,
            muon_lr=0.001,
            eval_every=20,
            eval_steps=5,
        )
        cfg.weight_decay = 0.01
        cfg.dropout = 0.0
        cfg.collect_router_stats = False
        return cfg


def _app_for_wizard(
    *,
    run_callback,
    snapshot: HardwareSnapshot,
    stats: ParameterEstimates,
) -> typer.Typer:
    runner_app = typer.Typer()

    @runner_app.command()
    def launch() -> None:
        console = Console(force_terminal=False, color_system=None)
        run_wizard(
            console=console,
            run_callback=run_callback,
            auto_config_factory=StubConfigurator,
            hardware_collector=lambda: snapshot,
            parameter_estimator=lambda _: stats,
        )

    return runner_app


def test_wizard_exit_without_run(tmp_path: Path) -> None:
    snapshot = HardwareSnapshot(gpus=[], num_cpus=8, system_memory_gb=32.0, cuda_available=False)
    stats = ParameterEstimates(
        total_params=123_456,
        trainable_params=123_456,
        active_params_per_token=2_048,
        total_expert_params=8_192,
        approx_dense_params=64_000,
    )
    runner = CliRunner()
    app = _app_for_wizard(run_callback=None, snapshot=snapshot, stats=stats)

    # Sequence of blank inputs opts out of customisation and chooses exit option 3
    user_inputs = "\n" * 9
    result = runner.invoke(app, [], input=user_inputs)

    assert result.exit_code == 0
    assert "Hardware Snapshot" in result.output
    assert "Total params" in result.output
    assert "Goodbye" in result.output


def test_wizard_emits_run_request(tmp_path: Path) -> None:
    snapshot = HardwareSnapshot(
        gpus=[GPUSpec(index=0, name="Stub GPU", memory_gb=10.0)],
        num_cpus=16,
        system_memory_gb=64.0,
        cuda_available=True,
    )
    stats = ParameterEstimates(
        total_params=555_000,
        trainable_params=555_000,
        active_params_per_token=4_096,
        total_expert_params=16_384,
        approx_dense_params=111_000,
    )
    recorded: List[RunRequest] = []

    def capture(request: RunRequest) -> None:
        recorded.append(request)

    runner = CliRunner()
    app = _app_for_wizard(run_callback=capture, snapshot=snapshot, stats=stats)

    out_dir = tmp_path / "wizard-out"
    inputs = (
        "\n"  # customise architecture? -> default no
        "\n"  # adjust training? -> default no
        "\n"  # adjust sequence? -> default no
        "\n"  # router stats -> default yes
        "my-exp\n"  # experiment id
        "\n"  # base seed -> default 42
        "2\n"  # number of seeds
        f"{out_dir}\n"
        "1\n"  # choose launch option
        "\n"  # confirm start -> default yes
        "\n"  # wandb? -> default no
        "\n"  # notes -> empty
        "\n"  # save checkpoint -> default yes
    )
    result = runner.invoke(app, [], input=inputs)

    assert result.exit_code == 0
    assert recorded, "run_callback not triggered"
    request = recorded[0]
    assert request.experiment_id == "my-exp"
    assert request.num_seeds == 2
    assert request.out_dir == out_dir
    assert request.config.num_experts == 2
    assert request.config.collect_router_stats is True
    assert request.notes is None
    assert request.wandb_project is None
    assert request.save_checkpoint is True
