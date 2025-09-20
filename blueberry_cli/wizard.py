from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import auto_config
from llm import MoEModelConfig

from .config import DEFAULT_RUNS_DIR, generate_experiment_id
from .hardware import HardwareSnapshot, collect_hardware_snapshot
from .modeling import ParameterEstimates, estimate_parameter_count
from .storage import write_json


@dataclass
class RunRequest:
    config: MoEModelConfig
    experiment_id: str
    out_dir: Path
    seed: int
    num_seeds: int
    save_checkpoint: bool
    notes: Optional[str]
    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    wandb_group: Optional[str]
    wandb_mode: Optional[str]
    tags: List[str]


RunCallback = Callable[[RunRequest], None]


def _render_hardware(console: Console, snapshot: HardwareSnapshot) -> None:
    table = Table(box=box.ROUNDED, title="Hardware Snapshot")
    table.add_column("Component", justify="left")
    table.add_column("Details", justify="left")
    gpu_summary = "No GPUs detected"
    if snapshot.num_gpus:
        rows: List[str] = []
        for gpu in snapshot.gpus:
            rows.append(f"#{gpu.index}: {gpu.name} ({gpu.memory_gb:.1f} GB)")
        gpu_summary = "\n".join(rows)
    table.add_row("GPU", gpu_summary)
    table.add_row("CUDA", "Available" if snapshot.cuda_available else "CPU-only")
    table.add_row("CPU cores", str(snapshot.num_cpus) if snapshot.num_cpus else "Unknown")
    if snapshot.system_memory_gb is not None:
        table.add_row("System RAM", f"{snapshot.system_memory_gb:.1f} GB")
    console.print(table)


def _render_config(console: Console, config: MoEModelConfig) -> None:
    table = Table(box=box.SIMPLE_HEAVY, title="Current Configuration")
    table.add_column("Field")
    table.add_column("Value", justify="right")
    fields = [
        ("d_model", config.d_model),
        ("n_layers", config.n_layers),
        ("n_heads", config.n_heads),
        ("d_ff", config.d_ff),
        ("num_experts", config.num_experts),
        ("expert_top_k", config.expert_top_k),
        ("batch_size", config.batch_size),
        ("grad_accum", config.gradient_accumulation_steps),
        ("optimizer", getattr(config, "optimizer", "adamw")),
        ("learning_rate", getattr(config, "learning_rate", None)),
        ("muon_lr", config.muon_lr),
        ("max_seq_len", config.max_seq_len),
        ("max_steps", config.max_steps),
        ("eval_steps", config.eval_steps),
    ]
    for key, value in fields:
        table.add_row(key, str(value))
    console.print(table)


def _render_parameters(console: Console, stats: ParameterEstimates) -> None:
    table = Table(box=box.SIMPLE_HEAVY, title="Parameter Estimate")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total params", f"{stats.total_params:,} ({stats.total_params_m:.2f} M)")
    table.add_row("Trainable params", f"{stats.trainable_params:,} ({stats.trainable_params_m:.2f} M)")
    table.add_row("Active params/token", f"{stats.active_params_per_token:,}")
    table.add_row("Total expert params", f"{stats.total_expert_params:,}")
    if stats.approx_dense_params is not None:
        table.add_row("Approx dense params", f"{stats.approx_dense_params:,}")
    console.print(table)


def _prompt_int(message: str, default: int, minimum: int = 1) -> int:
    while True:
        value = typer.prompt(message, default=str(default))
        try:
            result = int(value)
            if result < minimum:
                raise ValueError
            return result
        except ValueError:
            typer.echo(f"Please enter an integer >= {minimum}.")


def _prompt_float(message: str, default: float, minimum: float = 0.0) -> float:
    while True:
        value = typer.prompt(message, default=str(default))
        try:
            result = float(value)
            if result < minimum:
                raise ValueError
            return result
        except ValueError:
            typer.echo(f"Please enter a value >= {minimum}.")


def _parse_tags(text: str) -> List[str]:
    return [tag.strip() for tag in text.split(",") if tag.strip()]


def run_wizard(
    *,
    console: Console,
    run_callback: Optional[RunCallback] = None,
    auto_config_factory: Callable[[], auto_config.BlueberryAutoConfigurator] = auto_config.auto_configure,
    hardware_collector: Callable[[], HardwareSnapshot] = collect_hardware_snapshot,
    parameter_estimator: Callable[[MoEModelConfig], ParameterEstimates] = estimate_parameter_count,
) -> None:
    console.print(Panel.fit("Welcome to the Blueberry Research Lab", title="🫐 Blueberry"))

    configurator = auto_config_factory()
    snapshot = hardware_collector()
    _render_hardware(console, snapshot)

    config = configurator.get_model_config()
    config.expert_top_k = max(1, min(config.expert_top_k, config.num_experts))
    if not config.expert_top_k:
        config.expert_top_k = min(2, config.num_experts)

    console.print("Recommended configuration based on hardware:")
    _render_config(console, config)

    if typer.confirm("Would you like to customise the architecture?", default=False):
        config.d_model = _prompt_int("Hidden size (d_model)", config.d_model)
        config.n_layers = _prompt_int("Number of transformer layers", config.n_layers)
        config.n_heads = _prompt_int("Attention heads", config.n_heads)
        config.d_ff = _prompt_int("Expert FFN width (d_ff)", config.d_ff)
        config.num_experts = _prompt_int("Number of experts", config.num_experts)
        config.expert_top_k = _prompt_int(
            "Top-k experts per token", max(1, config.expert_top_k), minimum=1
        )
    if typer.confirm("Adjust training hyperparameters?", default=False):
        config.batch_size = _prompt_int("Batch size", config.batch_size)
        config.gradient_accumulation_steps = _prompt_int(
            "Gradient accumulation steps", config.gradient_accumulation_steps
        )
        config.muon_lr = _prompt_float("Muon learning rate", config.muon_lr, minimum=1e-5)
        current_lr = getattr(config, "learning_rate", config.muon_lr * 0.1)
        config.learning_rate = _prompt_float("AdamW learning rate", current_lr, minimum=1e-6)
        optimizer_choice = typer.prompt(
            "Optimizer (adamw/muon)", default=getattr(config, "optimizer", "adamw")
        )
        config.optimizer = optimizer_choice.strip().lower() or "adamw"
        if config.optimizer not in {"adamw", "muon"}:
            typer.echo("Unknown optimizer; defaulting to adamw")
            config.optimizer = "adamw"
        config.max_steps = _prompt_int("Training steps", config.max_steps)
    if typer.confirm("Adjust sequence length or evaluation settings?", default=False):
        config.max_seq_len = _prompt_int("Max sequence length", config.max_seq_len)
        config.eval_steps = _prompt_int("Evaluation steps", config.eval_steps)
    collect_router_stats = typer.confirm("Collect router diagnostics during training?", default=True)
    config.collect_router_stats = collect_router_stats

    experiment_id = typer.prompt("Experiment ID", default=generate_experiment_id("exp"))
    seed = _prompt_int("Base random seed", 42)
    num_seeds = _prompt_int("Number of seed replicates", 1)
    out_dir = Path(typer.prompt("Output directory", default=str(DEFAULT_RUNS_DIR)))
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = parameter_estimator(config)
    _render_parameters(console, stats)
    _render_config(console, config)

    console.print(Panel.fit("Ready to launch or export your experiment."))
    console.print("1) Launch training + evaluation\n2) Save config to JSON\n3) Exit")
    choice = _prompt_int("Select an option", 3, minimum=1)

    if choice == 2:
        path = Path(typer.prompt("Path to write config JSON", default=str(out_dir / "blueberry_config.json")))
        config_payload = asdict(config)
        if "log_milestones" in config_payload:
            milestones = config_payload["log_milestones"]
            if isinstance(milestones, tuple):
                config_payload["log_milestones"] = list(milestones)
        write_json(path, config_payload)
        console.print(Panel.fit(f"Configuration saved to {path}", title="Saved"))
        return
    if choice == 3:
        console.print("👋 Goodbye! Run `blueberry --help` anytime.")
        return

    if run_callback is None:
        console.print("No run callback available. Exiting.")
        return

    if not typer.confirm("Start training and evaluation now?", default=True):
        console.print("Run cancelled.")
        return

    use_wandb = typer.confirm("Log to Weights & Biases?", default=False)
    wandb_project = wandb_entity = wandb_group = wandb_mode = None
    tags: List[str] = []
    if use_wandb:
        wandb_project = typer.prompt("wandb project", default="research-experiments")
        wandb_entity = typer.prompt("wandb entity/team", default="", show_default=False) or None
        wandb_group = typer.prompt("wandb group", default=experiment_id)
        wandb_mode = typer.prompt("wandb mode (online/offline/run)", default="online")
        tag_input = typer.prompt("Tags (comma separated)", default="wizard")
        tags = _parse_tags(tag_input)
    notes = typer.prompt("Run notes", default="", show_default=False) or None
    save_checkpoint = typer.confirm("Save final checkpoint?", default=True)

    request = RunRequest(
        config=config,
        experiment_id=experiment_id,
        out_dir=out_dir,
        seed=seed,
        num_seeds=num_seeds,
        save_checkpoint=save_checkpoint,
        notes=notes,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        wandb_mode=wandb_mode,
        tags=tags,
    )
    console.print("🚀 Launching training...")
    run_callback(request)
