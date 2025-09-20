# Workflow Guide — Scaling Law Experiments

This playbook walks through the recommended end-to-end process for running Mixture-of-Experts scaling studies with reliable logging, optimizer comparisons, and reproducible artifacts.

## 1. Prepare the environment

```bash
make setup      # installs dependencies via setup.sh
python -m venv .venv && source .venv/bin/activate  # optional, but encouraged
```

The helper `Makefile` keeps the default Weights & Biases mode offline (`WANDB_MODE=offline`). Override it per command (e.g. `WANDB_MODE=online make demo-optimizers`) when you are ready to sync to the cloud.

## 2. Capture the hardware baseline

```bash
make auto-config
```

`auto_config.py` detects GPUs/CPUs and the wizard reuses the same snapshot. Each CLI or grid run also writes an `artifacts/environment.json` file with hardware specs, Python/Torch versions, selected optimizer, and git metadata.

## 3. Smoke-test a single experiment

```bash
make cli-run
```

This launches `python -m blueberry_cli run` for ~50 steps with AdamW, router diagnostics, and offline W&B logging. Outputs (under `experiments/runs/make_cli_run/`):

- `artifacts/metrics.json`, training/eval histories, `training_curve.png`
- `artifacts/environment.json` (hardware + software provenance)
- `deliverables.json` summarising checkpoints, W&B status, and validation flags

Use this step to confirm dataset download, tokenizer caching, and wandb credentials before larger sweeps.

## 4. Run the optimizer comparison grid

```bash
make demo-optimizers
```

This command executes a fixed-FLOPs grid with `E ∈ {1,4}` across both `adamw` and `muon` optimizers (single seed, 60 steps). The CLI now automatically:

- Logs per-run metrics + router stats for each variant
- Records environment metadata for every seed
- Writes `grid_summary.json`, `grid_aggregate.json` (means & standard deviations), and `grid_metadata.json` (command, git SHA, package versions)
- Generates the full plot catalogue (A1–D2) in `experiments/runs/demo_opt/artifacts/plots/`

All artifacts live under `experiments/runs/demo_opt/` and are safe to archive or sync. Switch to `WANDB_MODE=online` (or supply `--wandb-project`) to stream results to W&B.

## 5. Inspect outputs

Key files worth checking after the grid finishes:

- `artifacts/environment.json` inside each seed directory — hardware + software snapshot
- `artifacts/metrics.json` — final metrics including optimizer choice
- `artifacts/plots/` — regenerated A1–D2 figures for reports
- `grid_aggregate.json` — per-variant mean/std statistics
- `grid_metadata.json` — CLI invocation, optimizer list, status counts, git SHA (supports reproducibility checklists)

## 6. Promote to online W&B logging (optional)

```bash
export WANDB_API_KEY=<your-key>
WANDB_MODE=online make demo-optimizers \
  WANDB_PROJECT=research-experiments WANDB_ENTITY=<team>
```

All runs stream step metrics, summaries, and environment metadata. Each run also uploads `environment.json` and `metrics.json` so they are traceable in the W&B UI.

## 7. Build documentation and review plots

```bash
make docs
python -m mkdocs serve  # optional live preview
```

The documentation references the generated plots (A1–D2). If you rerun the grid, the CLI refreshes the images automatically, so rebuilding the docs reflects the latest results.

## 8. Clean up

```bash
make clean-artifacts
```

Removes cached datasets under `data_cache/` and per-run directories inside `experiments/runs/`. Plots committed to `docs/assets/plots` stay intact.

Following these steps produces the exact artifacts expected by the research tutorial (metrics CSVs, environment metadata, plots, and deliverables), while keeping the workflow reproducible and W&B ready.
