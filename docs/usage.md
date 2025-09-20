# Usage

## Install

```bash
chmod +x setup.sh && ./setup.sh
pip install pandas seaborn matplotlib mkdocs mkdocs-material
```

## Run a Fixed-FLOPs Grid

```bash
python experiments/run_scaling.py \
  --regime flops \
  --experts 1,2,4,8,16,32 \
  --top-k 2 \
  --d-model 384 --n-heads 8 --n-layers 6 --d-ff-dense 1536 \
  --seq-len 512 --batch-size 16 --max-steps 200 \
  --seeds 3 \
  --collect-router-stats \
  --out-dir experiments/out
```

## Run a Fixed-Params Grid

```bash
python experiments/run_scaling.py --regime params --experts 1,2,4,8,16,32 --out-dir experiments/out
```

## Generate Plots

```bash
python experiments/plotting.py --csv experiments/out/results_flops.csv --out experiments/plots
```

### Generate Mock Plots for Docs

```bash
python -m experiments.generate_mock_plots --out docs/assets/plots
```

## Experiment CLI (`blueberry-cli`)

The new Typer-based CLI wraps training, evaluation, and logging into a single entry point. Run a single experiment and stream metrics to Weights & Biases:

```bash
python -m blueberry_cli run \
  --experiment-id exp_debug \
  --d-model 384 --n-heads 8 --n-layers 6 --d-ff 1536 \
  --num-experts 8 --expert-top-k 2 \
  --max-steps 200 --batch-size 24 \
  --optimizer adamw --learning-rate 0.001 \
  --wandb-project research-experiments --tag debug
```

Each run writes structured deliverables into `experiments/runs/<id>/`, including:

- `artifacts/metrics.json`, `train_history.json`, and `training_curve.png`
- `artifacts/environment.json` capturing hardware + software metadata
- Optional `artifacts/final_model.pt` checkpoint
- Markdown reports under `reports/`
- `deliverables.json` summarizing WandB runs, checkpoints, and validation flags (schema mirrors the research deliverables checklist)

Use `--num-seeds` to repeat the run with different random seeds; aggregate metrics are written to `artifacts/aggregate_metrics.json`.

### Guided Wizard (`./blueberry`)

Prefer an interactive workflow? Launch the wizard:

```bash
./blueberry
```

The wizard detects hardware (GPU/CPU), suggests a configuration, estimates parameter counts, and walks through evaluation/logging choices before dispatching to `run`. Choose to export the config as JSON or launch training/eval directly (with optional Weights & Biases logging).

While customising training hyperparameters you can switch between `adamw` (default) and `muon`; the wizard updates both Muon and AdamW learning rates and records the selection for downstream logging.

### Scaling sweeps with the CLI

```bash
python -m blueberry_cli grid \
  --regime flops \
  --experts 1,2,4,8,16 \
  --top-k 2 \
  --max-steps 200 --batch-size 16 \
  --optimizers adamw,muon \
  --adamw-lr 0.001 --muon-lr 0.01 \
  --wandb-project research-experiments --tag scaling
```

The grid command now:

- Maintains the classic CSV log (`results_<regime>.csv`)
- Emits per-configuration deliverables (including `environment.json`)
- Produces aggregated statistics with mean/standard deviation (`grid_aggregate.json`)
- Records provenance metadata (`grid_metadata.json`) containing the CLI invocation, git SHA, and package versions
- Auto-generates the named plot set (A1–D2) into `artifacts/plots/`

You can compare optimisers by passing multiple values to `--optimizers`; variant labels include both expert count and optimiser for clarity.

### Makefile shortcuts

Common workflows are available through the top-level `Makefile`:

- `make setup` — run `setup.sh`
- `make auto-config` — print detected hardware + suggested config
- `make cli-run` — launch a single short run with offline Weights & Biases logging
- `make demo-optimizers` — execute a small fixed-FLOPs grid across `adamw` and `muon`, generating plots and metadata
- `make docs` — build the MkDocs site
- `make clean-artifacts` — remove run outputs under `experiments/runs/`

## Build Docs

```bash
mkdocs build
mkdocs serve -a 0.0.0.0:8000
```

!!! note
    Data download for training uses Hugging Face datasets and may require network access.
