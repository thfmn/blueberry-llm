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
  --wandb-project research-experiments --tag debug
```

Each run writes structured deliverables into `experiments/runs/<id>/`, including:

- `artifacts/metrics.json`, `train_history.json`, and `training_curve.png`
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

### Scaling sweeps with the CLI

```bash
python -m blueberry_cli grid \
  --regime flops \
  --experts 1,2,4,8,16 \
  --top-k 2 \
  --max-steps 200 --batch-size 16 \
  --wandb-project research-experiments --tag scaling
```

The grid command maintains the classic CSV log (`results_<regime>.csv`), emits per-configuration deliverables, and produces aggregated summaries (`grid_summary.json`, `grid_aggregate.json`).

## Build Docs

```bash
mkdocs build
mkdocs serve -a 0.0.0.0:8000
```

!!! note
    Data download for training uses Hugging Face datasets and may require network access.
