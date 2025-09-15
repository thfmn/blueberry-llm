# Research Tutorial — MoE Scaling Study

This is a practical, mentor-style guide for running a small but credible AI research project on Mixture-of-Experts (MoE) scaling using this repository. Follow the steps, keep runs reproducible, and resist scope creep.

Project recap (what we’re studying)
- Question: Do more, narrower experts beat fewer, wider experts at the same compute or parameter budgets?
- Regimes: Fixed FLOPs (primary) and Fixed Params (secondary).
- Metrics: validation perplexity, throughput, router entropy/usage, memory.
- Artifacts: CSV logs + seaborn plots A1–D2 (see Plots page).

Phase 1 — Define the Study (1–2 hours)
- Hypothesis: Write a one-sentence claim to validate or refute (sharp and falsifiable).
- Constraints: Pick token budget, hardware, and time per run (e.g., 200–2,000 steps).
- Fairness: Choose compute proxies (active params/token; FLOPs/token) and param proxy (E × d_ff_expert).
- Grid: E ∈ {1,2,4,8,16,32}, top_k=2; pick d_model, n_layers, batch/seq.

Phase 2 — Pilot Runs (2–4 hours)
- Goal: Verify end-to-end stability, logging, and routing sanity with tiny configs.
- Actions:
  - Run 2–3 quick points (E=1,4,16) with short steps.
  - Confirm CSV contains proxies and metrics; check that plots render.
  - Sanity: PPL decreases during training; router entropy and usage are finite and non-degenerate.

Phase 3 — Full Grid (1–2 days depending on hardware)
- Fixed FLOPs sweep:
  - `python experiments/run_scaling.py --regime flops --experts 1,2,4,8,16,32 --top-k 2 --out-dir experiments/out --seeds 3 [other config]`
  - Verify each run logs rows; resume if interrupted.
- Fixed Params sweep:
  - `python experiments/run_scaling.py --regime params --experts 1,2,4,8,16,32 --out-dir experiments/out --seeds 3 [other config]`
- Combine/clean CSVs if needed (consistent columns, dtypes).

Phase 4 — Plot and Inspect (1–2 hours)
- Generate plots from real CSVs:
  - `python experiments/plotting.py --csv experiments/out/results_flops.csv --out experiments/plots`
- For documentation demos:
  - `python -m experiments.generate_mock_plots --out docs/assets/plots`
- Visual checks: monotonicity trends, variance across seeds, router usage shape.

Phase 5 — Stress and Ablate (optional, 0.5–1 day)
- Adjust: top_k, aux loss weight, router temperature; re-run a small subset.
- Scale sensitivity: try +/− 2× steps or data; test extreme E.
- Record negatives: cases where MoE underperforms or destabilizes.

Phase 6 — Synthesize and Write (0.5–1 day)
- Draft 3–4 validated claims tied to specific figures (A1–D2 mapping).
- Keep causal language conservative; emphasize matched budgets and proxies.
- Fill limitations and practical guidance (when MoE helps; when not worth it).

Checklists
- Reproducibility
  - [ ] Global seed set exactly once; deterministic ops where possible.
  - [ ] Exact configs saved (JSON/YAML or CLI string).
  - [ ] Hardware logged (GPU count, memory, CUDA/cuDNN versions).
  - [ ] Dataset identifiers and splits documented.
- Data/Compute sanity
  - [ ] FLOPs/token proxy constant within each fixed-FLOPs subset.
  - [ ] Total expert params constant within fixed-params subset.
  - [ ] Throughput vs E trend plausible; memory within limits.
- Plots
  - [ ] A1–D2 render without NaNs; axes and titles correct.
  - [ ] Seed aggregation clear (mean ± CI or all seeds visible).

Common pitfalls (and fixes)
- Apples-to-oranges comparisons: enforce regime constraints in code; spot-check rows.
- Router collapse (one expert dominates): add/strengthen load-balance loss; tune router temperature.
- Unstable training: lower LR, warmup more, reduce batch/seq, clip grads.
- Misleading proxies: report both active params/token and total expert params; show both regimes.

Commands cheat sheet
- Auto-setup: `chmod +x setup.sh && ./setup.sh`
- Hardware config preview: `python auto_config.py`
- Single device train: `python train_auto.py`
- Multi-GPU: `torchrun --nproc_per_node=N train_auto.py`
- Inference sanity: `python inference.py --prompt "Hello" --checkpoint blueberry_model.pt`
- Monitor GPUs: `python gpu_monitor.py`

Paper scaffolding
- Use `draft/draft.md` as the sharpie outline; link figures after you generate real plots.
- After results stabilize, port sections to Overleaf and replace placeholders.

Milestones and timeboxing (suggested)
- Day 1: Pilot runs + mock plots + finalize grid.
- Days 2–3: Full grid sweeps; monitor and pre-plot partial results.
- Day 4: Ablations; finalize plots; draft claims and limitations.
- Day 5: Write pass; reproducibility appendix; final formatting.

What “good” looks like
- Claims are directly supported by matched-budget findings (A1 primary).
- Figures are legible and tie to concrete guidance (how to choose E given compute).
- Reproducibility is turnkey: one command regenerates plots from CSVs.

