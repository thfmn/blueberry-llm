from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from experiments.compute import (
    ComputeBudget,
    d_ff_expert_for_fixed_active,
    d_ff_expert_for_fixed_total_params,
)
from experiments.plotting import (
    plot_A1_perplexity_vs_experts,
    plot_A2_learning_curves,
    plot_A3_active_params_vs_ppl,
    plot_A4_throughput,
    plot_A5_expert_usage_histogram,
    plot_A6_router_entropy,
    plot_B1_params_regime,
    plot_C1_ppl_vs_flops,
    plot_B2_ppl_vs_flops_params,
    plot_B3_flops_vs_experts_params,
    plot_C2_ppl_vs_total_params,
    plot_C3_heatmap_ppl_E_dff,
    plot_D1_peak_mem_vs_experts,
    plot_D2_seconds_per_mtokens,
)


def _dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _mock_final_df(E_list: List[int], seeds: List[int]) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    d_model = 384
    top_k = 2

    # Fixed FLOPs regime: keep top_k * d_ff_e constant
    d_ff_active = 1024
    d_ff_e_fixed = d_ff_expert_for_fixed_active(d_ff_active, top_k)

    # Fixed total params regime baseline
    e_ref, d_ff_ref = 4, 1024

    rows = []
    for regime in ["flops", "params"]:
        for E in E_list:
            # Choose d_ff_expert per regime
            if regime == "flops":
                d_ff_e = d_ff_e_fixed
            else:
                d_ff_e = d_ff_expert_for_fixed_total_params(d_ff_ref, e_ref, E)

            budget = ComputeBudget(
                d_model=d_model, top_k=top_k, d_ff_expert=d_ff_e, num_experts=E
            )

            # Tokens/sec: mild overhead as E grows
            base_tps = 3500.0
            tps = base_tps * (1.0 - 0.05 * np.log2(max(E, 1))) + rng.normal(0, 80)
            tps = max(500.0, tps)

            # Peak memory grows slightly with E
            peak_mem_gb = 5.0 + 0.15 * np.log2(max(E, 1)) + abs(rng.normal(0, 0.2))
            peak_mem_bytes = int(peak_mem_gb * (1024**3))

            # Router entropy (mock): higher with more experts, noisy
            router_entropy_mean = max(
                0.1, 0.4 + 0.1 * np.log2(max(E, 1)) + rng.normal(0, 0.05)
            )

            # Expert usage distribution per E (Dirichlet)
            usage = rng.dirichlet(np.ones(E)).tolist()

            # Perplexity: improve slightly with more experts under fixed FLOPs
            for seed in seeds:
                # Base depends on regime; params regime may be flatter
                if regime == "flops":
                    base = 20.0 - 0.9 * np.log2(max(E, 1))
                else:
                    base = 21.0 - 0.3 * np.log2(max(E, 1))
                val_ppl = max(5.0, base + rng.normal(0, 0.5))

                rows.append(
                    {
                        "regime": regime,
                        "num_experts": E,
                        "seed": seed,
                        "d_model": d_model,
                        "top_k": top_k,
                        "d_ff": d_ff_e,
                        "tokens_per_second": tps,
                        "router_entropy_mean": router_entropy_mean,
                        "val_perplexity": val_ppl,
                        "tokens_seen": 1_000_000,  # final snapshot
                        "peak_mem_bytes": peak_mem_bytes,
                        "router_usage_frac": usage,
                        **budget.as_dict(),
                    }
                )

    return pd.DataFrame(rows)


def _mock_learning_curves(E_list: List[int]) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    d_model = 384
    top_k = 2
    d_ff_active = 1024
    d_ff_e = d_ff_expert_for_fixed_active(d_ff_active, top_k)

    rows = []
    for E in E_list:
        budget = ComputeBudget(
            d_model=d_model, top_k=top_k, d_ff_expert=d_ff_e, num_experts=E
        )
        # More experts learn slightly faster at same compute in this mock
        floor = 7.0 + 1.0 / max(1, np.log2(E))
        start = 50.0 - 1.5 * np.log2(max(E, 1))
        for t in np.linspace(50_000, 1_000_000, 12, dtype=int):
            # Simple exponential decay towards floor
            frac = (1 - (t / 1_000_000.0))
            ppl = floor + (start - floor) * max(0.0, frac) + rng.normal(0, 0.6)
            rows.append(
                {
                    "regime": "flops",
                    "num_experts": E,
                    "seed": 0,
                    "d_model": d_model,
                    "top_k": top_k,
                    "d_ff": d_ff_e,
                    "tokens_per_second": np.nan,
                    "router_entropy_mean": np.nan,
                    "val_perplexity": max(5.0, ppl),
                    "tokens_seen": int(t),
                    "peak_mem_bytes": np.nan,
                    "router_usage_frac": None,
                    **budget.as_dict(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate mock Seaborn plots for docs")
    ap.add_argument(
        "--out", type=str, default="docs/assets/plots", help="Output directory for images"
    )
    args = ap.parse_args()

    out_dir = _dir(Path(args.out))

    E_list = [1, 2, 4, 8, 16, 32]
    seeds = [0, 1, 2]

    df_final = _mock_final_df(E_list, seeds)
    df_learning = _mock_learning_curves(E_list)

    # Combined DF used for plots that don't care which subset produced the data
    df_all = pd.concat([df_final, df_learning], ignore_index=True)

    # Generate all named plots into docs/assets/plots
    plot_A1_perplexity_vs_experts(df_final, out_dir)
    plot_A2_learning_curves(df_learning, out_dir)
    plot_A3_active_params_vs_ppl(df_final, out_dir)
    plot_A4_throughput(df_final, out_dir)
    plot_A5_expert_usage_histogram(df_final, out_dir)
    plot_A6_router_entropy(df_final, out_dir)
    plot_B1_params_regime(df_all, out_dir)
    plot_C1_ppl_vs_flops(df_all, out_dir)
    plot_B2_ppl_vs_flops_params(df_all, out_dir)
    plot_B3_flops_vs_experts_params(df_all, out_dir)
    plot_C2_ppl_vs_total_params(df_final, out_dir)
    plot_C3_heatmap_ppl_E_dff(df_final, out_dir)
    plot_D1_peak_mem_vs_experts(df_final, out_dir)
    plot_D2_seconds_per_mtokens(df_final, out_dir)


if __name__ == "__main__":
    main()

