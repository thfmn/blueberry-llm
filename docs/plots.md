# Plots (Named)

### Plot A1 — Perplexity vs Experts (Fixed FLOPs)
- y: validation perplexity
- x: `E`
- hue: seed (ribbon optional)

![A1: Perplexity vs Experts](assets/plots/A1_ppl_vs_experts.png)

### Plot A2 — Learning Curves @Fixed FLOPs
- y: validation perplexity
- x: tokens seen
- hue: `E`

![A2: Learning Curves](assets/plots/A2_learning_curves.png)

### Plot A4 — Throughput vs Experts (Fixed FLOPs)
- y: tokens/s
- x: `E`

![A4: Throughput vs Experts](assets/plots/A4_throughput_vs_experts.png)

### Plot A6 — Router Entropy vs Experts
- y: mean router entropy (per step)
- x: `E`

![A6: Router Entropy vs Experts](assets/plots/A6_router_entropy_vs_experts.png)

### Plot B1 — Perplexity vs Experts (Fixed Params)
- y: validation perplexity
- x: `E`

![B1: PPL vs Experts (Params)](assets/plots/B1_ppl_vs_experts_fixed_params.png)

### Plot C1 — Perplexity vs FLOPs/Token (All Regimes)
- y: validation perplexity
- x: forward FLOPs/token proxy
- hue: `E`
- style: regime (`flops` vs `params`)

![C1: PPL vs FLOPs/Token](assets/plots/C1_ppl_vs_flops.png)

!!! info "Saved to"
    During development, mock plots are generated into `docs/assets/plots/`.
### Plot A3 — PPL vs Active Params/Token (Sanity)
- y: validation perplexity
- x: active params per token (proxy)

![A3: PPL vs Active Params/Token](assets/plots/A3_ppl_vs_active_params.png)

### Plot A5 — Expert Usage Histogram
- Per `E`, average fraction of tokens per expert (bars)

Samples:

![A5: Expert Usage E=8](assets/plots/A5_expert_usage_E8.png)
![A5: Expert Usage E=32](assets/plots/A5_expert_usage_E32.png)

### Plot A7 — Load-Balance Loss (avg)
- y: average auxiliary loss during training
- x: `E`

### Plot B2 — PPL vs FLOPs/Token (Iso-Params)
- y: validation perplexity
- x: forward FLOPs/token (proxy)

![B2: PPL vs FLOPs/Token (Params)](assets/plots/B2_ppl_vs_flops_params.png)

### Plot B3 — FLOPs/Token vs Experts (Fixed Params)
- y: forward FLOPs/token (proxy)
- x: `E`

![B3: FLOPs/Token vs Experts (Params)](assets/plots/B3_flops_vs_experts_params.png)

### Plot C2 — Perplexity vs Total Expert Params
- y: validation perplexity
- x: total expert parameters

![C2: PPL vs Total Expert Params](assets/plots/C2_ppl_vs_total_params.png)

### Plot C3 — Heatmap: PPL over (E, d_ff_expert)
- 2D pivot of mean PPL; contours implicit

![C3: Heatmap](assets/plots/C3_heatmap_ppl_E_dff.png)

### Plot D1 — Peak Memory vs Experts
- y: GB
- x: `E`

![D1: Peak Memory vs Experts](assets/plots/D1_peak_mem_vs_experts.png)

### Plot D2 — sec / 1M tokens vs Experts
- y: seconds per million tokens
- x: `E`

![D2: sec / 1M tokens vs Experts](assets/plots/D2_sec_per_1M_tokens_vs_experts.png)

---

#### Re-generate Mock Plots

- Generate: `python experiments/generate_mock_plots.py --out docs/assets/plots`
- Theme: plots use Seaborn with `sns.set_theme(style="whitegrid")` for consistency.
