# Theory — Transformers and Mixture-of-Experts

This page summarizes the core mathematical ideas behind Transformers and sparse Mixture-of-Experts (MoE), with notation aligned to this project. Math is rendered with LaTeX.

## Transformer Block (brief)

Let token embeddings be `x \in \mathbb{R}^{d_\text{model}}`. A standard Transformer block alternates self-attention and a feed-forward network (FFN), each with residual connections and normalization.

Feed-forward sublayer (dense):

$$
\mathrm{FFN}(x) = W_2\,\sigma(W_1 x + b_1) + b_2, \quad W_1 \in \mathbb{R}^{d_\text{ff} \times d_\text{model}},\; W_2 \in \mathbb{R}^{d_\text{model} \times d_\text{ff}}.
$$

Per-token multiply-adds for FFN are roughly:

$$
\text{FLOPs}_\text{FFN/token} \approx 2\, d_\text{model} \cdot d_\text{ff}.
$$

Self-attention cost scales as `O(L^2 d_model)` per layer for sequence length `L`; in our scaling comparisons, attention cost is held constant across grids.

## Sparse Mixture-of-Experts FFN

An MoE layer replaces the dense FFN with a set of experts `\{f_i\}_{i=1}^E`, each an FFN of width `d_{\text{ff, expert}}`. A router (gating network) selects `\text{top\_k}` experts per token.

Router logits and probabilities:

$$
z = W_g x \in \mathbb{R}^E, \qquad p = \mathrm{softmax}(z) \in \Delta^{E-1}.
$$

Noisy Top-\(k\) gating (optional) adds Gaussian noise to `z` before selection to encourage exploration. The token output is the weighted sum over the selected experts:

$$
\mathrm{MoE}(x) = \sum_{i\in \text{TopK}(p)} \tilde p_i\, f_i(x), \qquad \tilde p_i = \frac{p_i}{\sum_{j\in \text{TopK}(p)} p_j}.
$$

Assuming all experts share width `d_{\text{ff, expert}}`, the active FFN compute per token is approximately

$$
\text{FLOPs}_\text{MoE/token} \approx 2\, d_\text{model} \cdot (\text{top\_k} \cdot d_{\text{ff, expert}}),
$$

which motivates our fixed-FLOPs regime: hold `\text{top\_k} \cdot d_{\text{ff, expert}}` constant when varying the number of experts `E`.

Total expert parameters across the layer are

$$
\text{Params}_\text{experts} \approx 2\, d_\text{model} \cdot (E\, d_{\text{ff, expert}}),
$$

used for the fixed-parameter regime by keeping `E\, d_{\text{ff, expert}}` constant.

## Load-Balancing and Capacity

Without regularization, routers can collapse, overloading a few experts. Switch Transformers and GShard introduce an auxiliary loss to encourage balanced routing. Define, over a batch, for expert `i`:

$$
f_i = \frac{\text{tokens routed to expert } i}{\text{total tokens}}, \qquad P_i = \text{mean router probability for expert } i.
$$

A common auxiliary (Switch) is

$$
\mathcal{L}_\text{aux} = E \cdot \sum_{i=1}^E f_i \, P_i, \quad \text{minimize}.
$$

Routing usually enforces a per-expert capacity constraint to bound memory/latency. For batch size `B` and sequence length `L`, a typical capacity is

$$
\text{cap} = \left\lceil \frac{\alpha \cdot B L \cdot \text{top\_k}}{E} \right\rceil,
$$

with capacity factor `\alpha \ge 1`. Excess tokens can be dropped or rerouted, trading off quality vs. throughput.

## Why MoE Helps

- Conditional computation: only `\text{top\_k}` experts run per token, increasing model capacity (parameters) at similar per-token compute.
- Specialization: experts can specialize on input submanifolds (languages, styles, topics), empirically improving perplexity at scale.
- System efficiency: with well-tuned routing and capacity, MoE can achieve favorable quality/compute trade-offs versus dense scaling.

## Practical Considerations

- Initialization and stability: warmup, gradient clipping, and conservative LR are helpful when increasing `E`.
- Router temperature/noise: tune to avoid expert collapse; monitor entropy and usage histograms.
- Communication: distributed MoE introduces all-to-all token exchanges; performance hinges on overlap and partitioning.

## Further Reading (selected)

- Vaswani et al., “Attention Is All You Need” (2017) — Transformer foundation.
- Shazeer et al., “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer” (2017).
- Lepikhin et al., “GShard: Scaling Giant Models with Conditional Computation” (2020).
- Fedus, Zoph, Shazeer, “Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity” (2021).
- Du et al., “GLaM: Efficient Scaling of Language Models with Mixture-of-Experts” (2022).

See References for links and more.

