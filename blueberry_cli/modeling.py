from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from experiments.compute import ComputeBudget
from llm import MoEModelConfig, MoEMinimalLLM


@dataclass
class ParameterEstimates:
    total_params: int
    trainable_params: int
    active_params_per_token: int
    total_expert_params: int
    approx_dense_params: Optional[int] = None

    @property
    def total_params_m(self) -> float:
        return self.total_params / 1_000_000

    @property
    def trainable_params_m(self) -> float:
        return self.trainable_params / 1_000_000


def _approximate_dense_params(config: MoEModelConfig) -> int:
    # Rough transformer dense parameter estimate (ignoring embeddings to simplify)
    d_model = config.d_model
    n_layers = config.n_layers
    vocab = config.vocab_size or 32000
    attn_params = n_layers * (4 * d_model * d_model)
    ffn_params = n_layers * (2 * d_model * config.d_ff)
    embedding_params = d_model * vocab
    return attn_params + ffn_params + embedding_params


def estimate_parameter_count(config: MoEModelConfig) -> ParameterEstimates:
    budget = ComputeBudget(
        d_model=config.d_model,
        top_k=config.expert_top_k,
        d_ff_expert=config.d_ff,
        num_experts=config.num_experts,
    )
    active = budget.active_params_per_token
    expert_total = budget.total_expert_params

    try:
        model = MoEMinimalLLM(config)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    except Exception:
        total = _approximate_dense_params(config) + expert_total
        trainable = total
    finally:
        # Free memory if we instantiated the model
        if "model" in locals():
            del model
            if torch.cuda.is_available():  # pragma: no cover - skipped on CPU tests
                torch.cuda.empty_cache()

    dense_estimate = _approximate_dense_params(config)
    return ParameterEstimates(
        total_params=total,
        trainable_params=trainable,
        active_params_per_token=active,
        total_expert_params=expert_total,
        approx_dense_params=dense_estimate,
    )
