"""Local contextual edge scorer used by PMARD matcher variant M4."""

from __future__ import annotations

from typing import Any


def build_contextual_edge_matcher(*, feature_dim: int = 13, hidden_dim: int = 64) -> Any:
    """Build a bounded local scorer lazily, keeping data-only tools torch-free."""
    if feature_dim <= 0 or hidden_dim <= 0:
        raise ValueError("matcher dimensions must be positive")
    import torch
    from torch import nn

    class ContextualEdgeMatcher(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.edge_encoder = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim), nn.GELU(),
                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            )
            self.contextual_score = nn.Sequential(
                nn.Linear(3 * hidden_dim, 2 * hidden_dim), nn.GELU(),
                nn.LayerNorm(2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(
            self, edge_features: torch.Tensor, hlt_index: torch.Tensor,
            offline_index: torch.Tensor,
        ) -> torch.Tensor:
            if edge_features.ndim != 2 or edge_features.shape[1] != feature_dim:
                raise ValueError("edge features differ from matcher contract")
            if hlt_index.shape != (len(edge_features),) or offline_index.shape != (len(edge_features),):
                raise ValueError("contextual matcher edge-index shape differs")
            encoded = self.edge_encoder(edge_features)
            def aggregate(index: torch.Tensor) -> torch.Tensor:
                size = int(index.max().item()) + 1 if len(index) else 0
                summed = encoded.new_zeros((size, hidden_dim)); counts = encoded.new_zeros((size, 1))
                summed.index_add_(0, index, encoded)
                counts.index_add_(0, index, encoded.new_ones((len(index), 1)))
                return summed / counts.clamp_min(1)
            hlt_context = aggregate(hlt_index)[hlt_index]
            offline_context = aggregate(offline_index)[offline_index]
            return self.contextual_score(
                torch.cat((encoded, hlt_context, offline_context), dim=-1)
            ).squeeze(-1)

    return ContextualEdgeMatcher()


__all__ = ["build_contextual_edge_matcher"]
