"""Sparse bipartite message-passing matcher used by PMARD variant M4."""

from __future__ import annotations

from typing import Any

from .matching import MATCH_EDGE_FEATURE_DIM, MATCH_NODE_FEATURE_DIM


def build_contextual_edge_matcher(
    *, feature_dim: int = MATCH_EDGE_FEATURE_DIM,
    node_dim: int = MATCH_NODE_FEATURE_DIM,
    hidden_dim: int = 64, message_passing_rounds: int = 3,
) -> Any:
    """Build a bounded sparse graph matcher lazily, keeping data tools torch-free."""
    if feature_dim <= 0 or node_dim <= 0 or hidden_dim <= 0 or message_passing_rounds < 2:
        raise ValueError("matcher dimensions/rounds differ from the sparse contextual contract")
    import torch
    from torch import nn

    def block(input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim), nn.GELU(), nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim), nn.GELU(),
        )

    class SparseBipartiteMatcher(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hlt_encoder = block(node_dim, hidden_dim)
            self.offline_encoder = block(node_dim, hidden_dim)
            self.edge_encoder = block(feature_dim, hidden_dim)
            self.hlt_messages = nn.ModuleList(
                block(2 * hidden_dim, hidden_dim) for _ in range(message_passing_rounds)
            )
            self.offline_messages = nn.ModuleList(
                block(2 * hidden_dim, hidden_dim) for _ in range(message_passing_rounds)
            )
            self.hlt_updates = nn.ModuleList(
                block(2 * hidden_dim, hidden_dim) for _ in range(message_passing_rounds)
            )
            self.offline_updates = nn.ModuleList(
                block(2 * hidden_dim, hidden_dim) for _ in range(message_passing_rounds)
            )
            self.edge_updates = nn.ModuleList(
                block(3 * hidden_dim, hidden_dim) for _ in range(message_passing_rounds)
            )
            self.score = nn.Sequential(
                nn.Linear(3 * hidden_dim, 2 * hidden_dim), nn.GELU(),
                nn.LayerNorm(2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(), nn.Linear(hidden_dim, 1),
            )

        @staticmethod
        def aggregate(messages: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
            summed = messages.new_zeros((size, messages.shape[-1]))
            counts = messages.new_zeros((size, 1))
            summed.index_add_(0, index, messages)
            counts.index_add_(0, index, messages.new_ones((len(messages), 1)))
            return summed / counts.clamp_min(1)

        def forward(
            self, edge_features: torch.Tensor, hlt_index: torch.Tensor,
            offline_index: torch.Tensor, hlt_node_features: torch.Tensor | None = None,
            offline_node_features: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if edge_features.ndim != 2 or edge_features.shape[1] != feature_dim:
                raise ValueError("edge features differ from sparse matcher contract")
            if hlt_index.shape != (len(edge_features),) or offline_index.shape != (len(edge_features),):
                raise ValueError("contextual matcher edge-index shape differs")
            hlt_count = (
                len(hlt_node_features) if hlt_node_features is not None
                else int(hlt_index.max().item()) + 1 if len(hlt_index) else 0
            )
            offline_count = (
                len(offline_node_features) if offline_node_features is not None
                else int(offline_index.max().item()) + 1 if len(offline_index) else 0
            )
            if hlt_node_features is None:
                hlt_node_features = edge_features.new_zeros((hlt_count, node_dim))
            if offline_node_features is None:
                offline_node_features = edge_features.new_zeros((offline_count, node_dim))
            if hlt_node_features.shape != (hlt_count, node_dim):
                raise ValueError("HLT node features differ from sparse matcher contract")
            if offline_node_features.shape != (offline_count, node_dim):
                raise ValueError("offline node features differ from sparse matcher contract")
            hlt = self.hlt_encoder(hlt_node_features)
            offline = self.offline_encoder(offline_node_features)
            edge = self.edge_encoder(edge_features)
            for h_message, o_message, h_update, o_update, e_update in zip(
                self.hlt_messages, self.offline_messages, self.hlt_updates,
                self.offline_updates, self.edge_updates, strict=True,
            ):
                to_hlt = h_message(torch.cat((edge, offline[offline_index]), dim=-1))
                to_offline = o_message(torch.cat((edge, hlt[hlt_index]), dim=-1))
                hlt = hlt + h_update(torch.cat((
                    hlt, self.aggregate(to_hlt, hlt_index, hlt_count),
                ), dim=-1))
                offline = offline + o_update(torch.cat((
                    offline, self.aggregate(to_offline, offline_index, offline_count),
                ), dim=-1))
                edge = edge + e_update(torch.cat((
                    edge, hlt[hlt_index], offline[offline_index],
                ), dim=-1))
            return self.score(torch.cat((
                edge, hlt[hlt_index], offline[offline_index],
            ), dim=-1)).squeeze(-1)

    return SparseBipartiteMatcher()


__all__ = ["build_contextual_edge_matcher"]
