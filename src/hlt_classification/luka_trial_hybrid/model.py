"""Native one-forward ParT plus independent removable FP32 prediction heads."""
from hlt_classification.luka_fullstat_hybrid_embedding.model import HybridParT
import torch
from .contracts import recipe


def new_model(node):
    # FullStat donor only provides head architecture/tap; no donor training or weights.
    torch.manual_seed(node['initialization_seed'])
    model = HybridParT(node['node_id'])
    model.registered = recipe(node['arm'],node['node_id'])
    return model
