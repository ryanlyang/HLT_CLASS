"""The unchanged original native student and prediction head."""
import torch
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer as Native
from .contracts import cell


def new_model(spec, task):
    cell(task)
    torch.manual_seed(spec['nodes'][task]['initialization_seed'])
    return AuxiliaryParT()


def shared_parameters(model):
    return [(n, p) for n, p in model.named_parameters()
            if p.requires_grad and n.startswith('mod.') and not n.startswith('mod.fc.')]
