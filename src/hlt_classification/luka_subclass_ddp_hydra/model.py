"""Unchanged native model and isolated auxiliary heads, including zero-weight control."""
import torch
from hlt_classification.luka_subclass_hydra.model import Student, Native, shared_parameters
from hlt_classification.luka_subclass_hydra.benchmark_training import Outputs, disable_dropout
from .contracts import cell


def new_model(spec,task):
    cell(task)
    torch.manual_seed(spec['nodes'][task]['initialization_seed'])
    return Student()
