"""Reuse the declared retained graph; never strip its predictor at deployment."""
import torch
from hlt_classification.luka_u100_multiview.model import Student, objective
from hlt_classification.jetclass2_delphes.runner import _forward
from .contracts import ARMS, check_node


def new_model(node, task):
    check_node(node, task)
    torch.manual_seed(node['initialization_seed'])
    return Student(ARMS[task])


def batch_loss(student, raw, bank, indices, task, device):
    if task not in ARMS or student.arm!=ARMS[task]: raise ValueError('Wrong retained arm')
    if (task=='train_retained')!=(bank is not None): raise PermissionError('Wrong teacher target access')
    logits=_forward(student, raw, device, bf16=torch.device(device).type=='cuda')
    labels=torch.from_numpy(raw['labels']).to(device)
    targets={} if bank is None else {'U100':torch.from_numpy(bank.targets[indices]).to(device)}
    loss,parts=objective(ARMS[task],logits,labels,student.predictions,targets,pass_coordinate=0.)
    return loss,parts


def load_deployed(node, task, state, device):
    value=new_model(node,task)
    # A native-only checkpoint is invalid for this architecture.
    value.load_state_dict(state,strict=True)
    value.capture=False
    return value.to(device).eval()
