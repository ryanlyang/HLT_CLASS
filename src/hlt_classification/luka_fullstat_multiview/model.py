"""Reuse the SPORC graphs with a separately registered total embedding weight."""
import torch
from torch.nn import functional as F
from hlt_classification.luka_u100_multiview import teachers
from hlt_classification.luka_u100_multiview.model import Student
from .contracts import TEACHERS, TASKS


def teacher_name(task):
    if task not in TASKS: raise PermissionError('Unregistered multiview task')
    return task.removeprefix('teacher_') if task.startswith('teacher_') else None


def new_model(node, task):
    name = teacher_name(task)
    if name: return teachers.new_teacher(name, node)
    torch.manual_seed(node['initialization_seed'])
    return Student('u100_concat_fusion')


def loss_for(net, task, raw, indices, banks, device):
    name = teacher_name(task)
    if set(banks) != (set() if name else set(TEACHERS)):
        raise PermissionError('Wrong supervision inventory')
    logits = teachers.forward(net, raw, device, bf16=torch.device(device).type == 'cuda', name=name)
    parts = {'ce': F.cross_entropy(logits.float(), torch.from_numpy(raw['labels']).to(device))}
    loss = parts['ce']
    for key in TEACHERS if not name else ():
        target = torch.from_numpy(banks[key].targets[indices]).to(device)
        predicted = net.predictions[key]
        if target.shape != predicted.shape or target.shape != (len(indices),128) or not torch.isfinite(target).all():
            raise ValueError('Invalid embedding targets')
        parts[key] = F.smooth_l1_loss(predicted.float(), target.detach().float(), beta=1, reduction='mean')
        loss = loss + (10./3)*parts[key]
    if not torch.isfinite(loss): raise ValueError('Nonfinite required objective')
    return loss, parts


def deploy(net, task):
    return net.eval() if teacher_name(task) else net.deployed()
