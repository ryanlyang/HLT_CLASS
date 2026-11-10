"""Native views and exact selected TRAIN supervision only."""
from pathlib import Path
import torch
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_teacher_sweep_hydra.inputs import views,delta,Bundle
from hlt_classification.luka_embedding_aux.targets import Bank
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from .contracts import TEACHERS,FOLDS,STUDENTS
from . import folds


def fit_views(spec,task):
    if task in FOLDS: return delta(spec).load(spec,'train'),None
    return views(spec,task in TEACHERS)


def banks(spec,task,train):
    from .worker import validate_teacher
    result={}
    for teacher in STUDENTS.get(task,()):
        validate_teacher(spec,teacher); folder=Path(spec['root'])/teacher
        result[teacher]=Bank(folder/'bank',train,teacher_ref=file_ref(folder/'fit/selected.pt'))
    return result


def context(spec,task,train,device):
    from .worker import validate_teacher
    if task=='TE':
        validate_teacher(spec,'T0'); ref=file_ref(Path(spec['root'])/'T0/fit/selected.pt')
        with torch.random.fork_rng(devices=[]): model=DelphesParticleTransformer().to(device).eval()
        model.load_state_dict(torch.load(ref['path'],map_location='cpu',weights_only=True),strict=True)
        for p in model.parameters(): p.requires_grad_(False)
        return dict(reference=model,parents=dict(reference=ref['sha256']))
    if task=='TF':
        weights,manifest=folds.assemble(spec,train)
        return dict(weights=weights,manifest=manifest,parents=dict(difficulty=manifest['content_hash']))
    if task in FOLDS: return dict(parents=dict(folds=folds.load(spec)[0]['content_hash']))
    return {}
