"""Allowlisted original inputs; U100 particle features are TRAIN-only here."""
from pathlib import Path
import torch
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_cross_message_hydra import inputs as donor
from hlt_classification.luka_cross_message_hydra.training import deployment
from hlt_classification.luka_cross_message_hydra.contracts import settings


def load_hlt(spec,role):
    if role not in ('train','validation'): raise PermissionError('Final test is sealed')
    return donor.load_view(spec,role,'D000')


def tokens(spec,train,device='cuda'):
    if train.role!='train': raise PermissionError('Shared-message tokens are TRAIN-only')
    return donor.oracle_cache(spec,train,device)


def bank(spec,train):
    if train.role!='train': raise PermissionError('Embedding supervision is TRAIN-only')
    return donor.embedding_bank(spec,train)


def frozen_teacher(spec,device='cuda'):
    proof=spec['teacher_proof']
    for ref in proof['files']: validate_file_ref(ref)
    checkpoint=next(r['path'] for r in proof['files'] if Path(r['path']).name=='selected.pt')
    report=load_json(Path(checkpoint).parent/'training_report.json')
    if report['content_hash']!=proof['report_hash']: raise ValueError('Frozen teacher report differs')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    return deployment(settings('teacher_m16'),state,device).requires_grad_(False).eval()
