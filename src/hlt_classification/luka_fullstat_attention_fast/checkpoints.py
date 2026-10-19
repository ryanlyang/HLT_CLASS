"""Immutable, checksum-bound epoch commits. No partial epoch is resumable."""
from pathlib import Path
import math
import random
import numpy as np
import torch
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from .contracts import artifact, validate
from hlt_classification.jetclass2_delphes.runner import selection_key


def rng_state():
    ns = np.random.get_state()
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        python=random.getstate(), numpy=[ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]])


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume CUDA device count differs')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    random.setstate(state['python'])
    ns = state['numpy']
    np.random.set_state((ns[0], np.asarray(ns[1], np.uint32), ns[2], ns[3], ns[4]))


class EpochStore:
    def __init__(self, root, *, binding):
        self.root, self.binding = Path(root), binding
        self.root.mkdir(parents=True, exist_ok=False)

    def save(self, state):
        epoch = state['epoch']
        path = self.root/f'epoch-{epoch:02d}.pt'
        save_state(path, state)
        # Commit marker published last; an orphan .pt is never accepted.
        receipt = artifact('EPOCH', parents=self.binding, epoch=epoch,
                           payload=file_ref(path), updates=state['updates'])
        write_immutable_json(self.root/f'epoch-{epoch:02d}.json', receipt)
        print(f'FAST durable checkpoint: epoch={epoch} receipt={self.root / (f"epoch-{epoch:02d}.json")}', flush=True)
        return receipt


def load(path, *, binding):
    receipt = load_json(path)
    validate(receipt, 'EPOCH', binding)
    payload = validate_file_ref(receipt['payload'])
    state = torch.load(payload, map_location='cpu', weights_only=True)
    required={'epoch','updates','history','selected','model','best_state','optimizer','rng','runtime_seconds'}
    if (set(state)!=required or not 1 <= receipt['epoch'] <= 20 or state['epoch'] != receipt['epoch']
            or state['updates'] != receipt['updates'] or len(state['history']) != state['epoch']
            or [r['pass_number'] for r in state['history']] != list(range(1, state['epoch']+1))
            or state['updates'] != state['history'][-1]['update']
            or type(state['selected']) is not int or not 1<=state['selected']<=state['epoch']
            or not math.isfinite(state['runtime_seconds']) or state['runtime_seconds']<0):
        raise ValueError('Checkpoint epoch/history differs')
    best=max(state['history'],key=lambda row:selection_key(row['validation'],row['update']))
    if best['pass_number']!=state['selected']:
        raise ValueError('Checkpoint selected epoch differs from the registered selection rule')
    return state
