"""Durable full-gated epoch evidence; no automatic resume."""
from pathlib import Path
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import write_immutable_json, load_json
from hlt_classification.luka_fullstat_attention_fast.checkpoints import rng_state
from .contracts import artifact, validate


class EpochStore:
    def __init__(self, root, *, binding):
        self.root, self.binding = Path(root), binding
        self.root.mkdir(parents=True, exist_ok=False)

    def save(self, state):
        epoch = state['epoch']
        if not 1 <= epoch <= 20 or len(state['history']) != epoch:
            raise ValueError('Incomplete epoch')
        path = self.root/f'epoch-{epoch:02d}.pt'
        save_state(path, state)
        receipt = artifact('EPOCH', parents=self.binding, epoch=epoch, updates=state['updates'],
                           payload=file_ref(path))
        write_immutable_json(self.root/f'epoch-{epoch:02d}.json', receipt)


def check(path, binding):
    row = load_json(path)
    validate(row, 'EPOCH', binding)
    validate_file_ref(row['payload'])
    return row
