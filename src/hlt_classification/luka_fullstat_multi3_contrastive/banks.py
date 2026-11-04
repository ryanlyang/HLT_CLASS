"""Reuse authenticated frozen embeddings; derive OFFLINE keys once in RAM."""
from pathlib import Path
import numpy as np
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullstat_u100_embedding.banks import Bank as U100Bank
from hlt_classification.luka_fullstat_multiview.targets import Bank as MultiBank
from hlt_classification.luka_fullstat_embedding_sweep.banks import Bank as OfflineBank
from .sampling import Sampler


def load(spec, train, device):
    records = spec['reuse']['banks']; banks = {}
    if train.role != 'train' or train.coordinate_name != 'D000':
        raise PermissionError('Only HLT TRAIN can consume teacher supervision')
    for name in ('U100','CONCAT','FUSION','OFFLINE'):
        item = records[name]; folder = Path(validate_file_ref(item['manifest'])).parent
        validate_file_ref(item['payload'])
        if name == 'U100': bank = U100Bank(folder,train,refs=item['refs'])
        elif name == 'OFFLINE': bank = OfflineBank(folder,train,refs=item['refs'])
        else: bank = MultiBank(folder,train,name,item['refs'],load_json(validate_file_ref(item['owner'])))
        if bank.manifest['content_hash'] != item['manifest_hash']:
            raise ValueError('Frozen bank changed')
        banks[name] = bank
    offline = banks.pop('OFFLINE')
    # Existing targets already have the exact FP64 TRAIN population transform.
    # Use the donor NumPy FP32 norm/reduction, not a numerically different kernel.
    keys = np.empty_like(offline.targets)
    for start in range(0,len(train),65536):
        x = offline.targets[start:start+65536]
        keys[start:start+len(x)] = x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    sampler = Sampler(keys,train.identities,train.labels,device=device)
    # The FullStat membership contract has canonical jet identities, no certified
    # event or alias groups. Do not invent groups from labels or source order.
    del offline
    return banks,sampler
