"""Component hashes and independent random-stream identities, without consuming RNG."""
import hashlib
import re
import numpy as np
from pathlib import Path
import torch
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from .contracts import artifact,validate,cell,seeds


def tensor_hash(state):
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        a=tensor.detach().cpu().contiguous().numpy()
        digest.update(repr((name,a.dtype.str,a.shape)).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def order_hash(spec,task):
    order=np.random.default_rng(np.random.SeedSequence([seeds(spec,task)['sampler'],1])).permutation(spec['counts']['train'])
    return hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()


def initial_record(spec,task,model,world):
    state=model.state_dict()
    components={name:tensor_hash({k:v for k,v in state.items() if k.startswith(prefix)})
                for name,prefix in (('native','mod.'),('predictor','predictor.'),('decoder','aux.'))}
    return artifact('INITIALIZATION',parents=dict(spec=spec['content_hash']),task=task,
        seeds=seeds(spec,task),components=components,world_size=world,
        dropout_rank_seeds=[seeds(spec,task)['dropout_base']+r for r in range(world)],
        epoch1_order_hash=order_hash(spec,task),decoder_present=not cell(task)['control'])


def validate_initial(spec,task,value,world):
    validate(value,'INITIALIZATION',dict(spec=spec['content_hash']))
    if (value['task']!=task or value['seeds']!=seeds(spec,task) or value['world_size']!=world
        or value['epoch1_order_hash']!=order_hash(spec,task)
        or value['dropout_rank_seeds']!=[seeds(spec,task)['dropout_base']+r for r in range(world)]
        or value['decoder_present']!= (not cell(task)['control'])
        or set(value['components'])!={'native','predictor','decoder'}
        or any(not re.fullmatch('[0-9a-f]{64}',v) for v in value['components'].values())):
        raise ValueError('Initialization/RNG identity differs')


def validate_pairs(spec,records,*,complete=False):
    """Compare any available pair; missing runs never count as successful pairs."""
    from .contracts import FITS,SEED_INDICES
    expected=set(spec.get('tasks',FITS))
    if not set(records)<=expected or (complete and set(records)!=expected):
        raise ValueError('Paired initialization coverage differs')
    for task,value in records.items():validate_initial(spec,task,value,1)
    for index in SEED_INDICES:
        values=[v for t,v in records.items() if cell(t)['seed_index']==index]
        for v in values[1:]:
            base=values[0]
            if (any(v['components'][p]!=base['components'][p] for p in ('native','predictor'))
                    or any(v[key]!=base[key] for key in ('seeds','dropout_rank_seeds','epoch1_order_hash'))):
                raise ValueError('Same-seed native/predictor/RNG pairing differs')
        decoders={v['components']['decoder'] for v in values if v['decoder_present']}
        if len(decoders)>1:raise ValueError('Same-seed reconstruction decoder initialization differs')


def prepare(spec):
    """Freeze untrained tensors once, independent of subsequent worker CPU models."""
    from .kernel import build
    from .contracts import SEED_INDICES
    from .optimization import isolated_rng
    root=Path(spec['root'])/'setup_a6000'
    entries={}
    with isolated_rng('cpu',seed=2026111601):
        for index in SEED_INDICES:
            task=f'R24_s{index}'
            model,opt=build(spec,task,0,1,'cpu',fresh=True)
            state=model.state_dict()
            path=root/f'initial_s{index}.pt';save_state(path,state)
            entries[str(index)]=dict(state=file_ref(path,root=root),
                components=initial_record(spec,task,model,1)['components'],
                named_tensors={n:tensor_hash({n:t}) for n,t in state.items()})
            del model,opt,state
    result=artifact('INITIAL_STATES',parents=dict(spec=spec['content_hash']),entries=entries,
        untrained=True,source='seeded_from_scratch_setup',precision='fp32')
    write_immutable_json(root/'initial_states.json',result)
    return result


def starts(spec):
    from .contracts import SEED_INDICES
    root=Path(spec['root'])/'setup_a6000';value=load_json(root/'initial_states.json')
    validate(value,'INITIAL_STATES',dict(spec=spec['content_hash']))
    if (set(value['entries'])!={str(i) for i in SEED_INDICES} or value['untrained'] is not True
        or value['source']!='seeded_from_scratch_setup' or value['precision']!='fp32'):
        raise ValueError('Canonical initialization coverage differs')
    for index,row in value['entries'].items():
        if row['state']['path']!=f'initial_s{index}.pt':raise ValueError('Canonical state path differs')
        validate_file_ref(row['state'],root=root)
    return value


def load_start(spec,task,model):
    root=Path(spec['root'])/'setup_a6000';row=starts(spec)['entries'][str(cell(task)['seed_index'])]
    state=torch.load(root/row['state']['path'],map_location='cpu',weights_only=True)
    if {n:tensor_hash({n:t}) for n,t in state.items()}!=row['named_tensors']:
        raise ValueError('Canonical named tensors differ')
    desired=model.state_dict()
    if set(desired)-set(state) or any(not n.startswith('aux.') for n in set(state)-set(desired)):
        raise ValueError('Canonical model inventory differs')
    model.load_state_dict({n:state[n] for n in desired},strict=True)
    check_start(spec,task,initial_record(spec,task,model,1))


def check_start(spec,task,record):
    row=starts(spec)['entries'][str(cell(task)['seed_index'])]
    components=('native','predictor') if cell(task)['control'] else ('native','predictor','decoder')
    if any(record['components'][n]!=row['components'][n] for n in components):
        raise ValueError('Fit does not start from its registered untrained tensors')
