"""Component hashes and independent random-stream identities, without consuming RNG."""
import hashlib
import re
import numpy as np
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
