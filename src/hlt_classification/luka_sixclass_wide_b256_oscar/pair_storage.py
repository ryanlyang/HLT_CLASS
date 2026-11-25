"""Instance-local, lossless storage of CONCAT pair autograd tensors on CPU."""
from copy import deepcopy
import torch
from hlt_classification.jetclass2_delphes.concat_k2_model import (
    PAIR_STORAGE, PARITY_TOLERANCES, parity_backend,
    _pack_pair_tensor, _unpack_pair_tensor, validate_storage_stats)

POLICY = dict(PAIR_STORAGE, version='sixclass_concat_pair_storage_v1', stage='CONCAT')


class Storage:
    """Keep native module/state keys and the complete BN batch unchanged."""
    def __init__(self, pair):
        self.pair = pair
        self.native = pair.forward
        self.stats = dict(calls=0, saved_cuda_tensors=0, saved_cuda_bytes=0, restored_cuda_tensors=0)
        pair.forward = self.forward

    def forward(self, vectors, *args, **kwargs):
        if not (self.pair.training and torch.is_grad_enabled() and vectors.device.type == 'cuda'):
            return self.native(vectors, *args, **kwargs)
        self.stats['calls'] += 1

        def pack(tensor):
            packed = _pack_pair_tensor(tensor)
            if tensor.device.type == 'cuda':
                self.stats['saved_cuda_tensors'] += 1
                self.stats['saved_cuda_bytes'] += tensor.numel()*tensor.element_size()
            return packed

        def unpack(packed):
            tensor = _unpack_pair_tensor(packed)
            if tensor.device.type == 'cuda':
                self.stats['restored_cuda_tensors'] += 1
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            return self.native(vectors, *args, **kwargs)


def parity(raw, device, *, bf16):
    """Six-output native CE: compare three AdamW steps, including all BN state."""
    from . import model, training, contracts as c
    cuda = torch.device(device).type == 'cuda'
    if bf16 and not cuda:
        raise ValueError('BF16 pair-storage acceptance requires CUDA')
    tolerance = PARITY_TOLERANCES['bf16' if bf16 else 'fp32']
    with parity_backend(device), torch.random.fork_rng(devices=[torch.device(device).index or 0] if cuda else []):
        native = model.new_model('CONCAT').to(device).train()
        stored = model.new_model('CONCAT', pair_storage=True).to(device).train()
        stored.load_state_dict(native.state_dict(), strict=True)
        nets = (native, stored)
        opts = [training.optimizer(n, c.recipe()) for n in nets]
        vectors, mask = [torch.from_numpy(raw[k]).to(device) for k in ('vectors','mask')]
        features = [torch.tensor(raw['features'], device=device, requires_grad=True) for _ in nets]
        labels = torch.as_tensor(raw['labels'], device=device, dtype=torch.long)
        for step in range(3):
            snapshots = []
            for net, opt, x in zip(nets, opts, features):
                opt.zero_grad(set_to_none=True); x.grad = None
                torch.manual_seed(8140+step)
                with torch.autocast(torch.device(device).type, enabled=bf16, dtype=torch.bfloat16):
                    logits = net(x, vectors, mask)
                loss = torch.nn.functional.cross_entropy(logits.float(), labels)
                loss.backward()
                snap = dict(logits=logits.detach().float().cpu(), loss=loss.detach().cpu(),
                            feature_gradient=x.grad.detach().cpu().clone())
                for name, p in net.named_parameters():
                    if p.grad is None or not torch.isfinite(p.grad).all():
                        raise ValueError('Missing/nonfinite pair-storage gradient: '+name)
                    snap['gradient/'+name] = p.grad.detach().cpu().clone()
                opt.step()
                snap.update({'state/'+k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
                for name,p in net.named_parameters():
                    snap.update({f'optimizer/{name}/{k}':v.detach().cpu().clone()
                                 for k,v in opt.state[p].items() if isinstance(v,torch.Tensor)})
                snapshots.append(snap)
            if snapshots[0].keys() != snapshots[1].keys():
                raise ValueError('Pair-storage parity coverage differs')
            for name,a in snapshots[0].items():
                b=snapshots[1][name]
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise ValueError('Nonfinite parity tensor: '+name)
                torch.testing.assert_close(a,b,**tolerance,msg=lambda msg: name+': '+msg)
        stats=deepcopy(stored.pair_storage.stats)
        if cuda:validate_storage_stats(stats,calls=3)
        with torch.inference_mode():
            a,b=[n.eval()(x,vectors,mask) for n,x in zip(nets,features)]
            torch.testing.assert_close(a,b,**tolerance)
        if stored.pair_storage.stats != stats:raise ValueError('Inference offloaded tensors')
    return dict(precision='bf16' if bf16 else 'fp32',steps=3,tolerance=dict(tolerance),
                policy=POLICY,statistics=stats,passed=True)
