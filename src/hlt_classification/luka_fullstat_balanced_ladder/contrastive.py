"""Eligible TRAIN mix matching; occurrence-keyed RNG and bounded lookahead."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
import numpy as np
import torch
from hlt_classification.luka_fullstat_multi3_contrastive import sampling as donor
from hlt_classification.luka_u100_contrastive.sampling import Sampler as Reference
from .contracts import DRAWS, NEGATIVE_SEED


class OccurrenceRNG:
    def rng(self, token, anchor, component):
        if type(token) is not int or token < 1:
            raise ValueError('Positive occurrence token required')
        epoch, position = divmod(token-1,DRAWS)
        data=['BALANCED_MATCH_OCCURRENCE/v1',self.root_seed,self.replicate,
              epoch+1,position,bytes(self.identities[anchor]).hex(),self.stream,component]
        seed=int.from_bytes(hashlib.sha256(json.dumps(data,separators=(',',':')).encode()).digest(),'little')
        return np.random.Generator(np.random.PCG64(seed))


class SerialReference(OccurrenceRNG, Reference):
    def occurrences(self,anchors,epoch,positions):
        outputs=[]; evidence=[]
        for anchor,pos in zip(anchors,positions,strict=True):
            value,proof=self.sample([anchor],int((epoch-1)*DRAWS+pos+1),'mix')
            outputs.append(value);evidence.extend(proof)
        return np.concatenate(outputs),evidence


@dataclass(frozen=True)
class Proposal:
    anchors: np.ndarray
    tokens: np.ndarray
    pools: np.ndarray
    ranks: np.ndarray


class Sampler(OccurrenceRNG, donor.Sampler):
    def __init__(self,*args,**kwargs):
        kwargs.setdefault('root_seed',NEGATIVE_SEED)
        super().__init__(*args,**kwargs)
        self._parallel_cpu=None

    def prepare(self,anchors,epoch,positions):
        anchors=np.asarray(anchors,dtype=np.int64);positions=np.asarray(positions,dtype=np.int64)
        if (type(epoch) is not int or epoch<1 or anchors.ndim!=1 or not len(anchors)
                or positions.shape!=anchors.shape or np.any(positions<0) or np.any(positions>=DRAWS)
                or len(np.unique(positions))!=len(positions) or np.any(anchors<0) or np.any(anchors>=len(self.labels))):
            raise ValueError('Invalid query occurrences')
        tokens=(epoch-1)*DRAWS+positions+1
        parallel=self._parallel_cpu
        if parallel is not None:
            chunks=[x for x in np.array_split(np.arange(len(anchors)),parallel.workers) if len(x)]
            pieces=parallel.collect([parallel.pool.submit(_prepare,anchors[x],epoch,positions[x]) for x in chunks])
            return Proposal(anchors,tokens,np.concatenate([p.pools for p in pieces]),np.concatenate([p.ranks for p in pieces]))
        pools=np.stack([self.balanced(4096,self.excluded(i),self.rng(int(t),i,'H_pool'),
                           [c for c in range(11) if c!=self.labels[i]]) for i,t in zip(anchors,tokens)])
        ranks=np.stack([self.rng(int(t),i,'H_select').choice(np.arange(256,1024),256,replace=False)
                        for i,t in zip(anchors,tokens)])
        return Proposal(anchors.copy(),tokens,pools,ranks)

    def finish(self,prepared,*,evidence=True):
        with torch.no_grad(),torch.autocast(self.keys.device.type,enabled=False):
            pools=torch.tensor(prepared.pools,device=self.keys.device)
            anchors=torch.tensor(prepared.anchors,device=self.keys.device)
            scores=torch.cat([(self.keys[pools[j:j+8]]*self.keys[anchors[j:j+8],None]).sum(-1)
                              for j in range(0,len(anchors),8)])
            if not hasattr(self,'_tie_tensor'):
                self._tie_tensor=torch.tensor(self.tie_rank,device=self.keys.device)
            canonical=torch.argsort(self._tie_tensor[pools],dim=1,stable=True)
            ranked=torch.argsort(scores.gather(1,canonical),dim=1,descending=True,stable=True)
            ranks=torch.tensor(prepared.ranks,device=self.keys.device)
            hard=pools.gather(1,canonical.gather(1,ranked.gather(1,ranks))).cpu().numpy()
        parallel=self._parallel_cpu
        if parallel is not None:
            chunks=[x for x in np.array_split(np.arange(len(anchors)),parallel.workers) if len(x)]
            pieces=parallel.collect([parallel.pool.submit(_finish,prepared.anchors[x],hard[x],prepared.ranks[x],
                                      prepared.tokens[x],evidence) for x in chunks])
            return np.concatenate([p[0] for p in pieces]),[r for p in pieces for r in p[1]]
        return self.finish_occurrences(prepared.anchors,hard,prepared.ranks,prepared.tokens,evidence)

    def finish_occurrences(self,anchors,hard,ranks,tokens,evidence):
        outputs=[]; details=[]
        for i,h,r,t in zip(anchors,hard,ranks,tokens,strict=True):
            value,proof=super().finish_cpu([i],[h],[r],int(t),evidence)
            outputs.append(value);details.extend(proof)
        return np.concatenate(outputs),details

    def sample(self,anchors,epoch,positions,*,evidence=True):
        return self.finish(self.prepare(anchors,epoch,positions),evidence=evidence)


_CPU=None


def _init(state):
    global _CPU
    from hlt_classification.jetclass2_delphes.cache import _limit_worker_threads
    _limit_worker_threads();torch.set_num_threads(1)
    _CPU=object.__new__(Sampler);_CPU.__dict__.update(state);_CPU._parallel_cpu=None
    for value in (_CPU.identities,_CPU.labels,*_CPU.classes):value.flags.writeable=False


def _prepare(anchors,epoch,positions):
    return donor._cpu_result(_CPU.prepare(anchors,epoch,positions))


def _finish(anchors,hard,ranks,tokens,evidence):
    return donor._cpu_result(_CPU.finish_occurrences(anchors,hard,ranks,tokens,evidence))


class CPUWorkers(donor.CPUWorkers):
    def __enter__(self):
        if self.sampler._parallel_cpu is not None:raise RuntimeError('CPU pool already active')
        state={k:getattr(self.sampler,k) for k in
               ('identities','labels','classes','groups','rng_version','root_seed','replicate','stream')}
        self.pool=ProcessPoolExecutor(max_workers=self.workers,mp_context=multiprocessing.get_context('spawn'),
                                      initializer=_init,initargs=(state,))
        self.sampler._parallel_cpu=self
        return self


class Lookahead(donor.Lookahead):
    def _submit(self):
        if self.next>=len(self.order):return None
        start=self.next;ids=self.order[start:start+self.batch_size].copy();self.next+=len(ids)
        return self.pool.submit(self._occurrences,ids,np.arange(start,self.next,dtype=np.int64))

    def _occurrences(self,ids,positions):
        prepared=self.sampler.prepare(ids,self.epoch,positions)
        if self.stream is None:
            candidates,_=self.sampler.finish(prepared,evidence=False)
        else:
            with torch.cuda.device(self.sampler.keys.device),torch.cuda.stream(self.stream):
                candidates,_=self.sampler.finish(prepared,evidence=False)
        return prepared.anchors,candidates
