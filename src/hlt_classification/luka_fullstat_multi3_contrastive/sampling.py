"""Exact mix sampler with stable GPU ranking and bounded candidate lookahead.

The reference reduction shape remains eight queries: changing to GEMM or a
larger reduction could change rounded scores and therefore hard candidates.
"""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing
import os
import numpy as np
import torch
from hlt_classification.luka_u100_contrastive.sampling import Sampler as ReferenceSampler
from hlt_classification.luka_u100_contrastive.contracts import POOL, NEGATIVES


@dataclass(frozen=True)
class Proposal:
    anchors: np.ndarray
    epoch: int
    pools: np.ndarray
    positions: np.ndarray


class Sampler(ReferenceSampler):
    def diagnostic_sampler(self):
        result=super().diagnostic_sampler()
        # A training pool must never supply the distinct diagnostic RNG stream.
        result._parallel_cpu=None
        return result

    @staticmethod
    def draw_sorted(pool,size,values,rng):
        """Reference draw with already sorted, in-pool exclusions."""
        if len(pool)<size:raise ValueError('Candidate quota unavailable')
        if not len(values):return pool[rng.choice(len(pool),size,replace=False)]
        if len(values)>len(pool)//4:
            pool=pool[~np.isin(pool,values)]
            if len(pool)<size:raise ValueError('Eligible candidate quota unavailable')
            return pool[rng.choice(len(pool),size,replace=False)]
        proposal=pool[rng.choice(len(pool),min(len(pool),size+len(values)),replace=False)]
        pos=np.searchsorted(values,proposal);valid=pos<len(values)
        keep=np.ones(len(proposal),dtype=bool);keep[valid]=values[pos[valid]]!=proposal[valid]
        result=proposal[keep][:size]
        if len(result)!=size:raise ValueError('Eligible candidate quota unavailable')
        return result

    def balanced(self,size,blocked,rng,classes=None,pools=None):
        if pools is not None:return super().balanced(size,blocked,rng,classes,pools)
        classes=list(range(11)) if classes is None else list(classes)
        quota={c:size//len(classes) for c in classes}
        for c in rng.permutation(classes)[:size%len(classes)]:quota[int(c)]+=1
        values=np.asarray(sorted(blocked),dtype=np.int64)
        labels=self.labels[values]
        return np.concatenate([self.draw_sorted(self.classes[c],quota[c],values[labels==c],rng) for c in classes])

    def prepare(self, anchors, epoch):
        anchors = np.asarray(anchors, dtype=np.int64).copy()
        if (anchors.ndim != 1 or not len(anchors) or np.any(anchors < 0)
                or np.any(anchors >= len(self.labels))):
            raise ValueError('Invalid anchor indices')
        parallel=getattr(self,'_parallel_cpu',None)
        if parallel is not None:
            chunks=[a for a in np.array_split(anchors,parallel.workers) if len(a)]
            pieces=parallel.collect([parallel.pool.submit(_prepare_cpu,a,epoch) for a in chunks])
            return Proposal(anchors,epoch,np.concatenate([p.pools for p in pieces]),
                            np.concatenate([p.positions for p in pieces]))
        pools = np.stack([self.balanced(POOL, self.excluded(i), self.rng(epoch, i, 'H_pool'),
            [c for c in range(11) if c != self.labels[i]]) for i in anchors])
        positions=np.stack([self.rng(epoch,i,'H_select').choice(np.arange(256,1024),256,replace=False)
                            for i in anchors])
        anchors.flags.writeable = False; pools.flags.writeable = False
        positions.flags.writeable=False
        return Proposal(anchors, epoch, pools, positions)

    def finish(self, prepared, *, evidence=True):
        anchors, epoch, pools = prepared.anchors, prepared.epoch, prepared.pools
        scores = []
        with torch.no_grad(), torch.autocast(self.keys.device.type, enabled=False):
            # Upload once. Per-chunk blocking index uploads would serialize each
            # preceding mining kernel despite the single final score readback.
            all_pools=torch.tensor(pools,device=self.keys.device)
            all_anchors=torch.tensor(anchors,device=self.keys.device)
            for offset in range(0, len(anchors), 8):
                # Match reference indexing and sum, including the final partial chunk.
                ix = all_pools[offset:offset+8]
                a = all_anchors[offset:offset+8]
                scores.append((self.keys[ix]*self.keys[a, None]).sum(-1))
            values=torch.cat(scores)
            if not hasattr(self,'_tie_tensor'):
                self._tie_tensor=torch.as_tensor(self.tie_rank,device=self.keys.device)
            # Stable descending sort after canonical sorting implements the
            # reference lexsort exactly, including all-equal and signed-zero ties.
            canonical=torch.argsort(self._tie_tensor[all_pools],dim=1,stable=True)
            ranked=torch.argsort(values.gather(1,canonical),dim=1,descending=True,stable=True)
            positions_gpu=torch.tensor(prepared.positions,device=self.keys.device)
            chosen=canonical.gather(1,ranked.gather(1,positions_gpu))
            hard=all_pools.gather(1,chosen).cpu().numpy()
        parallel=getattr(self,'_parallel_cpu',None)
        if parallel is not None:
            chunks=[a for a in np.array_split(np.arange(len(anchors)),parallel.workers) if len(a)]
            pieces=parallel.collect([parallel.pool.submit(_finish_cpu,anchors[ix],hard[ix],
                prepared.positions[ix],epoch,evidence) for ix in chunks])
            return np.concatenate([p[0] for p in pieces]),[d for p in pieces for d in p[1]]
        return self.finish_cpu(anchors,hard,prepared.positions,epoch,evidence)

    def finish_cpu(self,anchors,hard,selected_positions,epoch,evidence):
        results, details = [], []
        for i,h,positions in zip(anchors,hard,selected_positions):
            blocked = self.excluded(i)
            blocked.update(map(int,h))
            s = self.draw(self.classes[self.labels[i]],256,blocked,self.rng(epoch,i,'S'))
            blocked.update(map(int,s))
            g = self.balanced(512,blocked,self.rng(epoch,i,'G'))
            negative = np.concatenate((h,s,g))
            if (len(negative) != NEGATIVES or len(np.unique(negative)) != NEGATIVES
                    or self.excluded(i).intersection(map(int,negative))):
                raise ValueError('Candidate identity exclusion/uniqueness failed')
            results.append(np.concatenate(([i],negative)))
            if evidence:
                details.append(dict(anchor=int(i), components={k:np.bincount(self.labels[v],minlength=11).tolist()
                    for k,v in (('H',h),('S',s),('G',g))}, hard_ranks=(positions+1).tolist(),
                    duplicates=0, excluded_count=len(self.excluded(i)),
                    same_class_fraction=float(np.mean(self.labels[negative] == self.labels[i]))))
        return np.stack(results), details

    def sample(self, anchors, epoch, strategy='mix', *, evidence=True):
        if strategy != 'mix':
            raise ValueError('This campaign registers mix only')
        return self.finish(self.prepare(anchors,epoch),evidence=evidence)


_CPU_SAMPLER=None


def _init_cpu(state):
    global _CPU_SAMPLER
    from hlt_classification.jetclass2_delphes.cache import _limit_worker_threads
    _limit_worker_threads();torch.set_num_threads(1)
    # Only immutable CPU indices/identities. No target keys, model or CUDA
    # objects cross a process boundary; spawning never forks a CUDA context.
    _CPU_SAMPLER=object.__new__(Sampler)
    _CPU_SAMPLER.__dict__.update(state)
    for a in (_CPU_SAMPLER.identities,_CPU_SAMPLER.labels,*_CPU_SAMPLER.classes):a.flags.writeable=False


def _cpu_result(value):
    peak=0
    if os.name=='posix':
        import resource
        peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    return value,os.getpid(),peak


def _prepare_cpu(anchors,epoch):return _cpu_result(_CPU_SAMPLER.prepare(anchors,epoch))


def _finish_cpu(anchors,hard,positions,epoch,evidence):
    return _cpu_result(_CPU_SAMPLER.finish_cpu(anchors,hard,positions,epoch,evidence))


class CPUWorkers:
    """Bounded CPU parallelism with ordered results and unchanged per-jet RNG.

    Three workers leave the other allocated CPUs for model/input work and
    candidate coordination. The pool persists across all scientific epochs.
    """
    def __init__(self,sampler,workers=3):
        if workers not in (1,2,3):raise ValueError('Unregistered CPU worker count')
        self.sampler,self.workers=sampler,workers;self.peaks={}

    def __enter__(self):
        if getattr(self.sampler,'_parallel_cpu',None) is not None:raise RuntimeError('CPU pool already active')
        state={k:getattr(self.sampler,k) for k in
            ('identities','labels','classes','groups','rng_version','root_seed','replicate','stream')}
        self.pool=ProcessPoolExecutor(max_workers=self.workers,mp_context=multiprocessing.get_context('spawn'),
                                      initializer=_init_cpu,initargs=(state,))
        self.sampler._parallel_cpu=self
        return self

    def collect(self,futures):
        result=[]
        for future in futures:
            value,pid,peak=future.result()
            self.peaks[pid]=max(self.peaks.get(pid,0),peak);result.append(value)
        return result

    def memory(self):return sum(self.peaks.values())

    def __exit__(self,*exc):
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.sampler._parallel_cpu=None


class Lookahead:
    """One worker prepares frozen candidates while the student trains.

    A dedicated stream reads only fixed keys. It waits for key publication;
    CPU candidate return synchronizes mining before use. No global RNG, input
    batches or trainable tensors enter this worker. At most one pending batch.
    """
    def __init__(self, sampler, order, epoch, batch_size=128):
        self.sampler, self.order, self.epoch, self.batch_size = sampler, order, epoch, batch_size
        self.pool = None

    def __enter__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mix-proposal')
        self.stream=None
        if self.sampler.keys.device.type=='cuda':
            self.stream=torch.cuda.Stream(device=self.sampler.keys.device)
            self.stream.wait_stream(torch.cuda.current_stream(self.sampler.keys.device))
        self.next = 0
        self.future = self._submit()
        return self

    def _submit(self):
        if self.next >= len(self.order): return None
        ids = self.order[self.next:self.next+self.batch_size].copy()
        self.next += len(ids)
        return self.pool.submit(self._candidates,ids)

    def _candidates(self,ids):
        prepared=self.sampler.prepare(ids,self.epoch)
        if self.stream is None:
            candidates,_=self.sampler.finish(prepared,evidence=False)
        else:
            with torch.cuda.device(self.sampler.keys.device),torch.cuda.stream(self.stream):
                candidates,_=self.sampler.finish(prepared,evidence=False)
        return prepared.anchors,candidates

    def __iter__(self): return self

    def __next__(self):
        if self.future is None: raise StopIteration
        result = self.future.result()
        self.future = self._submit()
        return result

    def __exit__(self, *exc):
        self.pool.shutdown(wait=True,cancel_futures=True)
