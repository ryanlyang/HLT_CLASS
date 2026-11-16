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
from .contrastive_reference import Sampler as ReferenceSampler
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
        classes=list(range(6)) if classes is None else list(classes)
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
            [c for c in range(6) if c != self.labels[i]]) for i in anchors])
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
                details.append(dict(anchor=int(i), components={k:np.bincount(self.labels[v],minlength=6).tolist()
                    for k,v in (('H',h),('S',s),('G',g))}, hard_ranks=(positions+1).tolist(),
                    duplicates=0, excluded_count=len(self.excluded(i)),
                    same_class_fraction=float(np.mean(self.labels[negative] == self.labels[i]))))
        return np.stack(results), details

    def sample(self, anchors, epoch, strategy='mix', *, evidence=True):
        if strategy != 'mix':
            raise ValueError('This campaign registers mix only')
        return self.finish(self.prepare(anchors,epoch),evidence=evidence)


from hlt_classification.luka_fullstat_multi3_contrastive.sampling import _cpu_result, Lookahead, CPUWorkers
