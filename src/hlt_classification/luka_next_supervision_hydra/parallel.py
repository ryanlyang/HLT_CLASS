"""CPU-only fork workers share immutable arrays; all CUDA work stays in the parent."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import numpy as np

_SOURCE=None


def _ready(): return os.getpid()


def _batch(job):
    indices,epoch,cfg=job
    return _SOURCE.batch(indices,epoch,cfg)


def _prepare(job):
    indices,epoch=job
    return [_SOURCE.semi_prepare(int(i),epoch) for i in indices]


def _finish(job):
    prepared,scores=job
    return [_SOURCE.semi_finish(p,s) for p,s in zip(prepared,scores)]


class ParallelCandidates:
    def __init__(self,source,workers=4,*,executor=None):
        global _SOURCE
        if source.p.role!='train': raise PermissionError('CPU candidate workers are TRAIN-only')
        _SOURCE=source; self.source=source; self.capacity=dict(source.capacity,sampler_workers=workers)
        if executor is None:
            if os.name!='posix' or 'fork' not in multiprocessing.get_all_start_methods():
                raise RuntimeError('CPU candidate workers require Linux fork; CUDA is forbidden in children')
            self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork'))
        else: self.pool=executor
        # Eagerly start workers before a live student/autograd graph exists.
        self.pool.submit(_ready).result()

    def batch(self,indices,epoch,cfg,mine=None):
        chunks=[np.asarray(indices[i:i+8],np.int64) for i in range(0,len(indices),8)]
        if cfg['family']=='P' and cfg['law']=='Hsemi':
            if mine is None: raise ValueError('Parent detached scorer required')
            prepared=list(self.pool.map(_prepare,[(ix,epoch) for ix in chunks]))
            jobs=[]; start=0
            for group in prepared:
                proposals=np.stack([p['proposal'] for p in group])
                scores=mine(np.arange(start,start+len(group)),self.source.p.features[proposals])
                jobs.append((group,scores)); start+=len(group)
            rows=[r for group in self.pool.map(_finish,jobs) for r in group]
            ix=np.stack([r['indices'] for r in rows])
            return dict(features=self.source.p.features[ix],labels=np.stack([r['labels'] for r in rows])),[r['evidence'] for r in rows]
        results=list(self.pool.map(_batch,[(ix,epoch,cfg) for ix in chunks]))
        return ({k:np.concatenate([batch[k] for batch,_ in results]) for k in results[0][0]},
                [row for _,evidence in results for row in evidence])

    def child_pss_bytes(self):
        # Add child proportional RSS to parent peak RSS, conservatively counting
        # some shared pages twice rather than omitting child-private caches.
        total=0
        for pid in getattr(self.pool,'_processes',{}):
            path=Path(f'/proc/{pid}/smaps_rollup')
            if not path.is_file(): raise RuntimeError('Cannot account for live CPU candidate worker memory')
            entries=[line.split() for line in path.read_text().splitlines() if line.startswith('Pss:')]
            if len(entries)!=1: raise RuntimeError('Missing child proportional RSS')
            total+=int(entries[0][1])*1024
        return total

    def close(self):
        global _SOURCE
        self.pool.shutdown(wait=True,cancel_futures=True)
        if _SOURCE is self.source: _SOURCE=None
