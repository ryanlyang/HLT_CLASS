"""Bounded process retries, preserving the accepted prefix and every row."""
from collections import deque
import errno
import multiprocessing
from pathlib import Path
import sys
import threading
import time
from . import contracts as c


class UnreapedPoolError(RuntimeError):pass


def initialize(repair_path):
    if repair_path is None:return  # Synthetic process/failure fixtures only.
    from .runtime import install
    old=install(c.read(repair_path),light=True)
    old.inputs.initialize_decode_worker()


def execute(function,batch,recheck):
    if recheck:
        for dataset,record in {(arg[0],arg[1]['path']): (arg[0],arg[1]) for arg in batch}.values():
            if c.digest(Path(dataset)/record['path'])!=record['sha256']:
                raise ValueError('Raw source checksum differs before retry: '+record['path'])
    return function(batch)


def stop(pool,timeout=15):
    errors=[]
    def terminate():
        try:pool.terminate();pool.join()
        except BaseException as error:errors.append(error)
    thread=threading.Thread(target=terminate,daemon=True);thread.start();thread.join(timeout)
    if thread.is_alive():raise UnreapedPoolError('Owned input pool could not stop; abort allocation without retry')
    if errors:raise UnreapedPoolError('Owned input pool cleanup failed') from errors[0]


def retryable(error):
    return isinstance(error,multiprocessing.TimeoutError) or (
        isinstance(error,OSError) and error.errno in {errno.EIO,errno.ETIMEDOUT,errno.ESTALE,
            errno.ECONNRESET,errno.ECONNABORTED,errno.EHOSTUNREACH,errno.ENETDOWN,errno.ENETUNREACH})


def decoded(arguments,*,workers=4,task_timeout=600,function=None,repair_path=None,
            max_restarts=2,context=None,batcher=None,on_retry=None):
    if workers<1 or task_timeout<=0 or max_restarts not in (0,1,2):raise ValueError('Invalid bounded retry configuration')
    if function is None:
        original=sys.modules['hlt_classification.luka_d20_full_hydra.inputs']
        function=original.work_batch
    if batcher is None:batcher=sys.modules['hlt_classification.luka_d20_full_hydra.inputs'].batches
    jobs=iter(batcher(arguments));pending=deque();restarts=0;pool=None
    context=multiprocessing.get_context('spawn') if context is None else context
    def start():return context.Pool(workers,initializer=initialize,initargs=(repair_path,))
    def submit(batch,recheck=False):
        return batch,pool.apply_async(execute,(function,batch,recheck)),time.monotonic()+task_timeout
    try:
        pool=start()
        for _ in range(workers):
            batch=next(jobs,None)
            if batch is not None:pending.append(submit(batch))
        while pending:
            batch,future,deadline=pending.popleft()
            try:results=future.get(timeout=max(0.,deadline-time.monotonic()))
            except Exception as error:
                if not retryable(error):raise
                replay=[batch,*[item[0] for item in pending]]
                owned=pool;pool=None;stop(owned)
                if restarts>=max_restarts:
                    raise TimeoutError(f'Input retries exhausted after {restarts} pool restarts; '
                        f'file={batch[0][1]["path"]} role={batch[0][5]}; no rows skipped') from error
                restarts+=1
                evidence=dict(restart=restarts,file=batch[0][1]['path'],role=batch[0][5],
                    spans=[[arg[3],arg[4]] for arg in batch],replayed_batches=len(replay),reason=type(error).__name__)
                print('D20_INPUT_RETRY '+str(evidence),flush=True)
                if on_retry is not None:on_retry(evidence)
                pool=start();pending=deque(submit(item,True) for item in replay)
                continue
            yield from zip(batch,results,strict=True)
            following=next(jobs,None)
            if following is not None:pending.append(submit(following))
    finally:
        if pool is not None:stop(pool)
