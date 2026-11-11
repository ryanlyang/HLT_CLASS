"""Exact donor sampling with post-exec worker memory accounting."""
from pathlib import Path
import sys
from hlt_classification.luka_fullstat_multi3_contrastive.sampling import (
    Sampler, ReferenceSampler, Lookahead, CPUWorkers as DonorCPUWorkers)
from .contracts import artifact


def memory_policy():
    return artifact('WORKER_MEMORY_POLICY',
        worker_peak='post_exec_proc_status_VmHWM',
        aggregation='parent_peak_plus_sum_worker_peaks',
        inherited_ru_maxrss='diagnostic_only', workers=3,
        missing_measurement='fail_closed', scientific_sampler='unchanged')


def parse_worker_status(text, pid):
    fields = {}
    for line in text.splitlines():
        key, _, value = line.partition(':')
        if key in ('Pid', 'VmHWM', 'VmRSS'):
            if key in fields:
                raise ValueError('Duplicate worker memory field')
            fields[key] = value.split()
    if fields.get('Pid') != [str(pid)]:
        raise ValueError('Worker memory PID differs')
    values = {}
    for key in ('VmHWM', 'VmRSS'):
        value = fields.get(key, [])
        if len(value) != 2 or value[1] != 'kB' or not value[0].isdigit():
            raise ValueError('Missing or invalid worker memory: '+key)
        values[key] = int(value[0])*1024
    if not 0 < values['VmRSS'] <= values['VmHWM']:
        raise ValueError('Invalid worker resident memory bounds')
    return values['VmHWM']


def worker_peak(pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError('Invalid worker PID')
    if sys.platform == 'linux':
        # Unlike getrusage's lifetime counter, VmHWM belongs to the worker's
        # current address space. The spawn/exec parent image is not counted
        # again for each of the three small CPU-only processes.
        return parse_worker_status(Path('/proc', str(pid), 'status').read_text(), pid)
    raise RuntimeError('Oscar worker memory measurement requires Linux /proc')


class CPUWorkers(DonorCPUWorkers):
    """Keep donor proposals/order/RNG; replace only the child peak counter."""
    def __init__(self, sampler, workers=3):
        super().__init__(sampler, workers)
        self.lifetime_peaks = {}

    def collect(self, futures):
        result = []
        for future in futures:
            value, pid, lifetime_peak = future.result()
            peak = worker_peak(pid)
            self.peaks[pid] = max(self.peaks.get(pid, 0), peak)
            self.lifetime_peaks[pid] = max(self.lifetime_peaks.get(pid, 0), lifetime_peak)
            result.append(value)
        return result
