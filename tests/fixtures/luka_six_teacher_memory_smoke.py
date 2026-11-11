"""Linux-only stdlib reproduction; no research data, Torch or GPU required.

Run: python3 tests/fixtures/luka_six_teacher_memory_smoke.py
Tests the actual patched reader functions via AST extraction so this small
OS-accounting check does not require a second ML environment.
"""
import ast
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time


def probe(index):
    time.sleep(.15)
    return index, os.getpid(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


def main():
    if sys.platform != 'linux':
        raise RuntimeError('This reproduction requires Linux')
    source = Path(__file__).resolve().parents[2]/'src/hlt_classification/luka_fullstat_six_teacher/sampling.py'
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('parse_worker_status', 'worker_peak')]
    namespace = dict(Path=Path, sys=sys)
    exec(compile(tree, str(source), 'exec'), namespace)
    # The small child must not be charged the parent's pre-exec address space.
    resident = bytearray(256*2**20)
    parent = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    actual, legacy = {}, {}
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(probe, range(12)))
        assert [row[0] for row in results] == list(range(12))
        for _, pid, lifetime in results:
            actual[pid] = namespace['worker_peak'](pid)
            legacy[pid] = lifetime
        assert len(actual) == 3
        assert max(actual.values()) < 64*2**20
        assert min(legacy.values()) > 200*2**20
        assert parent+sum(actual.values()) < 512*2**20 < parent+sum(legacy.values())
    print(json.dumps(dict(parent_peak_bytes=parent, worker_post_exec_peaks=actual,
                          worker_lifetime_peaks=legacy, passed=True), sort_keys=True))
    del resident


if __name__ == '__main__':
    main()
