"""The complete, immutable 30-fit scientific graph; no quality-dependent pruning."""
from copy import deepcopy

CONDITIONS = ('B', 'K', 'I', 'T', 'IT', 'KIT')
COORDINATES = ('D000', 'D025', 'D033', 'D050', 'D066', 'D075', 'U100', 'OFFLINE')
QUERY_NAMES = ('K', 'I', 'T', 'IT', 'J')
SCALES = ('coarse', 'medium', 'fine')
SLICES = dict(coarse=(0, 2), medium=(2, 6), fine=(6, 14))


def _registry():
    rows = {'C0': dict(family='control', role='hlt', input='D000', parent=None, weight=0.)}
    for prefix, views in (('LS', ('D050', 'D000')), ('LM', ('D066', 'D033', 'D000')),
                          ('LL', ('D075', 'D050', 'D025', 'D000'))):
        parent = None
        for view in views:
            task = prefix+'_'+view
            rows[task] = dict(family='ladder', role='hlt' if view == 'D000' else 'assistant',
                              input=view, parent=parent, weight=5.)
            parent = task
    rows['TR'] = dict(family='response_teacher', role='teacher', input='conditions', parent=None, weight=1.)
    rows['TP'] = dict(family='region_teacher', role='teacher', input='sources', parent=None, weight=1.)
    queries = [('I',), ('T',), ('K',), ('I','T','IT'), ('I','T','IT'),
               ('I','T','IT'), ('I','T','IT'), ('I','T','J'), ('I','T','IT')]
    weights = (1., 1., 1., .3, 1., 3., 10., 1., 1.)
    for i, (q, w) in enumerate(zip(queries, weights), 1):
        rows[f'R{i}'] = dict(family='response', role='hlt', input='D000', parent='TR',
                            weight=w, queries=list(q), absolute=i == 9)
    for i, w in enumerate(weights, 1):
        scales = (SCALES[i-1],) if i <= 3 else SCALES if i <= 8 else ('whole',)
        rows[f'P{i}'] = dict(family='region', role='hlt', input='D000', parent='TP', weight=w,
                            scales=list(scales), source='HLT' if i == 8 else 'OFFLINE')
    return rows


_CELLS = _registry()
FITS = ('C0', 'LS_D050', 'TR', 'TP', 'LM_D066', 'LL_D075',
        'LS_D000', 'R5', 'P5', 'LM_D033', 'LL_D050', 'R6', 'P6',
        'LM_D000', 'LL_D025', 'R4', 'P4', 'LL_D000', 'R7', 'P7',
        'R1', 'P1', 'R2', 'P2', 'R3', 'P3', 'R8', 'P8', 'R9', 'P9')
ASSISTANTS = tuple(t for t in FITS if _CELLS[t]['role'] == 'assistant')
PRODUCERS = (*ASSISTANTS, 'TR', 'TP')
HLT_FITS = tuple(t for t in FITS if _CELLS[t]['role'] == 'hlt')
ALIASES = {'R0': 'C0', 'P0': 'C0'}


def cell(task):
    if task not in _CELLS:
        raise ValueError('Unknown scientific task: '+str(task))
    return deepcopy(_CELLS[task])


def dependencies(task):
    parent = cell(task)['parent']
    return [] if parent is None else [parent]


def teacher_condition(epoch, batch, *, regional=False):
    """Six balanced conditions, independent of labels, identical on both ranks."""
    import numpy as np
    if epoch < 1 or batch < 0:
        raise ValueError('One-based epoch and zero-based batch required')
    order = np.random.default_rng(np.random.SeedSequence([20261110, epoch, int(regional)])).permutation(6)
    index = int(order[batch % 6])
    return (('HLT','OFFLINE')[index//3], SCALES[index%3]) if regional else CONDITIONS[index]
