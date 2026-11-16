"""Primary in-range selection; diagnostic subsets share the selected model."""
import numpy as np
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_fullstat_balanced_ladder.reporting import rejection, recovery
from . import contracts as c


def summary(y,p):
    complete=set(np.unique(y))==set(range(11))
    measured=evaluate_probabilities(y,p) if complete else dict(rows=len(y),status='missing_class')
    return dict(summary=measured,R50=rejection(y,p,.5),R30=rejection(y,p,.3))


def metrics(val,probabilities,outside,out_prob,arrays,parents):
    pt=arrays['validation_pt']
    if len(pt)!=len(val) or not np.all((pt>=200)&(pt<2000)):
        raise ValueError('Primary validation includes outside-range jets')
    out_pt=arrays['outside_validation_pt']
    if len(out_pt)!=len(outside) or np.any((out_pt>=200)&(out_pt<2000)):
        raise ValueError('Diagnostic validation includes in-range jets')
    values=dict(in_range=summary(val.labels,probabilities),outside_range=summary(outside.labels,out_prob))
    for low in range(200,2000,100):
        mask=(pt>=low)&(pt<low+100)
        values[f'{low}_{low+100}']=summary(val.labels[mask],probabilities[mask])
    return c.artifact('METRICS',parents=parents,populations=values,checkpoint_reselection=False,
        selection_population='in_range',diagnostic_population='outside_range')
