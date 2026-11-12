"""Identical model/RNG path; the subclass term contributes exactly zero."""
from hlt_classification.luka_fullstat_six_teacher.model import Student, new_model
from hlt_classification.luka_fullstat_six_teacher.model import objective as previous_objective


def objective(net, raw, ids, banks, sampler, candidates, weight, device, *, subclass_weight, ramp):
    if subclass_weight != 0.:
        raise ValueError('S0 experiment requires subclass coefficient zero')
    return previous_objective(net, raw, ids, banks, sampler, candidates, weight, device,
                              subclass_weight=0., ramp=ramp)
