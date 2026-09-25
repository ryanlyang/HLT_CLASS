"""Frozen reports can be audited without regenerating or changing anything."""
from copy import deepcopy
import json

import numpy as np
import pytest

from hlt_classification.cms2jc2_response import c_accounting_audit as audit
from hlt_classification.cms2jc2_response.bridge import Particles, p4_from_coordinates
from hlt_classification.cms2jc2_response.c_diagnostic_metrics import Counters
from hlt_classification.cms2jc2_response.contracts import artifact
from hlt_classification.cms2jc2_response.dev_diagnostics import Histograms, NAMES


def particles(pt, category, charge=None):
    n = len(pt)
    charge = charge if charge is not None else [1 if c in (0, 3, 4) else 0 for c in category]
    return Particles(p4_from_coordinates(pt, np.zeros(n), np.zeros(n), np.zeros(n)),
        np.array(charge), np.array(category), np.zeros((n, 4)), np.zeros((n, 4), bool),
        tuple(str(i) for i in range(n)))


def cell(*jets):
    ranges = artifact("DEV_RANGES", definitions={name: dict(edges=[-1000., 0., 1000.]) for name in NAMES})
    hist = Histograms(ranges)
    for p in jets:
        hist.add("real", ["all"], p, p)
    return hist.payload()["cells"]["real/all"]


def test_pooled_share_is_not_mean_per_jet_fraction_and_empty_is_retained():
    c = cell(particles([9, 1], [0, 1]), particles([1, 99], [0, 1]), particles([], []))
    row = audit.particle_accounting([c])
    assert row["jet_observations"] == 3 and row["empty_jets"] == 1
    assert row["charged_fraction_jet_denominator"] == 2
    assert row["mean_per_nonempty_jet_charged_fraction"] == pytest.approx(.455)
    assert row["pooled_known_charged_pt_share"] == pytest.approx(10/110)
    assert row["by_pid"][0]["scalar_pt_per_jet"] == pytest.approx(10/3)
    assert row["by_pid"][0]["mean_particle_pt"] == 5
    pooled = audit.particle_accounting([c, c, c])
    assert pooled["jet_observations"] == 9
    assert pooled["by_pid"][0]["scalar_pt_per_jet"] == row["by_pid"][0]["scalar_pt_per_jet"]
    assert pooled["mean_per_nonempty_jet_charged_fraction"] == row["mean_per_nonempty_jet_charged_fraction"]


def test_unknown_pid_not_silently_assigned_to_charged_or_neutral():
    row = audit.particle_accounting([cell(particles([10, 30, 60], [0, 1, 5], [1, 0, 1]))])
    assert row["pooled_actual_charged_pt_share_bounds"] == [.1, .7]
    assert row["mean_per_nonempty_jet_charged_fraction"] == pytest.approx(.7)
    assert row["pooled_unknown_pt_share"] == .6
    assert row["by_pid"][3]["particles"] == 0
    assert row["by_pid"][3]["mean_particle_pt"] is None


def test_all_empty_population_has_zero_counts_and_undefined_shares():
    row = audit.particle_accounting([cell(particles([], []))])
    assert row["particles"] == 0 and row["sum_pt"] == 0
    assert row["mean_per_nonempty_jet_charged_fraction"] is None
    assert row["pooled_actual_charged_pt_share_bounds"] == [None, None]
    assert all(r["pooled_pt_share"] is None for r in row["by_pid"])
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("metric,field,value", [
    ("photon_pt", "sum", 10), ("jet_count_0", "sum", 8),
    ("jet_scalar_pt", "count", 0), ("jet_charged_fraction", "count", 0),
    ("particle_pt", "sum", float("nan")),
])
def test_corrupt_or_incomplete_accounting_fails_closed(metric, field, value):
    c = cell(particles([2, 3, 4], [0, 1, 2]))
    c["variables"][metric][field] = value
    with pytest.raises(ValueError):
        audit.particle_accounting([c])


def test_mechanism_budgets_include_fallback_and_split_counts_not_emissions():
    p = particles([2, 3], [0, 1])
    # A split emission has two outputs; an identity fallback is still output.
    info = dict(flags={}, events=[dict(module="emission2", mechanism="split", state=[0, 1, 0, 0, 1, 0, 0, 0],
        output=p, missing_value_module=True, invalid_state=False)],
        operations=[], input_categories=np.array([], dtype=int), support_excursions=np.empty((0, 43)),
        missing_modules=[], draws={})
    counters = Counters(); counters.add(info)
    account = audit.particle_accounting([cell(p)])
    budgets = audit.mechanism_accounting(counters.value, account)
    assert sum(r["particles"] for r in budgets) == 2
    emissions = audit.emission_accounting(counters.value)
    assert len(emissions) == 1 and emissions[0]["emissions"] == 1
    assert emissions[0]["rates"]["missing_value_module"] == 1
    bad = deepcopy(counters.value)
    next(iter(bad["mechanisms"].values()))["pt"]["sum"] += 1
    with pytest.raises(ValueError, match="mechanism PID pT"):
        audit.mechanism_accounting(bad, account)


def test_residual_inventory_is_availability_not_occupancy_or_summed_unique_jets():
    backend = artifact("RESIDUAL_BACKEND", cells=[
        dict(key=[0], supported=True, jets=2000),
        dict(key=[0, 0], supported=True, jets=1200),
        dict(key=[1], supported=False, jets=10)], with_crowding=True)
    response = dict(modules={"emission1_value": dict(kind="continuous", backends=[dict(state=[0], backend=backend)]),
                             "emission1_state": dict(states=[[0], [1]])})
    before = deepcopy(response)
    result = audit.residual_inventory(response)
    assert result["observed_selected_level_counts"] is None
    assert result["overlapping_cells"] is True
    assert result["modules"][0]["categorical_states"] == 2
    assert result["modules"][0]["residual_state_backends"] == 1
    assert len(result["cells"]) == 3
    assert result["cells"][2]["supported_cells"] == 0
    assert response == before
    response["modules"]["emission1_value"]["backends"][0]["backend"]["cells"][0]["jets"] += 1
    with pytest.raises(ValueError):
        audit.residual_inventory(response)


def test_coordinate_clipping_denominator():
    from hlt_classification.cms2jc2_response.c_diagnostic_metrics import moments
    value = dict(exposures=2, clipped=1, before=moments([0, 3]), after=moments([0, 1]),
                 absolute_correction=moments([0, 2]))
    row = audit.coordinate_accounting(dict(coordinates={"emission1/singleton/pid0/response/log_pt_ratio": value}))[0]
    assert row["clipped_fraction"] == .5
    assert row["mean_absolute_correction"] == 1
    assert row["mean_absolute_correction_when_clipped"] == 2
