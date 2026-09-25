from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hlt_classification.cms2jc2_response import c_topology as campaign, c_topology_worker as worker
from hlt_classification.cms2jc2_response import dev_campaign as dev, dev_data as data, dev_submission as submission
from hlt_classification.cms2jc2_response.association import associate, policy
from hlt_classification.cms2jc2_response.bridge import Particles, p4_from_coordinates
from hlt_classification.cms2jc2_response.contracts import with_content_hash, sha256_file
from hlt_classification.cms2jc2_response.c_topology_generation import observed_emissions, FixedGenerator, Emission
from hlt_classification.cms2jc2_response.c_diagnostic_generation import DiagnosticGenerator
from hlt_classification.cms2jc2_response.c_diagnostic_worker import historical_replay_equal
from hlt_classification.cms2jc2_response.readers import Pair
from test_cms2jc2_response_science import particles
from test_cms2jc2_frozen_c import fitted, frozen_root, free_disk


def neutral(pt, eta, keys):
    return Particles(p4_from_coordinates(pt, eta, np.zeros(len(pt)), np.zeros(len(pt))),
        np.zeros(len(pt), np.int8), np.ones(len(pt), np.int8), np.zeros((len(pt), 4)),
        np.zeros((len(pt), 4), bool), tuple(keys))


@pytest.mark.parametrize("case", ["singleton", "merge", "split", "additional", "loss", "empty"])
def test_observed_codec_all_mechanisms(case):
    if case == "merge":
        p, h = neutral([2., 8.], [0., .001], ["a", "b"]), neutral([10.], [.0008], ["h"])
    elif case == "split":
        p, h = neutral([10.], [.0008], ["a"]), neutral([2., 8.], [0., .001], ["h0", "h1"])
    elif case == "additional":
        p, h = neutral([], [], []), particles(keys=("h0", "h1"))
    elif case == "loss":
        p, h = particles(), neutral([], [], [])
    elif case == "empty":
        p = h = neutral([], [], [])
    else:
        p, h = particles(), particles(.9, keys=("h0", "h1"))
    rules = policy()
    match = associate(p, h, rules)
    emissions, truth, errors = observed_emissions(p, h, match, rules)
    assert sum(len(e.state)//4 for e in emissions) == len(h)
    assert all(v < 1e-8 for v in errors.values())
    if case in ("merge", "split", "additional"):
        assert any(e.mechanism == {"merge": "merged"}.get(case, case) for e in emissions)
    if case == "loss":
        assert len(truth["operations"]) == 2
    assert not any(hasattr(e, name) for e in emissions for name in ("target", "coordinates", "hlt", "x"))


def test_observed_rejects_unresolved_duplicate_and_wrong_identity():
    p, h, rules = particles(), particles(.9, keys=("h0", "h1")), policy()
    match = associate(p, h, rules)
    for change in (dict(resolved=False), dict(hlt_keys=["wrong", "keys"]),
                   dict(hypotheses=match["hypotheses"]*2)):
        with pytest.raises(ValueError):
            observed_emissions(p, h, with_content_hash(dict(match, **change)), rules)


def test_fixed_predictor_uses_only_offline_and_state_and_replays_free(fitted):
    p, h = particles(), particles(.9, keys=("h0", "h1"))
    rules = fitted["rules"]
    match = associate(p, h, rules)
    emissions, _, _ = observed_emissions(p, h, match, rules)
    gen = FixedGenerator(fitted)
    free = DiagnosticGenerator(fitted, "FULL")
    for replica in range(3):
        from hlt_classification.cms2jc2_response.rng import group_key
        expected, drawn = free(p, jet="held", replica=replica)
        # Parity is meaningful when BOTH registry and state equal the free draw;
        # observed HLT state need not equal a stochastic free prediction.
        replay = tuple(Emission(op["operation"], "singleton", group_key(op["input_keys"]),
            tuple(p.keys.index(k) for k in op["input_keys"]), tuple(op["state"])) for op in drawn["operations"])
        output, info = gen(p, replay, jet="held", replica=replica)
        for name in ("p4", "tracking", "charge", "category", "valid"):
            np.testing.assert_array_equal(getattr(output, name), getattr(expected, name))
        assert sum(info["selected_residual_levels"].values()) == len(replay)
    # Change actual HLT momenta/tracking without changing the given topology or
    # categorical state; the fixed prediction must remain bit-exact.
    changed = Particles(h.p4*1.03, h.charge, h.category, h.tracking*2, h.valid, h.keys)
    other, _, _ = observed_emissions(p, changed, match, rules)
    assert other == emissions
    a, _ = gen(p, emissions, jet="held")
    b, _ = gen(p, other, jet="held")
    np.testing.assert_array_equal(a.p4, b.p4)


def test_missing_module_never_substitutes_target_or_identity(fitted):
    gen = FixedGenerator(fitted)
    emissions = (Emission("emission2", "split", "a", (0,), (0, 1, 15, 1, 1, 0, 0, 1)),)
    assert gen.missing(emissions) == ["emission2_value"]
    with pytest.raises(ValueError, match="unestimable"):
        gen(particles(), emissions, jet="held")


def test_source_rounding_energy_projection_is_explicit():
    p = neutral([10.], [0.], ["p"])
    raw = p.p4.copy(); raw[:, 3] *= 1-5e-8
    h = Particles(raw, p.charge, p.category, p.tracking, p.valid, ("h",))
    rules = policy()
    _, _, errors = observed_emissions(p, h, associate(p, h, rules), rules)
    assert errors["source_energy_projection_maximum"] > 0
    assert errors["p4_maximum"] < 1e-8


def test_missing_backend_is_counted_central_prediction(fitted):
    altered = deepcopy(fitted)
    altered["modules"]["emission1_value"]["backends"] = []
    gen = FixedGenerator(with_content_hash(altered))
    p, h = particles(), particles(.9, keys=("h0", "h1"))
    emissions, _, _ = observed_emissions(p, h, associate(p, h, gen.rules), gen.rules)
    a, info = gen(p, emissions, jet="held")
    b, _ = gen(p, emissions, jet="held", variant="CENTRAL")
    np.testing.assert_array_equal(a.p4, b.p4)
    assert all(e["missing_state_backend"] for e in info["events"])
    assert sum(n for k, n in info["selected_residual_levels"].items() if k.endswith("missing_state_backend")) == 2


def test_zero_or_unestimable_subsets_do_not_fail(fitted):
    from hlt_classification.cms2jc2_response.dev_diagnostics import fit_ranges
    p = neutral([10.], [.0008], ["a"])
    h = neutral([2., 8.], [0., .001], ["h0", "h1"])
    pairs = [Pair("held", "file", p, h)]
    ranges = fit_ranges(pairs, "a"*64)
    result = worker.run_pairs(pairs, fitted, ranges, workers=1, total_jets=1)
    assert result["resolved"] == 1 and result["comparable"] == 0
    assert result["missing_modules"] == {"emission2_value": 1}
    assert result["payloads"]["FREE_ALL"]["cells"]["real/all"]["jets"] == 1
    assert result["payloads"]["FIXED_FULL"]["cells"] == {}
    failed_rules = policy(max_component_objects=1)
    unres = with_content_hash(dict(fitted, rules=failed_rules))
    result = worker.run_pairs(pairs, unres, ranges, workers=1, total_jets=1)
    assert result["resolved"] == result["comparable"] == 0
    assert result["payloads"]["FREE_UNRESOLVED"]["cells"]["real/all"]["jets"] == 1


def test_tiny_root_parallel_end_to_end_and_isolation(frozen_root, tmp_path, monkeypatch):
    _, ctx, put, before, donor = frozen_root
    monkeypatch.setattr(worker, "COUNTS", data.COUNTS)
    spec = campaign.create(parent_spec=dev.stage_dir(donor)/"stage_spec.json", project_dir=tmp_path/"topo_project",
                           source_commit="e"*40, root=tmp_path/"topo")
    plan = submission.submit(spec)
    assert len(plan["commands"]) == 6 and plan["gpus"] == 0 and plan["cpu_upper_bound"] == 144
    assert [t["cpus"] for t in spec["tasks"]] == [2, 36, 36, 36, 36, 1]
    assert all("--partition=tier3" in r["argv"] and not any("--gres" in a for a in r["argv"]) for r in plan["commands"])
    bad = deepcopy(spec); bad["tasks"][1]["cpus"] = 64
    with pytest.raises(ValueError, match="registration"):
        dev.validate_stage(with_content_hash(bad))
    calls = []
    def scheduler(argv):
        calls.append(argv)
        out = "PartitionName=tier3 State=UP" if argv[0] == "scontrol" else str(3000+len(calls))+"\n"
        return SimpleNamespace(returncode=0, stdout=out, stderr="")
    monkeypatch.setattr(submission, "scheduler", scheduler)
    with pytest.raises(PermissionError):
        submission.submit(spec, execute=True, reviewed_plan_hash=plan["content_hash"])
    ledger = submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES["ctopo"], reviewed_plan_hash=plan["content_hash"])
    live = [cmd for cmd in calls if cmd[0] == "sbatch" and "--test-only" not in cmd]
    assert len(live) == 6
    assert all("--dependency=afterok:"+ledger["jobs"]["ct_acceptance"] in cmd for cmd in live[1:5])
    assert "--dependency=afterok:"+":".join(ledger["jobs"][f"ct_eval_{i}"] for i in range(4)) in live[5]
    assert submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES["ctopo"], reviewed_plan_hash=plan["content_hash"]) == ledger
    result = worker.acceptance(ctx, spec)
    assert result["serial_process_parity"] and result["exact_free_replay"]
    put(spec, "ct_acceptance", dict(result=result))
    # Exercise parallel and serial shard aggregation, but don't spawn 144 test processes.
    for t in spec["tasks"][1:5]:
        t = dict(t, cpus=2 if t["params"]["shard"] == 0 else 1)
        put(spec, t["task_id"], dict(result=worker.evaluate(ctx, spec, t)))
    from hlt_classification.cms2jc2_response import dev_diagnostics as diag
    def one_page(*args, cohort, **kw):
        if cohort != "all": return
        pages = diag.plot_pages(*args, cohort=cohort, **kw)
        try:
            page = next(pages, None)
            if page is not None: yield page
        finally: pages.close()
    monkeypatch.setattr(worker, "plot_pages", one_page)
    result = worker.report(ctx, spec)
    assert result["jets"] == 12 and result["views"]["FREE_ALL"]["proxy"]["jet_observations"] == 36
    assert result["figures"] and not result["final_test_accessed"] and result["observed_state_privileged"]
    put(spec, "ct_report", dict(result=result))
    from hlt_classification.cms2jc2_response.c_topology_results import read, render
    def forbidden(*args, **kwargs): raise AssertionError("Read-only results attempted raw data or scheduler access")
    monkeypatch.setattr(worker, "sample_stream", forbidden)
    monkeypatch.setattr(submission, "scheduler", forbidden)
    assert "NOT a deployable" in render(read(spec))
    assert before == {p: sha256_file(p) for p in Path(donor["root"]).rglob("*") if p.is_file()}
    path = Path(donor["root"])/dev.verified_outputs(donor, "candidate_C_L")["outputs"]["response"]["relative"]
    path.write_text("{}")
    with pytest.raises(ValueError, match="Corrupt"):
        read(spec)
