"""Focused contracts for the isolated JetClass2 salience campaign."""
from fractions import Fraction

import numpy as np
import pytest

from test_jetclass2_delphes import particle_values, snapshot
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.reader import Jet, Particles
from hlt_classification.jetclass2_delphes.salience_campaign import build_campaign_plan
from hlt_classification.jetclass2_delphes.salience_foundation import (
    audit_assignment_sample, audit_sample, build_assignment_shard, build_foundation_lock,
    build_foundation_spec,
)
from hlt_classification.jetclass2_delphes.salience_cache import prepare_cache
from hlt_classification.jetclass2_delphes.salience_production import task_graph
from hlt_classification.jetclass2_delphes.salience_screen import REGISTRY, _selection_indices
from hlt_classification.jetclass2_delphes.salience_views import build_view, match_particles
from hlt_classification.jetclass2_delphes.split_registry import build_registry, select_profile


@pytest.fixture
def registered(snapshot):
    data, inventory, reservoirs = snapshot
    registry = build_registry(data, inventory, reservoirs, training_sizes=(11,),
                              validation_size=11, test_size=11, step_size=3)
    return data, inventory, select_profile(registry, inventory, "TRAIN_11")


@pytest.mark.parametrize("candidate", REGISTRY)
@pytest.mark.parametrize("nh,no", [(2, 4), (4, 2), (3, 3)])
def test_persistent_hlt_full_cardinality_endpoints(candidate, nh, no):
    hlt, offline = particle_values(nh), particle_values(no)
    hlt[:, 1] = np.linspace(0, .2, nh); offline[:, 1] = np.linspace(.01, .21, no)
    jet = Jet("a" * 64, 1, Particles(hlt), Particles(offline))
    mapping = match_particles(jet.hlt, jet.offline, candidate)
    assert np.count_nonzero(mapping >= 0) == min(nh, no)
    assert len(np.unique(mapping[mapping >= 0])) == min(nh, no)
    u000 = build_view(jet, u=Fraction(0), f=Fraction(0), candidate=candidate, mapping=mapping)
    u100 = build_view(jet, u=Fraction(1), f=Fraction(0), candidate=candidate, mapping=mapping)
    d000 = build_view(Jet(jet.identity, jet.label, jet.hlt, None),
                      u=Fraction(1), f=Fraction(1), candidate=candidate)
    assert len(u000) == max(nh, no)
    assert len(u100) == nh
    assert d000 is jet.hlt
    for slot, offline_index in enumerate(mapping):
        np.testing.assert_array_equal(u100.values[slot],
                                      offline[offline_index] if offline_index >= 0 else hlt[slot])


def test_salience_foundation_ram_cache_and_three_spines(registered, tmp_path):
    data, inventory, splits = registered
    foundation = build_foundation_spec(inventory, splits, REGISTRY[0])
    root = tmp_path / "foundation"
    for task in foundation["assignment_tasks"]:
        build_assignment_shard(foundation, data_root=data, output_root=root,
                               file_index=task["file_index"])
    from hlt_classification.data.cache_contracts import write_immutable_json
    write_immutable_json(root / "assignment_audit.json",
                         audit_assignment_sample(foundation, root=root, data_root=data))
    lock = build_foundation_lock(foundation, root)
    assert lock["particle_views_persisted"] is False
    assert lock["persistent_hlt"] is True
    cache = prepare_cache(foundation, data_root=data, foundation_root=root,
                          role="train", coordinate_name="U000", workers=2,
                          max_ram_bytes=100_000_000)
    assert len(cache) == 11 and cache.nbytes > 0
    plan = build_campaign_plan(foundation)
    assert plan["branch_order"] == ["DIRECT", "COARSE", "DENSE"]
    assert plan["ultradense_present"] is False
    assert plan["fresh_fit_count"] == 16
    assert plan["probability_publication_count"] == 12
    assert len(task_graph(plan)) == 30


def test_screen_partition_is_deterministic_stratified_and_disjoint():
    labels = np.repeat(np.arange(11), 8)
    identities = np.asarray([
        np.frombuffer(bytes.fromhex(f"{index:064x}"), np.uint8)
        for index in range(len(labels))
    ], np.uint8)
    first = _selection_indices(labels, identities)
    second = _selection_indices(labels, identities)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    checkpoint, selection = first
    assert not set(checkpoint) & set(selection)
    assert set(checkpoint) | set(selection) == set(range(len(labels)))
    assert np.bincount(labels[checkpoint], minlength=11).tolist() == [6] * 11
    assert np.bincount(labels[selection], minlength=11).tolist() == [2] * 11


def test_candidate_readiness_is_non_scientific_and_dry_by_default(
    registered, tmp_path, monkeypatch,
):
    from hlt_classification.data.cache_contracts import load_json
    from hlt_classification.jetclass2_delphes import salience_readiness

    data, inventory, splits = registered
    monkeypatch.setattr(salience_readiness, "_source", lambda *args: None)
    spec = salience_readiness.create_readiness(
        inventory=inventory, split_profile=splits, candidate=REGISTRY[0],
        data_root=data, output_root=tmp_path / "readiness", project=tmp_path,
        source_commit="a" * 40,
    )
    ledger = load_json(tmp_path / "readiness/dry_run_submission_ledger.json")
    plan = load_json(tmp_path / "readiness/command_plan.json")
    assert spec["scientific_fits"] == 0
    assert ledger["dry_run"] is True
    assert [row["task_id"] for row in plan["commands"]] == ["sample", "assign", "lock"]
    assert not any("--gres" in token for row in plan["commands"] for token in row["command"])


def test_candidate_readiness_supports_bounded_recovery_walltime(
    registered, tmp_path, monkeypatch,
):
    from hlt_classification.jetclass2_delphes import salience_readiness

    data, inventory, splits = registered
    monkeypatch.setattr(salience_readiness, "_source", lambda *args: None)
    spec = salience_readiness.create_readiness(
        inventory=inventory, split_profile=splits, candidate=REGISTRY[0],
        data_root=data, output_root=tmp_path / "readiness_480", project=tmp_path,
        source_commit="a" * 40, assignment_minutes=480,
    )
    plan = load_json(tmp_path / "readiness_480/command_plan.json")
    assign = next(row for row in plan["commands"] if row["task_id"] == "assign")
    assert spec["contract"] == "JETCLASS2_DELPHES_SALIENCE_READINESS_SPEC/v2"
    assert spec["assignment_minutes"] == 480
    assert "--time=480" in assign["command"]
    salience_readiness.validate_readiness(spec)
    with pytest.raises(ValueError, match="walltime"):
        salience_readiness.create_readiness(
            inventory=inventory, split_profile=splits, candidate=REGISTRY[0],
            data_root=data, output_root=tmp_path / "readiness_bad", project=tmp_path,
            source_commit="a" * 40, assignment_minutes=481,
        )


def test_staged_specs_and_production_dry_shape(registered, tmp_path, monkeypatch):
    from hlt_classification.data.cache_contracts import write_immutable_json
    from hlt_classification.jetclass2_delphes.cache import cache_budgets
    from hlt_classification.jetclass2_delphes.contracts import artifact
    from hlt_classification.jetclass2_delphes.execution import execution_site
    from hlt_classification.jetclass2_delphes.foundation import (
        build_assignment_shard as build_bottleneck_assignment,
        build_foundation_lock as build_bottleneck_lock,
        build_foundation_spec as build_bottleneck_foundation,
    )
    from hlt_classification.jetclass2_delphes import salience_production, salience_screen

    data, inventory, splits = registered
    bottleneck_root = tmp_path / "bottleneck"
    bottleneck = build_bottleneck_foundation(inventory, splits)
    write_immutable_json(bottleneck_root / "foundation_spec.json", bottleneck)
    for task in bottleneck["assignment_tasks"]:
        build_bottleneck_assignment(bottleneck, data_root=data,
                                    output_root=bottleneck_root,
                                    file_index=task["file_index"])
    build_bottleneck_lock(bottleneck, bottleneck_root)

    candidate_roots = []
    for index, candidate in enumerate(REGISTRY):
        root = tmp_path / f"candidate_{index}"
        foundation = build_foundation_spec(inventory, splits, candidate)
        write_immutable_json(root / "foundation_spec.json", foundation)
        write_immutable_json(root / "sample_audit.json",
                             audit_sample(foundation, data_root=data, rows_per_file=1))
        for task in foundation["assignment_tasks"]:
            build_assignment_shard(foundation, data_root=data, output_root=root,
                                   file_index=task["file_index"])
        write_immutable_json(root / "assignment_audit.json",
                             audit_assignment_sample(foundation, root=root,
                                                     data_root=data, rows_per_file=1))
        build_foundation_lock(foundation, root)
        candidate_roots.append(root)

    template = artifact(
        "RUNTIME_PROFILE", version=2, execution_site=execution_site("sporc_a100"),
        ram_only_views=True, rolling_resume=False, final_test_accessed=False,
        cpus=8, memory_mb=73728, workers=8, train_minutes=808,
        reduce_minutes=43, max_train_minutes=2880,
        cache_budgets=cache_budgets(bottleneck, 73728, 8),
        selected_state_bytes=20_000_000,
        passed=True, measured_full_population=True,
    )
    template_path = tmp_path / "template.json"
    write_immutable_json(template_path, template)
    monkeypatch.setattr(salience_screen, "_source", lambda *args: None)
    monkeypatch.setattr(salience_production, "_source", lambda *args: None)
    screen_root = tmp_path / "screen"
    screen = salience_screen.create_screen(
        bottleneck_root=bottleneck_root, candidate_roots=candidate_roots,
        resource_template=template_path, data_root=data, output_root=screen_root,
        project=tmp_path, source_commit="a" * 40,
    )
    debug_root = tmp_path / "screen_debug"
    debug_screen = salience_screen.create_screen(
        bottleneck_root=bottleneck_root, candidate_roots=candidate_roots,
        resource_template=template_path, data_root=data, output_root=debug_root,
        project=tmp_path, source_commit="a" * 40,
        screen_execution_site_name="sporc_a100_debug",
    )
    debug_plan = load_json(debug_root / "command_plan.json")
    assert debug_screen["contract"] == "JETCLASS2_DELPHES_SALIENCE_SCREEN_SPEC/v2"
    assert debug_screen["production_execution_site"] == execution_site("sporc_a100")
    assert debug_screen["screen_execution_site"] == execution_site("sporc_a100_debug")
    assert len(debug_plan["commands"]) == 8
    for row in debug_plan["commands"]:
        assert "--partition=debug" in row["command"]
        assert row["command"][-1] == "sporc_a100_debug"
        if row["task_id"] == "preflight" or row["task_id"].startswith("fit_"):
            assert "--time=480" in row["command"]
            assert "--gres=gpu:a100:1" in row["command"]
        else:
            assert "--time=60" in row["command"]
            assert not any(token.startswith("--gres=") for token in row["command"])
    with pytest.raises(PermissionError, match="authorization"):
        salience_screen.submit_screen(screen, execute=True,
                                      authorization_phrase="not authorized")
    winner = load_json(candidate_roots[0] / "foundation_spec.json")
    profile = artifact(
        "SALIENCE_RUNTIME_PROFILE", screen_sha256=screen["content_hash"],
        source_commit="a" * 40, execution_site=execution_site("sporc_a100"),
        slurm_job_id="1", cpus=8, memory_mb=73728, workers=8,
        train_minutes=808, reduce_minutes=43, max_train_minutes=2880,
        cache_budgets=template["cache_budgets"], gpu={}, installed_environment={},
        selected_state_bytes=20_000_000,
        cache_seconds_by_candidate={}, peak_train_plus_validation_cache_bytes=1,
        probe_coordinates={},
        model={}, passed=True, ram_only_views=True, rolling_resume=False,
        final_test_accessed=False,
    )
    write_immutable_json(screen_root / "runtime_profile.json", profile)
    selection = artifact(
        "SALIENCE_SELECTION_LOCK", screen_sha256=screen["content_hash"],
        winner=REGISTRY[0], winner_foundation_root=str(candidate_roots[0].resolve()),
        winner_foundation_sha256=winner["content_hash"], candidates=REGISTRY,
        contextual_control="BOTTLENECK_CONTEXT", auc_tolerance=5e-5,
        fit_reports={}, final_test_accessed=False,
    )
    write_immutable_json(screen_root / "selection_lock.json", selection)
    write_immutable_json(screen_root / "screen_complete.json", artifact(
        "SALIENCE_SCREEN_COMPLETE", screen_sha256=screen["content_hash"],
        selection_lock_sha256=selection["content_hash"], scientific_fit_count=4,
        final_test_accessed=False,
    ))
    campaign = salience_production.create_campaign(
        screen_spec_path=screen_root / "screen_spec.json", data_root=data,
        campaign_root=tmp_path / "campaign", project=tmp_path,
        source_commit="a" * 40,
    )
    assert campaign["fresh_fit_count"] == 16
    assert campaign["reducer_count"] == 12
    assert campaign["task_count"] == 30
    assert len(load_json(tmp_path / "campaign/command_plan.json")["commands"]) == 30

    debug_profile = artifact(
        "SALIENCE_RUNTIME_PROFILE", version=2,
        screen_sha256=debug_screen["content_hash"], source_commit="a" * 40,
        execution_site=execution_site("sporc_a100"),
        screen_execution_site=execution_site("sporc_a100_debug"),
        execution_policy=debug_screen["execution_policy"],
        slurm_job_id="2", cpus=8, memory_mb=73728, workers=8,
        train_minutes=808, reduce_minutes=43, max_train_minutes=2880,
        cache_budgets=template["cache_budgets"], gpu={}, installed_environment={},
        selected_state_bytes=20_000_000, cache_seconds_by_candidate={},
        peak_train_plus_validation_cache_bytes=1, probe_coordinates={}, model={},
        passed=True, ram_only_views=True, rolling_resume=False,
        final_test_accessed=False,
    )
    write_immutable_json(debug_root / "runtime_profile.json", debug_profile)
    debug_selection = artifact(
        "SALIENCE_SELECTION_LOCK", screen_sha256=debug_screen["content_hash"],
        winner=REGISTRY[0], winner_foundation_root=str(candidate_roots[0].resolve()),
        winner_foundation_sha256=winner["content_hash"], candidates=REGISTRY,
        contextual_control="BOTTLENECK_CONTEXT", auc_tolerance=5e-5,
        fit_reports={}, final_test_accessed=False,
    )
    write_immutable_json(debug_root / "selection_lock.json", debug_selection)
    write_immutable_json(debug_root / "screen_complete.json", artifact(
        "SALIENCE_SCREEN_COMPLETE", screen_sha256=debug_screen["content_hash"],
        selection_lock_sha256=debug_selection["content_hash"], scientific_fit_count=4,
        final_test_accessed=False,
    ))
    debug_campaign = salience_production.create_campaign(
        screen_spec_path=debug_root / "screen_spec.json", data_root=data,
        campaign_root=tmp_path / "campaign_debug", project=tmp_path,
        source_commit="a" * 40,
    )
    assert debug_campaign["runtime_profile"]["execution_site"] == execution_site("sporc_a100")
    debug_production_plan = load_json(tmp_path / "campaign_debug/command_plan.json")
    assert all("--partition=tier3" in row["command"]
               for row in debug_production_plan["commands"])
    with pytest.raises(PermissionError, match="authorization"):
        salience_production.submit_campaign(
            campaign, bookkeeping_root=tmp_path / "campaign", execute=True,
            authorization_phrase="not authorized",
        )
