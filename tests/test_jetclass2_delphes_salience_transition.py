from copy import deepcopy

import pytest

from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.jetclass2_delphes.contracts import artifact
from hlt_classification.jetclass2_delphes import salience_transition
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger


TASKS = ["train_M0HLT", "train_U000", "aggregate"]


def _spec(identity, inventory, root):
    nodes = [
        dict(node_id="M0HLT", coordinate="D000", teacher=None, branch="REFERENCE",
             u=[1, 1], f=[1, 1], initialization_seed=1, sampler_seed=2, deployable=True),
        dict(node_id="U000", coordinate="U000", teacher=None, branch="REFERENCE",
             u=[0, 1], f=[0, 1], initialization_seed=3, sampler_seed=4, deployable=False),
    ]
    scientific = dict(
        nodes=nodes, branches={"DIRECT": []}, probability_publications=[],
        branch_order=["DIRECT"], recipe={"same": True},
        role_counts={"train": 500_000, "validation": 1_000_000, "final_test": 1_000_000},
    )
    return with_content_hash(dict(
        contract="JETCLASS2_DELPHES_SALIENCE_CAMPAIGN_SPEC/v1", schema_version=1,
        campaign_root=root, foundation={"inventory": {"content_hash": inventory}},
        scientific_plan=scientific,
        tasks=[{"task_id": task} for task in TASKS], fresh_fit_count=2,
        reducer_count=0, task_count=3, final_test_accessed=False,
        identity=identity,
    ))


def _ledgers(spec):
    commands = {task: ["sbatch", task] for task in TASKS}
    live = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"],
        jobs={task: str(100 + index) for index, task in enumerate(TASKS)},
        commands=commands, dry_run=False,
    )
    dry = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"],
        jobs={task: "1" for task in TASKS}, commands=commands, dry_run=True,
    )
    plan = artifact(
        "COMMAND_PLAN", campaign_sha256=spec["content_hash"],
        commands=[{"task_id": task, "dependencies": [], "command": commands[task]}
                  for task in TASKS],
    )
    return live, dry, plan


def test_transition_requires_new_dataset_and_exact_dry_graph(monkeypatch):
    monkeypatch.setattr(salience_transition, "validate_campaign",
                        lambda spec: spec["content_hash"])
    old = _spec("old", "1" * 64, "/old")
    new = _spec("new", "2" * 64, "/new")
    old_live, _, _ = _ledgers(old)
    _, new_dry, new_plan = _ledgers(new)
    states = {"100": "COMPLETED", "101": "RUNNING", "102": "PENDING"}
    result = salience_transition.build_transition_plan(
        old_spec=old, old_ledger=old_live, new_spec=new,
        new_dry_ledger=new_dry, new_command_plan=new_plan,
        states_by_job_id=states,
    )
    assert result["exact_job_ids"] == ["101", "102"]
    assert result["new_submission_performed"] is False

    bad = deepcopy(new_dry)
    bad["jobs"] = {"train_M0HLT": "DRY_RUN_0000"}
    with pytest.raises(ValueError):
        salience_transition.build_transition_plan(
            old_spec=old, old_ledger=old_live, new_spec=new,
            new_dry_ledger=with_content_hash(bad), new_command_plan=new_plan,
            states_by_job_id=states,
        )
    same = _spec("new", "1" * 64, "/new")
    _, same_dry, same_plan = _ledgers(same)
    with pytest.raises(ValueError, match="scope differs"):
        salience_transition.build_transition_plan(
            old_spec=old, old_ledger=old_live, new_spec=same,
            new_dry_ledger=same_dry, new_command_plan=same_plan,
            states_by_job_id=states,
        )


def test_transition_refuses_unknown_and_receipts_only_terminal(monkeypatch):
    monkeypatch.setattr(salience_transition, "validate_campaign",
                        lambda spec: spec["content_hash"])
    old = _spec("old", "1" * 64, "/old")
    new = _spec("new", "2" * 64, "/new")
    old_live, _, _ = _ledgers(old)
    _, new_dry, new_plan = _ledgers(new)
    with pytest.raises(ValueError, match="unknown"):
        salience_transition.build_transition_plan(
            old_spec=old, old_ledger=old_live, new_spec=new,
            new_dry_ledger=new_dry, new_command_plan=new_plan,
            states_by_job_id={"100": "COMPLETED", "101": "UNKNOWN", "102": "PENDING"},
        )
    plan = salience_transition.build_transition_plan(
        old_spec=old, old_ledger=old_live, new_spec=new,
        new_dry_ledger=new_dry, new_command_plan=new_plan,
        states_by_job_id={"100": "COMPLETED", "101": "RUNNING", "102": "PENDING"},
    )
    with pytest.raises(ValueError, match="not terminal"):
        salience_transition.build_transition_receipt(
            plan=plan, old_ledger=old_live,
            states_by_job_id={"100": "COMPLETED", "101": "CANCELLED", "102": "RUNNING"},
        )
    receipt = salience_transition.build_transition_receipt(
        plan=plan, old_ledger=old_live,
        states_by_job_id={"100": "COMPLETED", "101": "CANCELLED", "102": "CANCELLED"},
    )
    assert receipt["all_old_jobs_terminal"] is True
