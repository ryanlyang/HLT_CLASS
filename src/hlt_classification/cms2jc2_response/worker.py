"""Common measured fitting engine for bounded probes and full registered fits.

This engine does not grant Slurm submission, confirmation, or final-test access.
The stage coordinator must authenticate its source and context before calling.
"""
from __future__ import annotations

from pathlib import Path
import time

from .association import policy
from .audit import validate_inventory
from .contracts import artifact, validate_compatibility, candidates
from .parallel import parallel_collect
from .response import fit_response
from .splits import validate_roles, validate_memberships


def fit_task(*, cms_root: Path, inventory: dict, roles: dict, membership: dict, review: dict,
             source_hash: str, candidate_id: str, budget: str, gate: float = .1,
             cpus: int = 16, cap: int = 2_000_000, probe_jets: int | None = None):
    validate_inventory(inventory)
    validate_roles(roles, inventory)
    validate_memberships(membership, roles)
    validate_compatibility(review, inventory_hash=inventory["content_hash"])
    registry = candidates()
    if candidate_id not in {r["id"] for r in registry["candidates"]} or budget not in registry["budgets"]:
        raise ValueError("Unregistered response fit")
    if type(cpus) is not int or not 1 <= cpus <= 16:
        raise ValueError("Fit CPU envelope differs")
    if probe_jets is not None and probe_jets not in (20_000, 100_000):
        raise ValueError("Only registered 20k/100k engine probes are allowed")
    rules = policy(gate=gate)
    started = time.monotonic()
    phase, data, reports = {}, {}, {}
    for role, fraction in (("fit_location", .8), ("fit_residual", .2)):
        begin = time.monotonic()
        records, report = parallel_collect(root=cms_root, inventory=inventory, roles=roles, review=review,
                  membership=membership, budget=budget, fit_role=role, rules=rules,
                  workers=cpus, cap=cap, jet_limit=int(probe_jets*fraction) if probe_jets else None)
        expected = (int(probe_jets*fraction) if probe_jets else
                    sum(r["selected_entries"] for r in membership["budgets"][budget][role]))
        if report["counts"]["jets"] != expected:
            raise ValueError("Calibration worker did not consume the declared population")
        data[role], reports[role] = records, report
        phase[role+"_records"] = time.monotonic()-begin
    begin = time.monotonic()
    from threadpoolctl import threadpool_limits
    # All child ROOT readers have exited before numerical fitting gets threads.
    with threadpool_limits(limits=cpus):
        response = fit_response(data["fit_location"], data["fit_residual"],
                    location_report=reports["fit_location"], residual_report=reports["fit_residual"],
                    candidate_id=candidate_id, review=review, rules=rules,
                    budget=budget if probe_jets is None else f"PROBE_{probe_jets}_{budget}",
                    source_hash=source_hash)
    phase["fitting"] = time.monotonic()-begin
    result = artifact("FIT_TASK_REPORT", parents={"source": source_hash,
                      "response": response["content_hash"], "roles": roles["content_hash"],
                      "membership": membership["content_hash"], "compatibility": review["content_hash"]},
                      candidate_id=candidate_id, budget=budget, gate=gate, probe_jets=probe_jets,
                      phase_seconds=phase, runtime_seconds=time.monotonic()-started,
                      cpus=cpus, cap_per_role_module=cap, role_counts={r: v["counts"] for r, v in reports.items()},
                      full_registered_fit=probe_jets is None, rolling_resume=False,
                      durable_calibration_records=False, physical_status=response["physical_status"])
    return response, result
