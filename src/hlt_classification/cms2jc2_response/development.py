"""Bounded fit-role-only integration checks; never remote acceptance evidence."""
from __future__ import annotations

from itertools import islice
from pathlib import Path
import time

from .association import policy
from .contracts import artifact, load_json, publish, validate_compatibility
from .provenance import source_record, validate_source
from .readers import iter_cms
from .response import collect, fit_response, Generator


def preview(*, project_dir: Path, preparation_root: Path, cms_root: Path,
            review_path: Path, output_root: Path, jets_per_role: int = 32,
            candidate_id: str = "A_L") -> dict:
    if candidate_id not in {"A_L", "B_L", "C_L"}:
        raise ValueError("Development preview only runs the three low-complexity families")
    if type(jets_per_role) is not int or not 4 <= jets_per_role <= 256:
        raise ValueError("Development preview is restricted to 4..256 jets per internal fit role")
    root = output_root.resolve()
    if root.exists() or root.is_relative_to(cms_root.resolve()) or cms_root.resolve().is_relative_to(root):
        raise ValueError("Development preview requires a fresh root disjoint from raw data")
    inventory = load_json(preparation_root/"cms_inventory.json")
    roles = load_json(preparation_root/"response_roles.json")
    review = load_json(review_path)
    validate_compatibility(review, inventory_hash=inventory["content_hash"])
    source = source_record(project_dir, executable=False)
    rules = policy()
    start = time.monotonic()
    inputs = {}
    for role in ("fit_location", "fit_residual"):
        stream = iter_cms(cms_root, inventory, roles, review, role="response_fit", fit_role=role, chunk_rows=128)
        try:
            inputs[role] = list(islice(stream, jets_per_role))
        finally:
            stream.close()
        if len(inputs[role]) != jets_per_role:
            raise ValueError("Preview role capacity differs")
    loc, lr = collect(inputs["fit_location"], rules, cap=7200, progress_every=8)
    res, rr = collect(inputs["fit_residual"], rules, cap=7200, progress_every=8)
    fitted = fit_response(loc, res, location_report=lr, residual_report=rr, candidate_id=candidate_id,
                          review=review, rules=rules, budget="DEVELOPMENT_PREVIEW", source_hash=source["content_hash"])
    generator = Generator(fitted)
    traces = []
    for pair in inputs["fit_residual"][:4]:
        output, trace = generator(pair.offline, jet=pair.identity, trace=True)
        repeat, _ = generator(pair.offline, jet=pair.identity)
        if output.p4.tobytes() != repeat.p4.tobytes():
            raise AssertionError("Response replay differs")
        traces.append(dict(identity=pair.identity, offline_particles=len(pair.offline),
                           real_hlt_particles=len(pair.hlt), proxy_particles=len(output), audit=trace))
    result = artifact("DEVELOPMENT_PREVIEW", parents={"response": fitted["content_hash"],
                      "source": source["content_hash"], "compatibility": review["content_hash"]},
                      source=source, elapsed_seconds=time.monotonic()-start, traces=traces,
                      location_counts=lr["counts"], residual_counts=rr["counts"],
                      evaluated_on="fit_residual_development_only_not_held_out_closure",
                      production_acceptance=False, science_queue_ready=False,
                      physical_status=fitted["physical_status"], durable_raw_particles=False)
    validate_source(source, project_dir, executable=False)
    root.mkdir(parents=True, exist_ok=False)
    publish(root/"response.json", fitted, "FITTED_RESPONSE")
    publish(root/"development_report.json", result, "DEVELOPMENT_PREVIEW")
    return result
