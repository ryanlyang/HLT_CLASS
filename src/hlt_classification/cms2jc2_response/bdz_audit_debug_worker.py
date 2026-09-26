"""Execution adapter: unchanged audit kernels, external original acceptance."""
from pathlib import Path

from . import bdz_audit_debug as campaign, bdz_audit_worker as original
from . import bdz_audit_campaign as audit, bdz_audit_metrics as metrics, dev_campaign as dev
from .contracts import artifact, load_json, validate, sha256_file
from .dev_data import checked_file, sample_stream
from .storage import GIB, publish_bytes


def accepted(spec):
    campaign.verify_retirement(spec)
    subject = load_json(checked_file(spec["subject_spec"]))
    row = campaign.acceptance(subject)
    if row["content_hash"] != spec["migration"]["parents"]["acceptance"]:
        raise ValueError("Reused audit acceptance differs")
    return row


def evaluate(ctx, spec, task):
    accepted(spec)
    parent, model, ranges, reg = original.inputs(spec)
    i = task["params"]["shard"]
    with sample_stream(ctx, "evaluation", shard=i) as pairs:
        result, payload = original.run_pairs(pairs, model, reg["mapping"], ranges,
            workers=task["cpus"], count=audit.COUNTS["evaluation"]//4)
    previous = dev.product(parent, f"bz_eval_{i}", "result")
    validate(previous, "BDZ_SHARD", parents={"stage": parent["content_hash"], "registry": reg["content_hash"],
        "ranges": ranges["content_hash"], "samples": ctx["samples"]["content_hash"]})
    original.replay(result, previous)
    return artifact("BDZ_AUDIT_SHARD", parents=original.shard_parents(spec, ctx, reg, ranges, previous),
        shard=i, jets=result["jets"], ordered_identities=result["ordered_identities"],
        source_groups=result["source_groups"], audit=payload, historical_replay=True,
        all_jets_included=True, confirmation_accessed=False)


def report(ctx, spec):
    acceptance = accepted(spec)
    parent, _, ranges, reg = original.inputs(spec)
    payload, hashes, identities, jets = {}, [], set(), 0
    for i in range(4):
        row = dev.product(spec, f"ba_eval_{i}", "result")
        previous = dev.product(parent, f"bz_eval_{i}", "result")
        validate(row, "BDZ_AUDIT_SHARD", parents=original.shard_parents(spec, ctx, reg, ranges, previous))
        if (row["shard"] != i or row["jets"] != audit.COUNTS["evaluation"]//4
                or row["ordered_identities"] != previous["ordered_identities"]
                or row["source_groups"] != previous["source_groups"] or row["ordered_identities"] in identities
                or row["historical_replay"] is not True or row["all_jets_included"] is not True
                or row["confirmation_accessed"] is not False):
            raise ValueError("Debug audit shard population/replay differs")
        metrics.merge(payload, row["audit"])
        hashes.append(row["content_hash"])
        identities.add(row["ordered_identities"])
        jets += row["jets"]
    if jets != audit.COUNTS["evaluation"]:
        raise ValueError("Incomplete debug audit population")
    plots = {}
    for name, blob in original.figures(payload):
        relative = f"figures/{spec['name']}/{name}"
        path = publish_bytes(Path(spec["root"]), relative, blob, remaining_bytes=2*GIB)
        plots[name] = dict(relative=relative, sha256=sha256_file(path), bytes=len(blob))
    return artifact("BDZ_AUDIT_REPORT", parents={"stage": spec["content_hash"],
        "protocol": audit.protocol()["content_hash"], "selection": spec["reuse"]["parents"]["selection"],
        "acceptance": acceptance["content_hash"]}, jets=jets, shard_hashes=hashes, by_file=payload,
        diagnostics=metrics.diagnostics(payload), pooled_proxy_diagnostics=metrics.pooled_proxy_diagnostics(payload),
        figures=plots, protocol=audit.protocol(), historical_choice=spec["reuse"]["historical_choice"],
        development_reused=True, replicas_are_independent=False, confirmation_accessed=False,
        production_qualified=False, transfer_authorized=False, automatic_followup=False)


def dispatch(ctx, spec, task):
    if task["action"] == "ba_evaluate":
        return evaluate(ctx, spec, task)
    if task["action"] == "ba_report":
        return report(ctx, spec)
    raise ValueError("Debug audit cannot rerun acceptance or invoke another task")
