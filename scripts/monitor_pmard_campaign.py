#!/usr/bin/env python3
"""Query only exact PMARD ledger job IDs and publish an authenticated report."""

from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, validate_content_hash, with_content_hash, write_immutable_json  # noqa: E402
from hlt_classification.scouting.campaign import PMARD_LEDGER_CONTRACT, validate_pmard_campaign_spec  # noqa: E402


def _array_ids(value):
    if value is None: return [None]
    import re
    match = re.fullmatch(r"(\d+)-(\d+)(?:%\d+)?", value)
    if not match: raise ValueError("unsupported PMARD array expression")
    return list(range(int(match.group(1)), int(match.group(2)) + 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states-json", type=Path)
    args = parser.parse_args(); spec = load_json(args.campaign_spec); validate_pmard_campaign_spec(spec)
    ledger = load_json(args.submission_ledger); ledger_hash = validate_content_hash(ledger, expected_contract=PMARD_LEDGER_CONTRACT)
    job_ids = list(ledger["jobs"].values())
    if args.states_json:
        states = load_json(args.states_json)
    else:
        process = subprocess.run(["sacct", "-n", "-P", "-j", ",".join(job_ids), "-o", "JobIDRaw,State"], check=True, capture_output=True, text=True)
        states = {line.split("|")[0]: line.split("|")[1].split("+")[0] for line in process.stdout.splitlines() if line.split("|")[0] in set(job_ids)}
    if set(states) != set(job_ids): raise ValueError("monitor did not cover the exact ledger IDs")
    task_lookup = {task["name"]: task for task in spec["tasks"]}; root = Path(spec["campaign_root"])
    rows = []
    for task, job in ledger["jobs"].items():
        attestations = []
        for array_id in _array_ids(task_lookup[task].get("array")):
            suffix = "" if array_id is None else f"_{array_id}"
            path = root / "task_attestations" / f"{task}{suffix}.json"
            if path.is_file():
                payload = load_json(path)
                validate_content_hash(payload, expected_contract="hlt_classification_pmard_task_attestation_v1")
                if (payload.get("contract") != "hlt_classification_pmard_task_attestation_v1"
                        or payload.get("campaign_spec_sha256") != spec["content_hash"]
                        or payload.get("task") != task
                        or payload.get("array_task_id") != (None if array_id is None else str(array_id))):
                    raise ValueError("PMARD task attestation lineage differs")
                for output in payload.get("outputs", ()):
                    output_path = Path(output["path"]).resolve()
                    try: output_path.relative_to(root.resolve())
                    except ValueError as error: raise ValueError("PMARD attested output escapes campaign root") from error
                    from hlt_classification.data.cache_contracts import sha256_file
                    if not output_path.is_file() or sha256_file(output_path) != output.get("sha256"):
                        raise ValueError("PMARD attested output is absent or corrupt")
                attestations.append(payload["content_hash"])
        reusable = states[job] == "COMPLETED" and len(attestations) == len(_array_ids(task_lookup[task].get("array")))
        rows.append({"task": task, "job_id": job, "state": states[job], "attestations": attestations, "reusable": reusable})
    report = with_content_hash({"contract": "hlt_classification_pmard_monitor_v1", "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"], "ledger_sha256": ledger_hash,
        "jobs": rows})
    write_immutable_json(args.output, report); print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
