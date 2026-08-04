#!/usr/bin/env python3
"""Monitor only exact PRAD ledger IDs and authenticate task attestations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json  # noqa: E402
from hlt_classification.prad.campaign import validate_prad_campaign_spec, validate_prad_monitor_report, validate_prad_submission_ledger, validate_prad_task_attestation  # noqa: E402
from hlt_classification.provenance import validate_source_snapshot  # noqa: E402


def _states(job_ids: list[str]) -> dict[str, str]:
    process = subprocess.run(
        ["sacct", "-n", "-P", "-j", ",".join(job_ids), "-o", "JobIDRaw,State"],
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[str, str] = {}
    wanted = set(job_ids)
    for line in process.stdout.splitlines():
        raw_id, *tail = line.split("|")
        if raw_id in wanted and tail:
            result[raw_id] = tail[0].strip().split("+", 1)[0]
    missing = wanted - set(result)
    if missing:
        raise RuntimeError(f"sacct omitted exact PRAD IDs: {sorted(missing)}")
    return result


def _array_indices(value: str | None) -> list[int | None]:
    if value is None:
        return [None]
    match = re.fullmatch(r"([0-9]+)-([0-9]+)(?:%[0-9]+)?", value)
    if match is None:
        raise ValueError(f"unsupported PRAD array expression {value!r}")
    return list(range(int(match.group(1)), int(match.group(2)) + 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--states-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    validate_prad_campaign_spec(spec)
    validate_source_snapshot(spec["source_snapshot"], repository=args.repository, require_clean=True)
    ledger = load_json(args.submission_ledger)
    ledger_hash = validate_prad_submission_ledger(ledger, spec=spec)
    states = (
        _states([row["job_id"] for row in ledger["jobs"]])
        if args.states_json is None
        else load_json(args.states_json)
    )
    if set(states) != {row["job_id"] for row in ledger["jobs"]}:
        raise ValueError("PRAD monitor states do not cover exact ledger IDs")
    task_lookup = {row["name"]: row for row in spec["tasks"]}
    root = Path(spec["site"]["campaign_root"])
    rows = []
    for job in ledger["jobs"]:
        attestations = []
        for index in _array_indices(task_lookup[job["task"]]["array"]):
            suffix = "" if index is None else f"_{index}"
            path = root / "task_attestations" / f"{job['task']}{suffix}.json"
            if not path.is_file():
                continue
            payload = load_json(path)
            expected_array_id = None if index is None else str(index)
            attestations.append(
                validate_prad_task_attestation(
                    payload,
                    campaign_spec_sha256=spec["content_hash"],
                    task=job["task"],
                    array_task_id=expected_array_id,
                )
            )
        expected = len(_array_indices(task_lookup[job["task"]]["array"]))
        complete = states[job["job_id"]] == "COMPLETED" and len(attestations) == expected
        rows.append({"task": job["task"], "job_id": job["job_id"], "state": states[job["job_id"]], "attestations": attestations, "reusable": complete})
    report = with_content_hash(
        {
            "contract": "hlt_classification_prad_monitor_report_v1",
            "schema_version": 1,
            "campaign_spec_sha256": spec["content_hash"],
            "submission_ledger_sha256": ledger_hash,
            "jobs": rows,
        }
    )
    validate_prad_monitor_report(report, spec=spec, ledger=ledger)
    write_immutable_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
