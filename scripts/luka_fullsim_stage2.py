#!/usr/bin/env python
"""Explicit audit, convention registration, preparation and technical GPU check."""
import argparse
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json, write_immutable_json, validate_content_hash
from hlt_classification.luka_fullsim import stage2 as s
from hlt_classification.luka_fullsim.contracts import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for mode in ("audit", "prepare", "preflight", "preflight-v2"):
        p = commands.add_parser(mode)
        p.add_argument("--expected-commit", required=True)
        p.add_argument("--output", required=True, type=Path)
        if mode in ("preflight", "preflight-v2"):
            p.add_argument("--prepared", required=True, type=Path)
            p.add_argument("--site", choices=("sporc_a100", "sporc_a100_debug"), required=True)
            if mode == "preflight-v2":
                p.add_argument("--prepared-project", required=True, type=Path)
        else:
            p.add_argument("--foundation", required=True, type=Path)
        if mode == "prepare":
            p.add_argument("--audit", required=True, type=Path)
            p.add_argument("--conventions", required=True, type=Path)
    p = commands.add_parser("conventions")
    p.add_argument("--audit", required=True, type=Path)
    p.add_argument("--offline-unit", choices=("mm", "cm"), required=True)
    p.add_argument("--hlt-unit", choices=("mm", "cm"), required=True)
    p.add_argument("--zero-error", choices=("error_only_unavailable", "value_and_error_unavailable"), required=True)
    p.add_argument("--authority", choices=("producer_confirmed", "operator_provisional"), required=True)
    p.add_argument("--evidence", required=True)
    p.add_argument("--output", required=True, type=Path)
    p = commands.add_parser("summary")
    p.add_argument("path", type=Path)
    a = parser.parse_args()
    if a.command == "summary":
        report = load_json(a.path)
        kind, version = report["contract"].removeprefix("LUKA_FULLSIM_").rsplit("/", 1)
        if kind not in ("PARTICLE_AUDIT", "PREPARED", "GPU_PREFLIGHT"):
            raise ValueError("Unknown stage-2 report")
        if kind == "GPU_PREFLIGHT" and version == "v2":
            validate_content_hash(report, expected_contract="LUKA_FULLSIM_GPU_PREFLIGHT/v2")
        elif version == "v1":
            validate(report, kind)
        else:
            raise ValueError("Unknown stage-2 report version")
        print("Report:", kind, "Counts:", report["counts"])
        if kind == "PARTICLE_AUDIT":
            print("STORED UNCONFIRMED UNITS; zeros do not establish sentinel semantics. SD is width.")
            for role, stats in report["statistics"].items():
                print("\n", role, "issues:", report["issues"][role])
                print(f"{'side / field (charged tracks)':<34} {'mean':>12} {'SD':>12} {'zeros':>10} {'negative':>10} {'nonfinite':>10}")
                for side in ("offline", "hlt"):
                    print(side, "mean multiplicity:", stats[side + "/multiplicity"]["mean"])
                    for field in ("d0val", "dzval", "d0err", "dzerr"):
                        r = stats[f"{side}/charged/{field}"]
                        mean, sd = r["mean"], r["sd"]
                        print(f"{side+'/'+field:<34} {str(mean):>12} {str(sd):>12} {r['zero']:>10} {r['negative']:>10} {r['nonfinite']:>10}")
        else:
            for key in ("summaries", "technical_checks_passed", "resource_envelope_ok", "fits_debug_24h",
                        "projected_train_minutes", "projected_reduce_minutes"):
                if key in report:
                    print(key, report[key])
        print("Saved metadata inspection only; no test particles or jobs accessed.")
        return
    if a.command == "conventions":
        report = load_json(a.audit)
        validate(report, "PARTICLE_AUDIT")
        result = s.conventions(audit_sha256=report["content_hash"], offline_unit=a.offline_unit,
            hlt_unit=a.hlt_unit, zero_error=a.zero_error, authority=a.authority, evidence=a.evidence)
        write_immutable_json(a.output, result)
        print("Explicit conventions saved:", a.output)
        return
    project = Path(__file__).resolve().parents[1]
    kwargs = dict(project=project, expected_commit=a.expected_commit)
    if a.command == "audit":
        result = s.audit(a.foundation, a.output, **kwargs)
    elif a.command == "prepare":
        result = s.prepare(a.foundation, a.audit, a.conventions, a.output, **kwargs)
    elif a.command == "preflight-v2":
        from hlt_classification.luka_fullsim.preflight_v2 import run
        result = run(a.prepared, a.output, site_name=a.site, prepared_project=a.prepared_project, **kwargs)
    else:
        from hlt_classification.luka_fullsim.preflight import run
        result = run(a.prepared, a.output, site_name=a.site, **kwargs)
    print("Completed:", result["contract"], result["content_hash"], "Output:", a.output)
    print("No science jobs submitted. Final test remains sealed.")


if __name__ == "__main__":
    main()
