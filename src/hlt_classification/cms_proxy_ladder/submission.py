"""Exact-ID Slurm plans for staged gate and science DAG execution."""
from __future__ import annotations

from pathlib import Path
import shlex

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.execution import slurm_options
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag

from .contracts import artifact, validate, write_json
from .gate import (
    AUTHORIZATION as GATE_AUTHORIZATION,
    DEBUG_AUTHORIZATION as DEBUG_GATE_AUTHORIZATION,
    PREFLIGHT_RECOVERY_AUTHORIZATION,
    OSCAR_AUTHORIZATION, OSCAR_DUAL_SLOT_AUTHORIZATION,
    OSCAR_DIRECT_COARSE_AUTHORIZATION,
    validate_gate,
)
from .production import (
    AUTHORIZATION as SCIENCE_AUTHORIZATION,
    DIRECT_COARSE_AUTHORIZATION, validate_campaign,
)


def _walltime(minutes: int) -> str:
    days, remainder = divmod(int(minutes), 1440)
    hours, remainder = divmod(remainder, 60)
    value = f"{hours:02d}:{remainder:02d}:00"
    return f"{days}-{value}" if days else value


def _wrap(project: str, site: dict, mode: str, spec_path: str, task_id: str) -> str:
    worker = f"{project}/scripts/run_jetclass2_cms_proxy_ladder_task.py"
    arguments = (
        f"--{mode}-spec {shlex.quote(spec_path)} --task {shlex.quote(task_id)}"
    )
    return "\n".join((
        "set -euo pipefail",
        f"export PROJECT_DIR={shlex.quote(project)}",
        f"export JC2_SITE={shlex.quote(site['name'])}",
        f"source {shlex.quote(project + '/sbatch/jetclass2_delphes_common.sh')}",
        f"python -s {shlex.quote(worker)} {arguments}",
    ))


def _command(
    *, project: str, output_root: str, mode: str, spec_path: str,
    task: dict, job_prefix: str, site: dict,
) -> list[str]:
    dependency = ":".join(f"${{JOB_{item}}}" for item in task["dependencies"])
    args = [
        *slurm_options(site),
        f"--cpus-per-task={task['cpus']}", f"--mem={task['memory_mb']}M",
        f"--time={_walltime(task['minutes'])}",
        f"--job-name={job_prefix}_{task['task_id']}"[:128],
        f"--chdir={project}", f"--output={output_root}/slurm-%j.out",
    ]
    if task["kind"] == "gpu":
        args.append("--gres=" + site["gres"])
    if dependency:
        args.append("--dependency=afterok:" + dependency)
    args.append("--wrap=" + _wrap(project, site, mode, spec_path, task["task_id"]))
    return args


def gate_plan(spec: dict) -> dict:
    validate_gate(spec)
    site = spec.get("measurement_site", spec["execution_site"])
    path = str(Path(spec["gate_root"]) / "gate_spec.json")
    commands = []
    for task in spec["tasks"]:
        commands.append({
            "task_id": task["task_id"], "dependencies": task["dependencies"],
            "command": _command(
                project=spec["project_dir"], output_root=spec["gate_root"],
                mode="gate", spec_path=path, task=task,
                job_prefix={
                    1: "jc2pxg", 2: "jc2pxd", 3: "jc2pxr",
                    4: "jc2pxo", 5: "jc2pxq", 6: "jc2pxs", 7: "jc2lvg", 8: "jc2ctxg", 9: "jc2crg", 10: "jc2ctx2g", 11: "jc2ctxmg", 12: "jc2chtg",
                }[
                    spec.get("schema_version")
                ],
                site=site,
            ),
        })
    return artifact(
        "COMMAND_PLAN", parents={"subject": spec["content_hash"]},
        mode="gate", commands=commands,
    )


def science_plan(spec: dict) -> dict:
    validate_campaign(spec)
    profile = spec["runtime_profile"]
    site = profile["execution_site"]
    path = str(Path(spec["campaign_root"]) / "campaign_spec.json")
    commands = []
    for row in spec["tasks"]:
        if row["kind"] in {"train", "reduce"}:
            task = {
                **row, "kind": "gpu", "cpus": profile["cpus"],
                "memory_mb": profile["memory_mb"],
                "minutes": profile["train_minutes"] if row["kind"] == "train" else profile["reduce_minutes"],
            }
        else:
            task = {**row, "kind": "cpu", "cpus": 4, "memory_mb": 32_000, "minutes": 60}
        commands.append({
            "task_id": row["task_id"], "dependencies": row["dependencies"],
            "command": _command(
                project=spec["project_dir"], output_root=spec["campaign_root"],
                mode="campaign", spec_path=path, task=task,
                job_prefix=(
                    "jc2cht" if spec.get('schema_version') == 9 else
                    "jc2ctxm" if spec.get("schema_version") == 8 else
                    "jc2ctx2" if spec.get("schema_version") == 7 else
                    "jc2cr" if spec.get("schema_version") == 5 else
                    "jc2ctx" if spec.get("schema_version") == 4 else
                    "jc2lv" if spec.get("schema_version") == 3 else
                    "jc2pxc" if spec.get("schema_version") == 2 else "jc2px"
                ),
                site=site,
            ),
        })
    return artifact(
        "COMMAND_PLAN", parents={"subject": spec["content_hash"]},
        mode="science", commands=commands,
    )


def validate_plan(plan: dict, *, subject: dict, mode: str) -> str:
    digest = validate(plan, "COMMAND_PLAN", parents={"subject": subject["content_hash"]})
    expected = gate_plan(subject) if mode == "gate" else science_plan(subject)
    if plan != expected or plan["mode"] != mode:
        raise ValueError("Proxy-ladder exact command plan differs")
    return digest


def submit(
    *, subject: dict, mode: str, execute: bool, authorization_phrase: str | None,
) -> dict:
    if mode not in {"gate", "science"}:
        raise ValueError("Unknown proxy-ladder submission mode")
    root = Path(subject["gate_root"] if mode == "gate" else subject["campaign_root"])
    plan = gate_plan(subject) if mode == "gate" else science_plan(subject)
    plan_path = root / "command_plan.json"
    if plan_path.exists():
        if load_json(plan_path) != plan:
            raise FileExistsError("Proxy-ladder command plan differs")
    else:
        write_json(plan_path, plan)
    gate_authorizations = {
        1: GATE_AUTHORIZATION,
        2: DEBUG_GATE_AUTHORIZATION,
        3: PREFLIGHT_RECOVERY_AUTHORIZATION,
        4: OSCAR_AUTHORIZATION,
        5: OSCAR_DUAL_SLOT_AUTHORIZATION,
        6: OSCAR_DIRECT_COARSE_AUTHORIZATION,
    }
    from .literature import GATE_AUTHORIZATION as LITERATURE_GATE, SCIENCE_AUTHORIZATION as LITERATURE_SCIENCE
    gate_authorizations[7] = LITERATURE_GATE
    from .context import GATE_AUTHORIZATION as CONTEXT_GATE, SCIENCE_AUTHORIZATION as CONTEXT_SCIENCE
    gate_authorizations[8] = CONTEXT_GATE
    from .correlated import GATE_AUTHORIZATION as CORR_GATE, SCIENCE_AUTHORIZATION as CORR_SCIENCE
    gate_authorizations[9] = CORR_GATE
    if (mode, subject.get("schema_version")) in (("gate", 10), ("science", 7)):
        from .context_v2 import GATE_AUTHORIZATION as V2_GATE, SCIENCE_AUTHORIZATION as V2_SCIENCE
        gate_authorizations[10] = V2_GATE
    from .context_full import GATE_AUTHORIZATION as FULL_GATE, SCIENCE_AUTHORIZATION as FULL_SCIENCE
    gate_authorizations[11] = FULL_GATE
    from .correlated_topology import GATE_AUTHORIZATION as TOPO_GATE, SCIENCE_AUTHORIZATION as TOPO_SCIENCE
    gate_authorizations[12] = TOPO_GATE
    required = (
        gate_authorizations[subject.get("schema_version")]
        if mode == "gate" else (
            TOPO_SCIENCE if subject.get('schema_version') == 9 else
            FULL_SCIENCE if subject.get("schema_version") == 8 else
            V2_SCIENCE if subject.get("schema_version") == 7 else
            CORR_SCIENCE if subject.get("schema_version") == 5 else
            CONTEXT_SCIENCE if subject.get("schema_version") == 4 else
            LITERATURE_SCIENCE if subject.get("schema_version") == 3 else
            DIRECT_COARSE_AUTHORIZATION
            if subject.get("schema_version") == 2 else SCIENCE_AUTHORIZATION
        )
    )
    if execute and authorization_phrase != required:
        raise PermissionError(f"Live {mode} submission requires exact authorization phrase")
    literature = (mode == "gate" and subject.get("schema_version") == 7) or (
        mode == "science" and subject.get("schema_version") == 3)
    context = (mode == "gate" and subject.get("schema_version") in (8, 10, 11)) or (
        mode == "science" and subject.get("schema_version") in (4, 7, 8))
    correlated = (mode == "gate" and subject.get("schema_version") in (9, 12)) or (
        mode == "science" and subject.get("schema_version") in (5, 9))
    if execute and (literature or context or correlated):
        from .literature import check_submission_site, submit_claimed
        if context:
            from .context import check_submission_site, submit_claimed
        if correlated:
            from .correlated import check_submission_site, submit_claimed
        (validate_gate if mode == "gate" else validate_campaign)(subject, check_source=True)
        check_submission_site(plan)
        return submit_claimed(subject, plan, root)
    return submit_exact_dag(
        identity=subject["content_hash"], plan=plan,
        output=root / ("submission_ledger.json" if execute else "dry_run_submission_ledger.json"),
        canonical_dry_run=root / "dry_run_submission_ledger.json",
        execute=execute,
    )


__all__ = [
    "gate_plan", "science_plan", "submit", "validate_plan",
]
