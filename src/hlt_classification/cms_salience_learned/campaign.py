"""Source-pinned, staged SPORC submission; never touches another campaign."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

from hlt_classification.data.cache_contracts import canonical_sha256, load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from hlt_classification.scouting.splits import validate_split_manifest
from .contracts import (
    ACCEPTANCE_POLICY, AUTHORIZATION, COARSE_AUTHORIZATION, BUDGETS, acceptance_policy, allocation_site,
    artifact, graph, site_for_partition, validate, SHARED_TASKS,
)
from .storage import checked_file, fingerprint, load_receipt


def tasks(spec):
    count = sum(len(spec["split_roles"][r]) for r in ("train", "validation"))
    preparation = [dict(task_id="select", kind="select", dependencies=[])]
    preparation += [dict(task_id=f"match_{i:04d}", kind="match", dependencies=["select"], index=i) for i in range(count)]
    preparation += [dict(task_id="calibrate", kind="calibrate", dependencies=[f"match_{i:04d}" for i in range(count)])]
    preparation += [dict(task_id=f"couple_{i:04d}", kind="couple", dependencies=["calibrate", f"match_{i:04d}"], index=i) for i in range(count)]
    preparation += [dict(task_id="foundation", kind="foundation", dependencies=[f"couple_{i:04d}" for i in range(count)])]
    if spec.get("preparation_import") is not None:
        preparation = [dict(task_id="foundation", kind="import_foundation", dependencies=[])]
    science = deepcopy(spec["graph"]["tasks"])
    if spec.get("shared_source") is not None:
        for row in science:
            if row["task_id"] in SHARED_TASKS:
                row["kind"] = "import_shared"
    gate_kind = "import_preflight" if spec.get("acceptance_import") is not None else "preflight"
    return dict(prepare=preparation, gate=[dict(task_id="preflight", kind=gate_kind, dependencies=[])], science=science)


def validate_resources(resources):
    if (set(resources) != {"cpus", "workers", "memory_mb"}
        or any(type(x) is not int for x in resources.values())
        or not 1 <= resources["workers"] <= resources["cpus"] <= 36
        or not 65536 <= resources["memory_mb"] <= 340000):
        raise ValueError("CMS SPORC resource request differs")


def command_plan(spec, stage):
    if stage not in {"prepare", "gate", "science"}:
        raise ValueError("Unknown submission stage")
    commands = []
    for task in tasks(spec)[stage]:
        kind = task["kind"]
        gpu = kind in {"preflight", "train", "reduce", "extract"}
        cpus = spec["resources"]["cpus"] if kind in {"match", "couple", "preflight", "train", "reduce", "extract"} else 4
        memory = spec["resources"]["memory_mb"] if gpu else (32000 if kind in {"match", "couple"} else 16000)
        minutes = 1440 if kind == "train" else 720 if kind in {"preflight", "match", "couple", "calibrate"} else 480
        site = spec["site"]
        command = ["sbatch", "--parsable", f"--account={site['account']}", f"--partition={site['partition']}", f"--qos={site['qos']}",
            "--nodes=1", "--ntasks=1", "--export=ALL", "--no-requeue",
            f"--cpus-per-task={cpus}", f"--mem={memory}M", f"--time={minutes}",
            f"--job-name=cmslfh_{task['task_id']}", f"--chdir={spec['project_dir']}",
            f"--output={spec['campaign_root']}/slurm-%j.out"]
        if gpu:
            command.append("--gres=gpu:a100:1")
        if task["dependencies"]:
            command.append("--dependency=afterok:" + ":".join("${JOB_" + p + "}" for p in task["dependencies"]))
        if kind == "import_shared":
            token = "${SOURCE_JOB_" + task["task_id"] + "}"
            dependency = next((i for i, arg in enumerate(command) if arg.startswith("--dependency=")), None)
            if dependency is None:
                command.append("--dependency=afterok:" + token)
            else:
                command[dependency] += ":" + token
        command += [f"{spec['project_dir']}/sbatch/run_cms_salience_learned.sh", spec["project_dir"],
                    f"{spec['campaign_root']}/campaign_spec.json", task["task_id"]]
        commands.append(dict(task_id=task["task_id"], dependencies=task["dependencies"], command=command))
    return artifact("COMMAND_PLAN", campaign_spec_sha256=spec["content_hash"], stage=stage, commands=commands)


def validate_campaign(spec, *, check_source=False):
    digest = validate(spec, "CAMPAIGN_SPEC")
    if spec["schema_version"] in (4, 5):
        if spec.get("ladder") != "coarse" or "shared_source" not in spec:
            raise ValueError("CMS versions 4/5 require the registered coarse ladder")
    elif "ladder" in spec or "shared_source" in spec:
        raise ValueError("Legacy CMS specs remain dense")
    if spec["schema_version"] == 5:
        if spec.get("acceptance_import") is None or spec.get("shared_source") is None:
            raise ValueError("CMS v5 requires explicit accepted dense preflight reuse")
    elif "acceptance_import" in spec:
        raise ValueError("Legacy campaigns cannot bypass their own preflight")
    if spec["graph"] != graph(spec.get("ladder", "dense")) or spec["budgets"] != BUDGETS:
        raise ValueError("Registered CMS scientific graph differs")
    allocation_site(spec)
    acceptance_policy(spec)
    if spec["schema_version"] == 1 and "preparation_import" in spec:
        raise ValueError("Legacy CMS specs cannot import preparation")
    if spec["schema_version"] >= 2 and "preparation_import" not in spec:
        raise ValueError("CMS preparation mode is required")
    validate_resources(spec["resources"])
    if not re.fullmatch(r"[0-9a-f]{40}", spec["source_commit"]):
        raise ValueError("Exact source commit required")
    split = load_json(checked_file(spec["split_manifest"]))
    validate_split_manifest(split, source_manifest_sha256=split["source_manifest_sha256"])
    if split["content_hash"] != spec["split_manifest"]["content_hash"]:
        raise ValueError("Split semantic identity changed")
    if spec["split_roles"] != {role: split["roles"][role]["files"] for role in BUDGETS}:
        raise ValueError("Split role coverage changed")
    if spec["view_config_sha256"] != view_config_hash(spec["graph"]):
        raise ValueError("CMS view protocol changed")
    if spec["final_test_accessed"] is not False:
        raise PermissionError("Ordinary CMS campaign accessed final test")
    root = Path(spec["campaign_root"]).resolve()
    for external in (Path(spec["data_root"]).resolve(), Path(spec["split_manifest"]["path"]).resolve().parent):
        if root == external or external.is_relative_to(root) or root.is_relative_to(external):
            raise ValueError("New campaign overlaps a read-only input root")
    if check_source:
        validate_source_checkout(spec["project_dir"], expected_commit=spec["source_commit"])
    if spec.get("preparation_import") is not None:
        from .preparation_import import validate_import
        validate_import(spec)
    if spec.get("shared_source") is not None:
        from .shared_import import validate_shared_source
        validate_shared_source(spec)
    if spec.get("acceptance_import") is not None:
        from .preflight_reuse import validate_acceptance_import
        validate_acceptance_import(spec)
    return digest


def view_config_hash(registered_graph):
    return canonical_sha256(dict(matcher=registered_graph["matcher"],
        support="persistent_hlt_skeleton_remove_offline_tail_v1", discrete_seed=1337,
        scales="train_only_evenly_spaced_per_file_at_most4096_v1", balanced="existing_CMS_mass_balanced_v1"))


def create(*, split_manifest, data_root, campaign_root, project_dir, source_commit,
           cpus=16, workers=16, memory_mb=192000, partition="tier3", reuse_preparation_spec=None,
           ladder="dense", reuse_shared_spec=None, reuse_dense_preflight=False):
    root = Path(campaign_root).resolve()
    if root.exists():
        raise FileExistsError("Use a fresh isolated campaign root; existing roots are never overwritten")
    split = load_json(split_manifest)
    validate_split_manifest(split, source_manifest_sha256=split["source_manifest_sha256"])
    for role, budget in BUDGETS.items():
        if split["roles"][role]["mapped_entries"] < budget:
            raise ValueError(f"Insufficient {role} population")
    registered = graph(ladder)
    if reuse_shared_spec is not None and ladder != "coarse":
        raise ValueError("Shared-source reuse is only registered for the coarse replacement")
    if reuse_dense_preflight and (ladder != "coarse" or reuse_shared_spec is None):
        raise ValueError("Preflight reuse requires a coarse shared-source replacement")
    extra = dict(ladder="coarse", shared_source=None) if ladder == "coarse" else {}
    if reuse_dense_preflight:
        extra["acceptance_import"] = None
    version = 5 if reuse_dense_preflight else 4 if ladder == "coarse" else 3
    spec = artifact("CAMPAIGN_SPEC", contract_version=version,
        project_dir=str(Path(project_dir).resolve()), source_commit=source_commit,
        campaign_root=str(root), data_root=str(Path(data_root).resolve()), site=site_for_partition(partition),
        split_manifest=dict(fingerprint(split_manifest), content_hash=split["content_hash"]),
        split_roles={role: split["roles"][role]["files"] for role in BUDGETS}, graph=registered,
        budgets=BUDGETS, resources=dict(cpus=cpus, workers=workers, memory_mb=memory_mb),
        view_config_sha256=view_config_hash(registered), preparation_import=None,
        acceptance_policy=deepcopy(ACCEPTANCE_POLICY), final_test_accessed=False, **extra)
    if reuse_preparation_spec is not None:
        from .preparation_import import build_import
        value = dict(spec)
        value.pop("content_hash")
        value["preparation_import"] = build_import(spec, reuse_preparation_spec)
        spec = artifact("CAMPAIGN_SPEC", contract_version=spec["schema_version"], **value)
    if reuse_shared_spec is not None:
        from .shared_import import build_shared_source
        value = {k: v for k, v in spec.items() if k != "content_hash"}
        value["shared_source"] = build_shared_source(spec, reuse_shared_spec)
        spec = artifact("CAMPAIGN_SPEC", contract_version=version, **value)
    if reuse_dense_preflight:
        from .preflight_reuse import build_acceptance_import
        value = {k: v for k, v in spec.items() if k != "content_hash"}
        value["acceptance_import"] = build_acceptance_import(spec)
        spec = artifact("CAMPAIGN_SPEC", contract_version=version, **value)
    validate_campaign(spec, check_source=True)
    write_immutable_json(root / "campaign_spec.json", spec)
    for stage in ("prepare", "gate", "science"):
        plan = command_plan(spec, stage)
        write_immutable_json(root / f"{stage}_command_plan.json", plan)
        submit_exact_dag(identity=spec["content_hash"], plan=plan,
            output=root / f"{stage}_dry_run_submission_ledger.json",
            canonical_dry_run=root / f"{stage}_dry_run_submission_ledger.json", execute=False)
    return spec


def create_coarse_from_dense(*, source_spec, campaign_root, project_dir, source_commit, reuse_dense_preflight=False):
    """Infer immutable inputs/resources from the explicitly named dense source."""
    from .preparation_import import preparation_spec
    path = Path(source_spec).resolve()
    source = load_json(path)
    validate_campaign(source)
    if path != Path(source["campaign_root"]) / "campaign_spec.json" or source["graph"] != graph():
        raise ValueError("Use the canonical dense source spec")
    producer = preparation_spec(source)
    return create(split_manifest=source["split_manifest"]["path"], data_root=source["data_root"],
        campaign_root=campaign_root, project_dir=project_dir, source_commit=source_commit,
        **source["resources"], partition=source["site"]["partition"], ladder="coarse",
        reuse_preparation_spec=Path(producer["campaign_root"]) / "campaign_spec.json", reuse_shared_spec=path,
        reuse_dense_preflight=reuse_dense_preflight)


def validate_acceptance_resources(spec, value):
    """Check the version-bound limits and complete repeated-update evidence."""
    validate(value, "EXECUTION_ACCEPTANCE")
    policy = acceptance_policy(spec)
    expected_version = 2 if spec["schema_version"] >= 3 else 1
    if value["schema_version"] != expected_version:
        raise ValueError("CMS acceptance version differs from campaign policy")
    for key in ("peak_rss_bytes", "peak_cuda_bytes", "total_cuda_bytes"):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise ValueError("CMS acceptance memory measurement is invalid")
    if (value["peak_rss_bytes"] >= policy["cpu_peak_fraction_limit"] * spec["resources"]["memory_mb"] * 1024**2
        or value["peak_cuda_bytes"] >= policy["cuda_peak_fraction_limit"] * value["total_cuda_bytes"]):
        raise ValueError("CMS acceptance memory headroom is insufficient")
    if expected_version == 1:
        if "acceptance_policy" in value or "withdrawal_probe" in value:
            raise ValueError("Legacy CMS acceptance cannot claim the new policy")
        return
    if value.get("acceptance_policy") != policy:
        raise ValueError("CMS acceptance policy differs from campaign")
    expected = [(role, alpha, step)
        for role in ("fusion_acquisition", "fusion_withdrawal")
        for alpha in policy["withdrawal_probe_alphas"]
        for step in range(1, policy["withdrawal_probe_steps_per_alpha"] + 1)]
    rows = value.get("withdrawal_probe")
    if (not isinstance(rows, list) or len(rows) != len(expected)
        or any(not isinstance(row, dict) for row in rows)):
        raise ValueError("CMS consecutive withdrawal probe is incomplete")
    # Tiny local fixtures reduce BUDGETS; the registered 500k gate requires 256.
    batch_size = min(policy["withdrawal_probe_batch_size"], BUDGETS["train"])
    for row, (role, alpha, step) in zip(rows, expected):
        if (row.get("route") != role or row.get("alpha") != alpha
            or type(row.get("step")) is not int or row["step"] != step
            or type(row.get("batch_size")) is not int or row["batch_size"] != batch_size):
            raise ValueError("CMS consecutive withdrawal probe coverage differs")
        for key in ("peak_cuda_bytes", "peak_rss_bytes"):
            if type(row.get(key)) is not int or not 0 < row[key] <= value[key]:
                raise ValueError("CMS withdrawal probe memory evidence differs")


def gate_check(spec):
    validate_campaign(spec)
    load_receipt(spec, "foundation")
    load_receipt(spec, "preflight")
    if spec.get("acceptance_import") is not None:
        from .preflight_reuse import reused_gate_check
        return reused_gate_check(spec)
    value = load_json(Path(spec["campaign_root"]) / "execution_acceptance.json")
    validate_acceptance_resources(spec, value)
    if (value["campaign_spec_sha256"] != spec["content_hash"] or value["site"] != spec["site"]
        or value["source_commit"] != spec["source_commit"] or value["genuine_allocation"] is not True
        or value["installed_weaver_forward_backward"] is not True
        or value["exact_extraction"] is not True or value["endpoint_parity"] is not True
        or value["full_population_cache_rows"] != {r: BUDGETS[r] for r in ("train", "validation")}
        or value["final_test_accessed"] is not False):
        raise ValueError("CMS SPORC execution acceptance is absent or insufficient")
    proofs = value["miniature_reports"]
    if len(proofs) != 4 or {p["node"]["role"] for p in proofs} != {"reference_ce", "direct_kd", "fusion_acquisition", "fusion_withdrawal"}:
        raise ValueError("Acceptance did not exercise every production loss route")
    for proof in proofs:
        validate(proof, "TRAINING_REPORT")
        if proof["scientific_fit"] is not False or proof["passes"] != 1:
            raise ValueError("Acceptance was confused with a scientific fit")
    return value


def submit(spec, stage, *, execute=False, authorization_phrase=None):
    validate_campaign(spec, check_source=True)
    root = Path(spec["campaign_root"])
    plan = command_plan(spec, stage)
    if load_json(root / f"{stage}_command_plan.json") != plan:
        raise ValueError("Stored command plan changed")
    if execute:
        expected_authorization = COARSE_AUTHORIZATION if spec.get("ladder") == "coarse" else AUTHORIZATION
        if authorization_phrase != expected_authorization:
            raise PermissionError("Explicit exact-campaign authorization is required")
        if stage == "gate":
            load_receipt(spec, "foundation")
        if stage == "science":
            gate_check(spec)
    if stage == "science" and spec.get("shared_source") is not None:
        from .coarse_submission import submit_shared_dag
        return submit_shared_dag(spec, plan, execute=execute)
    return submit_exact_dag(identity=spec["content_hash"], plan=plan,
        output=root / f"{stage}_{'' if execute else 'dry_run_'}submission_ledger.json",
        canonical_dry_run=root / f"{stage}_dry_run_submission_ledger.json", execute=execute)
