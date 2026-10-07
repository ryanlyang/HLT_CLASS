"""Frozen CONTEXT_V1 relocated reader and Oscar 100k/50k campaign variant."""
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.literature_context_consumer import RelocatedDataset, MANIFEST_SHA256
from hlt_classification.literature_context_production.contracts import validate as validate_producer
from hlt_classification.jetclass2_delphes.execution import execution_site, validate_resources
from hlt_classification.jetclass2_delphes.model import model_contract
from .contracts import artifact, file_ref, validate, validate_file_ref, write_json

COUNTS = {"train": 100_000, "validation": 50_000}
DOMAIN = "JETCLASS2_CONTEXT_OSCAR_100K_50K/v1"
KIND = "controlled_context_synthetic_proxy"
GATE_AUTHORIZATION = "AUTHORIZE JETCLASS2 CONTEXT 100K OSCAR GATE"
SCIENCE_AUTHORIZATION = "AUTHORIZE JETCLASS2 CONTEXT 100K OSCAR DIRECT COARSE SCIENCE"
SOURCE_FILES = (
    "docs/plans/JETCLASS2_CONTEXT_OSCAR_100K_DIRECT_COARSE_PLAN.md",
    "docs/contracts/JETCLASS2_CONTEXT_OSCAR_LADDER.md",
    "scripts/jetclass2_context_ladder.py",
    "scripts/queue_jetclass2_context_ladder.sh",
    "tests/test_context_ladder.py",
    "tests/test_literature_context_consumer.py",
    "src/hlt_classification/literature_context_consumer.py",
    "src/hlt_classification/scouting/hcwdl_exact_dag_submission.py",
)


def reader(request):
    return RelocatedDataset(proxy_root=request["study_root"],
        offline_root=request["offline_root"], provenance_root=request["provenance_root"],
        expected_manifest_sha256=request["manifest_hash"])


def release_request(*, study_root, offline_root, provenance_root):
    paths = dict(study_root=str(Path(study_root).resolve(strict=True)),
        offline_root=str(Path(offline_root).resolve(strict=True)),
        provenance_root=str(Path(provenance_root).resolve(strict=True)),
        manifest_hash=MANIFEST_SHA256)
    dataset = reader(paths)
    # Authenticate all ordinary metadata now; particle banks are handled by
    # release construction, never by trusting path existence or a copy log.
    for role in COUNTS:
        for shard in dataset._shards(role, None):
            dataset._receipt(shard)
    return _request(paths, file_ref(Path(paths["study_root"])/"study_spec.json"),
                    file_ref(Path(paths["study_root"])/"dataset_manifest.json"))


def _request(paths, study_ref, manifest_ref):
    return artifact("RELEASE_REQUEST", version=3, **paths, counts=dict(COUNTS),
        study_ref=study_ref, manifest_ref=manifest_ref, dataset_kind=KIND,
        recipe="CONTEXT_V1", selection_domain=DOMAIN, labels_read=False,
        selection_depends_on_labels=False, allowed_roles=list(COUNTS))


def validate_request(value):
    digest = validate(value, "RELEASE_REQUEST", version=3)
    root = Path(value["study_root"]).resolve()
    path = validate_file_ref(value["study_ref"])
    manifest_path = validate_file_ref(value["manifest_ref"])
    if path.resolve() != root/"study_spec.json" or manifest_path.resolve() != root/"dataset_manifest.json":
        raise ValueError("Context request source paths differ")
    study, manifest = load_json(path), load_json(manifest_path)
    validate_producer(study, "STUDY", test=False)
    validate_producer(manifest, "MANIFEST", test=True)
    if (study["kind"] != KIND or manifest["content_hash"] != MANIFEST_SHA256
            or value["manifest_hash"] != MANIFEST_SHA256
            or manifest["parents"]["study"] != study["content_hash"]
            or manifest["recipe"] != "CONTEXT_V1"):
        raise ValueError("Context request dataset endpoint differs")
    paths = {k: value[k] for k in ("study_root", "offline_root", "provenance_root", "manifest_hash")}
    if value != _request(paths, value["study_ref"], value["manifest_ref"]):
        raise ValueError("Context request recipe/population/source differs")
    return digest


def validate_release_source(value):
    request = value["request"]
    study = load_json(validate_file_ref(request["study_ref"]))
    if (value["study_contract"] != "JC2_CONTEXT_PRODUCTION_STUDY/v1"
            or value["parents"]["study"] != study["content_hash"]
            or value["study_root"] != request["study_root"]
            or value["offline_root"] != request["offline_root"]):
        raise ValueError("Context release has wrong generator/source")
    expected = {s["shard_id"]: s["role"] for s in study["shards"] if s["role"] in COUNTS}
    if {r["shard_id"]: r["role"] for r in value["receipts"]} != expected:
        raise ValueError("Context release must freeze all ordinary shard receipts")


def gate_tasks():
    return [
        dict(task_id="authenticate_release", kind="cpu", dependencies=[],
             cpus=4, memory_mb=32000, minutes=120),
        dict(task_id="build_foundation", kind="cpu", dependencies=["authenticate_release"],
             cpus=6, memory_mb=90000, minutes=480),
        dict(task_id="preflight", kind="gpu", dependencies=["build_foundation"],
             cpus=6, memory_mb=90000, minutes=720),
    ]


def create_gate(*, study_root, offline_root, provenance_root, gate_root, project_dir, source_commit):
    from .gate import source_lock
    root = Path(gate_root).resolve()
    project = Path(project_dir).resolve(strict=True)
    if root.exists():
        raise FileExistsError("Fresh literature gate root required")
    source = source_lock(project, source_commit, literature=True, context=True)
    request = release_request(study_root=study_root, offline_root=offline_root, provenance_root=provenance_root)
    for protected in (Path(study_root).resolve(), Path(offline_root).resolve(), Path(provenance_root).resolve(), project):
        if root.is_relative_to(protected) or protected.is_relative_to(root):
            raise PermissionError("Gate output overlaps protected input/worktree")
    spec = artifact("GATE_SPEC", version=8,
        parents={"source": source["content_hash"], "request": request["content_hash"]},
        source=source, request=request, gate_root=str(root), project_dir=str(project),
        source_commit=source_commit, capacity=512, execution_site=execution_site("oscar_l40s"),
        tasks=gate_tasks(), workers=6, foundation_workers=6,
        scientific_branches=["DIRECT", "COARSE"], dataset_kind=KIND,
        full_views_persisted=False, site_transfer_policy=None,
        admission="fresh_literature_release_matching_then_full_population_oscar_preflight")
    validate_gate(spec, check_source=True)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "gate_spec.json", spec)
    return spec


def validate_gate(spec, *, check_source=False):
    from .gate import source_lock
    digest = validate(spec, "GATE_SPEC", version=8, parents={
        "source": spec["source"]["content_hash"], "request": spec["request"]["content_hash"],
    })
    validate(spec["source"], "SOURCE")
    validate_request(spec["request"])
    if (spec["source_commit"] != spec["source"]["commit"]
            or spec["request"]["schema_version"] != 3 or spec["dataset_kind"] != KIND
            or spec["execution_site"] != execution_site("oscar_l40s")
            or spec["tasks"] != gate_tasks() or spec["workers"] != 6
            or spec["foundation_workers"] != 6 or spec["capacity"] != 512
            or spec["scientific_branches"] != ["DIRECT", "COARSE"]
            or spec["site_transfer_policy"] is not None or "measurement_site" in spec
            or spec["full_views_persisted"] is not False):
        raise ValueError("Context Oscar gate semantics differ")
    if check_source and source_lock(Path(spec["project_dir"]), spec["source_commit"], literature=True, context=True) != spec["source"]:
        raise ValueError("Literature gate source differs")
    return digest


def scientific_plan(foundation, *, foundation_root=None):
    from .campaign import _build_scientific_plan, DIRECT_COARSE_BRANCHES
    if (foundation.get("schema_version") != 3 or foundation["release"].get("schema_version") != 3
            or foundation["role_counts"] != COUNTS):
        raise ValueError("Literature plan requires the fresh 100k/50k literature foundation")
    return _build_scientific_plan(foundation, registered_branches=DIRECT_COARSE_BRANCHES,
        version=4, foundation_root=foundation_root, node_prefix="CTXV1")


def validate_profile(profile, *, foundation, spec, _version=8, _plan_builder=None):
    from .cache import cache_budgets
    from hlt_classification.jetclass2_delphes.campaign import recipe
    from hlt_classification.jetclass2_delphes.contracts import validate as validate_jc2
    digest = validate(profile, "RUNTIME_PROFILE", version=_version, parents={
        "gate": spec["content_hash"], "foundation": foundation["content_hash"],
    })
    parity = profile["installed_weaver_parity"]
    validate_jc2(parity, "WEAVER_PARITY")
    validate_jc2(profile["installed_environment"], "INSTALLED_ENVIRONMENT", version=2)
    report = profile["acceptance_training_report"]
    validate_jc2(report, "KERNEL_TRAINING_REPORT")
    expected_node = next(n for n in (_plan_builder or scientific_plan)(foundation)["nodes"] if n["node_id"] == "U000")
    import math
    kd = profile["acceptance_kd_training_report"]
    validate_jc2(kd, "KERNEL_TRAINING_REPORT")
    if (kd["node"] != dict(expected_node, node_id="PREFLIGHT_CONTEXT_KD", teacher="PREFLIGHT_U000_CE")
            or kd["foundation_sha256"] != foundation["content_hash"]
            or kd["recipe_sha256"] != recipe()["content_hash"]
            or kd["scientific_fit"] is not False or kd["acceptance_only"] is not True
            or kd["passes"] != 1 or kd["final_test_accessed"] is not False
            or not math.isfinite(kd["runtime_seconds"]) or kd["runtime_seconds"] <= 0
            or not math.isfinite(report["runtime_seconds"]) or report["runtime_seconds"] <= 0):
        raise ValueError("Context KD acceptance differs")
    one_pass = max(report["runtime_seconds"], kd["runtime_seconds"])
    caches = profile["cache_seconds_by_coordinate"]
    if (set(caches) != {"U000", "D050"} or any(not math.isfinite(v) or v < 0 for v in caches.values())
            or not math.isfinite(one_pass) or one_pass <= 0
            or not math.isfinite(profile["inference_seconds"]) or profile["inference_seconds"] <= 0):
        raise ValueError("Literature timing measurements differ")
    train_minutes = max(60, math.ceil((max(caches.values()) + 100 * one_pass) * 1.75 / 60))
    reduce_minutes = max(30, math.ceil((max(caches.values()) + profile["inference_seconds"]) * 2 / 60))
    if (spec.get("schema_version") != _version or foundation["release"]["request"] != spec["request"]
            or foundation["role_counts"] != COUNTS or profile["measured_role_counts"] != COUNTS
            or profile["execution_site"] != execution_site("oscar_l40s")
            or profile.get("site_transfer_policy") is not None
            or profile.get("measurement_site") is not None
            or profile["cpus"] != 6 or profile["workers"] != 6 or profile["memory_mb"] != 90000
            or parity["passed"] is not True or parity["forward_and_feature_and_parameter_gradients"] is not True
            or parity["model"] != model_contract() or parity["device"] != "cuda"
            or parity["final_test_accessed"] is not False
            or profile["source_commit"] != spec["source_commit"]
            or report["foundation_sha256"] != foundation["content_hash"]
            or report["node"] != expected_node or report["recipe_sha256"] != recipe()["content_hash"]
            or report["scientific_fit"] is not False or report["acceptance_only"] is not True
            or report["passes"] != 1 or report["final_test_accessed"] is not False
            or profile["installed_environment"]["architecture"] != "x86_64"
            or profile["one_pass_seconds"] != one_pass
            or profile["train_minutes"] != train_minutes or profile["reduce_minutes"] != reduce_minutes
            or profile["foundation_sha256"] != foundation["content_hash"]
            or profile["model"] != model_contract() or profile["passed"] is not True
            or profile["measured_full_population"] is not True
            or profile["ram_only_views"] is not True or profile["rolling_resume"] is not False
            or not 60 <= profile["train_minutes"] <= 2880
            or not 30 <= profile["reduce_minutes"] <= 1440
            or "L40S" not in profile["gpu"]["name"]
            or not 0 < profile["gpu_peak_bytes"] <= .85 * profile["gpu"]["total_memory_bytes"]
            or profile["cache_budgets"] != cache_budgets(foundation, 90000, 6)):
        raise ValueError("Literature full-population measured Oscar profile differs")
    validate_resources(profile["execution_site"], 6, 90000, 6)
    return digest


def check_submission_site(plan):
    """Read-only scheduler feasibility checks, with inherited requests removed."""
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))}
    cluster = subprocess.run(["scontrol", "show", "config"], capture_output=True,
                             text=True, check=True, env=env).stdout
    import re
    if not re.search(r"(?m)^\s*ClusterName\s*=\s*slurmctld\s*$", cluster):
        raise PermissionError("Submit this campaign from the Oscar cluster")
    seen = set()
    for row in plan["commands"]:
        command = row["command"]
        shape = tuple(a for a in command if a.startswith(("--cpus-per-task=", "--mem=", "--time=", "--gres=")))
        if shape in seen:
            continue
        seen.add(shape)
        args = [a for a in command if not a.startswith(("--dependency=", "--wrap="))]
        args[1:1] = ["--test-only"]
        args.append("--wrap=true")
        result = subprocess.run(args, capture_output=True, text=True, env=env)
        if result.returncode:
            raise PermissionError("Oscar rejected resource shape: " + result.stdout + result.stderr)


def submit_claimed(subject, plan, root):
    """A single live submitter; ambiguous interruptions require explicit repair.

    Keep the claim after success/failure. A completed immutable ledger can be
    inspected idempotently; never auto-retry an unjournaled accepted sbatch.
    """
    import json
    import os
    from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
    from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
    root = Path(root)
    dry_path = root / "dry_run_submission_ledger.json"
    expected = build_submission_ledger(campaign_spec_sha256=subject["content_hash"],
        jobs={r["task_id"]: "1" for r in plan["commands"]},
        commands={r["task_id"]: r["command"] for r in plan["commands"]}, dry_run=True)
    if load_json(dry_path) != expected:
        raise ValueError("Canonical reviewed dry ledger differs")
    ledger = root / "submission_ledger.json"
    if not ledger.is_file():
        claim = root / "live_submission_claim.json"
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise PermissionError("Live submission active or interrupted; preserve journals and inspect before repair") from exc
        with os.fdopen(fd, "w") as handle:
            json.dump(artifact("SUBMISSION_CLAIM", parents={"subject": subject["content_hash"],
                      "plan": plan["content_hash"]}), handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))}
    return submit_exact_dag(identity=subject["content_hash"], plan=plan, output=ledger,
        canonical_dry_run=dry_path, execute=True, environment=env)


def create_campaign(*, gate_root, campaign_root, _adapter=None):
    from .data import validate_foundation
    from .campaign import task_graph
    gate_root = Path(gate_root).resolve(strict=True)
    gate = load_json(gate_root / "gate_spec.json")
    check_gate = _adapter.validate_gate if _adapter else validate_gate
    check_profile = _adapter.validate_profile if _adapter else validate_profile
    make_plan = _adapter.scientific_plan if _adapter else scientific_plan
    check_campaign = _adapter.validate_campaign if _adapter else validate_campaign
    check_gate(gate, check_source=True)
    foundation_root = gate_root / "foundation"
    foundation = load_json(foundation_root / "foundation.json")
    validate_foundation(foundation, root=foundation_root)
    profile = load_json(gate_root / "evidence/runtime_profile.json")
    check_profile(profile, foundation=foundation, spec=gate)
    _completed_gate(gate, foundation, profile)
    root = Path(campaign_root).resolve()
    protected = [gate_root, Path(gate["project_dir"]), Path(gate["request"]["study_root"]),
                 Path(gate["request"]["offline_root"]), Path(gate["request"]["provenance_root"])]
    if root.exists() or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise FileExistsError("Fresh separate literature campaign root required")
    plan = make_plan(foundation, foundation_root=foundation_root)
    spec = artifact("CAMPAIGN_SPEC", version=7 if _adapter else 4,
        parents={"gate": gate["content_hash"], "foundation": foundation["content_hash"],
                 "profile": profile["content_hash"], "plan": plan["content_hash"]},
        gate_root=str(gate_root), campaign_root=str(root), project_dir=gate["project_dir"],
        source_commit=gate["source_commit"], source=gate["source"],
        foundation_root=str(foundation_root), foundation=foundation,
        runtime_profile=profile, scientific_plan=plan, tasks=task_graph(plan), model=model_contract(),
        fresh_fit_count=9, reducer_count=5, selected_branches=["DIRECT", "COARSE"],
        dataset_kind=_adapter.KIND if _adapter else KIND, full_views_persisted=False, existing_campaign_mutations=False)
    check_campaign(spec, check_source=True)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "campaign_spec.json", spec)
    return spec


def _completed_gate(gate, foundation, profile):
    complete = load_json(Path(gate["gate_root"]) / "gate_complete.json")
    validate(complete, "GATE_COMPLETE", parents={"gate": gate["content_hash"],
        "foundation": foundation["content_hash"], "profile": profile["content_hash"]})
    if (complete["passed"] is not True or complete["source_commit"] != gate["source_commit"]
            or complete["release_sha256"] != foundation["release"]["content_hash"]
            or complete["foundation_sha256"] != foundation["content_hash"]
            or complete["runtime_profile_sha256"] != profile["content_hash"]):
        raise ValueError("Literature gate completion differs")


def validate_campaign(spec, *, check_source=False, _adapter=None):
    from .data import validate_foundation
    from .campaign import task_graph
    digest = validate(spec, "CAMPAIGN_SPEC", version=7 if _adapter else 4, parents={
        "gate": spec["parents"]["gate"], "foundation": spec["foundation"]["content_hash"],
        "profile": spec["runtime_profile"]["content_hash"], "plan": spec["scientific_plan"]["content_hash"],
    })
    gate = load_json(Path(spec["gate_root"]) / "gate_spec.json")
    (_adapter.validate_gate if _adapter else validate_gate)(gate, check_source=check_source)
    foundation = spec["foundation"]
    validate_foundation(foundation, root=Path(spec["foundation_root"]))
    (_adapter.validate_profile if _adapter else validate_profile)(spec["runtime_profile"], foundation=foundation, spec=gate)
    _completed_gate(gate, foundation, spec["runtime_profile"])
    plan = (_adapter.scientific_plan if _adapter else scientific_plan)(foundation, foundation_root=Path(spec["foundation_root"]))
    if (gate["content_hash"] != spec["parents"]["gate"] or spec["source"] != gate["source"]
            or spec["source_commit"] != gate["source_commit"] or spec["project_dir"] != gate["project_dir"]
            or Path(spec["foundation_root"]) != Path(spec["gate_root"]) / "foundation"
            or spec["scientific_plan"] != plan or spec["tasks"] != task_graph(plan)
            or spec["model"] != model_contract() or spec["fresh_fit_count"] != 9 or spec["reducer_count"] != 5
            or spec["selected_branches"] != ["DIRECT", "COARSE"] or spec["dataset_kind"] != (_adapter.KIND if _adapter else KIND)
            or "population_selection" in spec or spec["full_views_persisted"] is not False
            or spec["existing_campaign_mutations"] is not False):
        raise ValueError("Literature direct/coarse campaign differs")
    return digest
