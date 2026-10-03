"""Explicit NOISE_V3 consumer and SPORC-debug direct/coarse campaign variant.

The shared ladder namespace is execution tooling, never a generator label.
Old CMS artifacts remain v1; this adapter consumes only literature STUDY/v1.
"""
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.literature_proxy_production import campaign as producer
from hlt_classification.literature_proxy_production import population as population
from hlt_classification.literature_proxy_production.contracts import (
    validate as validate_producer,
)
from hlt_classification.literature_proxy_v3.contracts import read_reference
from hlt_classification.jetclass2_delphes.inventory import validate_inventory
from hlt_classification.jetclass2_delphes.execution import execution_site, validate_resources
from hlt_classification.jetclass2_delphes.model import model_contract

from .contracts import artifact, file_ref, validate, validate_file_ref, write_json

COUNTS = {"train": 200_000, "validation": 50_000}
DOMAIN = "JETCLASS2_LITERATURE_NOISE_V3_200K_50K/v1"
KIND = "controlled_literature_synthetic_proxy"
GATE_AUTHORIZATION = "AUTHORIZE JETCLASS2 LITERATURE V3 200K SPORC DEBUG GATE"
SCIENCE_AUTHORIZATION = "AUTHORIZE JETCLASS2 LITERATURE V3 200K DIRECT COARSE SCIENCE"
SOURCE_FILES = (
    "docs/plans/JETCLASS2_LITERATURE_V3_200K_DIRECT_COARSE_PLAN.md",
    "docs/contracts/JETCLASS2_LITERATURE_V3_LADDER.md",
    "scripts/jetclass2_literature_proxy_ladder.py",
    "scripts/queue_jetclass2_literature_proxy_ladder.sh",
    "tests/test_literature_proxy_ladder.py",
    "src/hlt_classification/scouting/hcwdl_exact_dag_submission.py",
)


def authenticate_dataset(root):
    """Cross-worktree, cross-architecture reader; never reexecute the generator.

    Validate immutable producer metadata and ordinary releases, without invoking
    its execution-only PROJECT/architecture guard or opening final-test receipts.
    Physical ordinary blocks are verified by the release builder and paired reader.
    """
    root = Path(root).resolve(strict=True)
    study = load_json(root / "study_spec.json")
    validate_producer(study, "STUDY", parents={
        "pilot": study["pilot_hash"], "population": study["population"]["content_hash"],
    }, test=False)
    population.validate_population(study["population"], "POPULATION", test=False)
    population.validate_shards(study["population"], study["shards"])
    frozen = producer.bundle(study)
    if (Path(study["root"]).resolve() != root or study["kind"] != KIND
            or study["counts"] != population.COUNTS
            or study["population"]["counts"] != population.COUNTS
            or study["physics_production_qualified"] is not False
            or study["source"] != frozen["source"]
            or study["resources"] != producer.RESOURCES):
        raise ValueError("Not the registered frozen literature V3 dataset")
    # The pilot inventory reference is path+sha256 (no byte-count field),
    # intentionally retained unchanged by production.
    inventory = read_reference(study["inventory"])
    validate_inventory(inventory)
    if inventory["content_hash"] != study["population"]["parents"]["inventory"]:
        raise ValueError("Literature dataset inventory lineage differs")
    for role in COUNTS:
        release = load_json(root / "releases" / f"{role}.json")
        validate_producer(release, "ROLE_MANIFEST", parents={
            "study": study["content_hash"], "population": study["population"]["content_hash"],
        }, test=False)
        expected = [f"shards/{s['shard_id']}.json" for s in study["shards"] if s["role"] == role]
        if (release["role"] != role or release["counts"] != {role: study["counts"][role]}
                or [r["relative"] for r in release["shards"]] != expected
                or release["recipe"] != "NOISE_V3" or release["dataset_kind"] != KIND
                or release["native_hlt_accessed"] is not False):
            raise ValueError("Literature ordinary release differs")
        for reference in release["shards"]:
            receipt_path = validate_file_ref(dict(path=reference["relative"],
                bytes=(root / reference["relative"]).stat().st_size, sha256=reference["sha256"]), root=root)
            receipt = load_json(receipt_path)
            validate_producer(receipt, "SHARD", test=False)
            if (receipt["content_hash"] != reference["content_hash"]
                    or receipt["role"] != role or receipt["parents"]["study"] != study["content_hash"]):
                raise ValueError("Literature ordinary receipt differs from released snapshot")
    return study, inventory


def release_request(*, study_root, offline_root):
    root = Path(study_root).resolve(strict=True)
    study, _ = authenticate_dataset(root)
    offline = Path(offline_root).resolve(strict=True)
    if offline != Path(study["data_root"]).resolve():
        raise ValueError("Offline source must be the frozen literature source")
    return artifact("RELEASE_REQUEST", version=2,
        study_root=str(root), offline_root=str(offline), counts=dict(COUNTS),
        study_ref=file_ref(root / "study_spec.json"),
        dataset_kind=KIND, recipe="NOISE_V3", selection_domain=DOMAIN,
        labels_read=False, selection_depends_on_labels=False, allowed_roles=list(COUNTS))


def validate_request(value):
    digest = validate(value, "RELEASE_REQUEST", version=2)
    path = validate_file_ref(value["study_ref"])
    root = Path(value["study_root"]).resolve()
    if path.resolve() != root / "study_spec.json":
        raise ValueError("Literature request source path differs")
    study = load_json(path)
    validate_producer(study, "STUDY", test=False)
    if (study["kind"] != KIND or Path(study["root"]).resolve() != root
            or Path(study["data_root"]).resolve() != Path(value["offline_root"]).resolve()):
        raise ValueError("Literature request dataset endpoint differs")
    # Structural validation is frequent during cache preparation. Do not
    # reauthenticate every shard recursively here: build_release/iter_paired
    # separately authenticate the producer and released physical bytes.
    expected = artifact("RELEASE_REQUEST", version=2,
        study_root=str(root), offline_root=str(Path(value["offline_root"]).resolve()),
        counts=dict(COUNTS), study_ref=value["study_ref"], dataset_kind=KIND,
        recipe="NOISE_V3", selection_domain=DOMAIN, labels_read=False,
        selection_depends_on_labels=False, allowed_roles=list(COUNTS))
    if value != expected:
        raise ValueError("Literature request recipe/population/source differs")
    return digest


def validate_release_source(value):
    request = value["request"]
    study = load_json(validate_file_ref(request["study_ref"]))
    if (value["study_contract"] != "JC2_LITERATURE_PRODUCTION_STUDY/v1"
            or value["parents"]["study"] != study["content_hash"]
            or value["study_root"] != request["study_root"]
            or value["offline_root"] != request["offline_root"]):
        raise ValueError("Literature release has wrong generator/source")
    expected = {s["shard_id"]: s["role"] for s in study["shards"] if s["role"] in COUNTS}
    if {r["shard_id"]: r["role"] for r in value["receipts"]} != expected:
        raise ValueError("Literature release must freeze all ordinary shard receipts")


def gate_tasks():
    return [
        dict(task_id="authenticate_release", kind="cpu", dependencies=[],
             cpus=4, memory_mb=32000, minutes=120),
        dict(task_id="build_foundation", kind="cpu", dependencies=["authenticate_release"],
             cpus=16, memory_mb=160000, minutes=480),
        dict(task_id="preflight", kind="gpu", dependencies=["build_foundation"],
             cpus=16, memory_mb=160000, minutes=480),
    ]


def create_gate(*, study_root, offline_root, gate_root, project_dir, source_commit):
    from .gate import source_lock
    root = Path(gate_root).resolve()
    project = Path(project_dir).resolve(strict=True)
    if root.exists():
        raise FileExistsError("Fresh literature gate root required")
    source = source_lock(project, source_commit, literature=True)
    request = release_request(study_root=study_root, offline_root=offline_root)
    for protected in (Path(study_root).resolve(), Path(offline_root).resolve(), project):
        if root.is_relative_to(protected) or protected.is_relative_to(root):
            raise PermissionError("Gate output overlaps protected input/worktree")
    spec = artifact("GATE_SPEC", version=7,
        parents={"source": source["content_hash"], "request": request["content_hash"]},
        source=source, request=request, gate_root=str(root), project_dir=str(project),
        source_commit=source_commit, capacity=512, execution_site=execution_site("sporc_a100_debug"),
        tasks=gate_tasks(), workers=16, foundation_workers=16,
        scientific_branches=["DIRECT", "COARSE"], dataset_kind=KIND,
        full_views_persisted=False, site_transfer_policy=None,
        admission="fresh_literature_release_matching_then_full_population_sporc_debug_preflight")
    validate_gate(spec, check_source=True)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "gate_spec.json", spec)
    return spec


def validate_gate(spec, *, check_source=False):
    from .gate import source_lock
    digest = validate(spec, "GATE_SPEC", version=7, parents={
        "source": spec["source"]["content_hash"], "request": spec["request"]["content_hash"],
    })
    validate(spec["source"], "SOURCE")
    validate_request(spec["request"])
    if (spec["source_commit"] != spec["source"]["commit"]
            or spec["request"]["schema_version"] != 2 or spec["dataset_kind"] != KIND
            or spec["execution_site"] != execution_site("sporc_a100_debug")
            or spec["tasks"] != gate_tasks() or spec["workers"] != 16
            or spec["foundation_workers"] != 16 or spec["capacity"] != 512
            or spec["scientific_branches"] != ["DIRECT", "COARSE"]
            or spec["site_transfer_policy"] is not None or "measurement_site" in spec
            or spec["full_views_persisted"] is not False):
        raise ValueError("Literature SPORC debug gate semantics differ")
    if check_source and source_lock(Path(spec["project_dir"]), spec["source_commit"], literature=True) != spec["source"]:
        raise ValueError("Literature gate source differs")
    return digest


def scientific_plan(foundation, *, foundation_root=None):
    from .campaign import _build_scientific_plan, DIRECT_COARSE_BRANCHES
    if (foundation.get("schema_version") != 2 or foundation["release"].get("schema_version") != 2
            or foundation["role_counts"] != COUNTS):
        raise ValueError("Literature plan requires the fresh 200k/50k literature foundation")
    return _build_scientific_plan(foundation, registered_branches=DIRECT_COARSE_BRANCHES,
        version=3, foundation_root=foundation_root, node_prefix="LITV3")


def validate_profile(profile, *, foundation, spec):
    from .cache import cache_budgets
    from hlt_classification.jetclass2_delphes.campaign import recipe
    from hlt_classification.jetclass2_delphes.contracts import validate as validate_jc2
    digest = validate(profile, "RUNTIME_PROFILE", version=7, parents={
        "gate": spec["content_hash"], "foundation": foundation["content_hash"],
    })
    parity = profile["installed_weaver_parity"]
    validate_jc2(parity, "WEAVER_PARITY")
    validate_jc2(profile["installed_environment"], "INSTALLED_ENVIRONMENT", version=2)
    report = profile["acceptance_training_report"]
    validate_jc2(report, "KERNEL_TRAINING_REPORT")
    expected_node = next(n for n in scientific_plan(foundation)["nodes"] if n["node_id"] == "U000")
    import math
    one_pass = report["runtime_seconds"]
    caches = profile["cache_seconds_by_coordinate"]
    if (set(caches) != {"U000", "D050"} or any(not math.isfinite(v) or v < 0 for v in caches.values())
            or not math.isfinite(one_pass) or one_pass <= 0
            or not math.isfinite(profile["inference_seconds"]) or profile["inference_seconds"] <= 0):
        raise ValueError("Literature timing measurements differ")
    train_minutes = max(60, math.ceil((max(caches.values()) + 100 * one_pass) * 1.75 / 60))
    reduce_minutes = max(30, math.ceil((max(caches.values()) + profile["inference_seconds"]) * 2 / 60))
    if (spec.get("schema_version") != 7 or foundation["release"]["request"] != spec["request"]
            or foundation["role_counts"] != COUNTS or profile["measured_role_counts"] != COUNTS
            or profile["execution_site"] != execution_site("sporc_a100_debug")
            or profile.get("site_transfer_policy") is not None
            or profile.get("measurement_site") is not None
            or profile["cpus"] != 16 or profile["workers"] != 16 or profile["memory_mb"] != 160000
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
            or not 60 <= profile["train_minutes"] <= 1440
            or not 30 <= profile["reduce_minutes"] <= 1440
            or "A100" not in profile["gpu"]["name"]
            or not 0 < profile["gpu_peak_bytes"] <= .85 * profile["gpu"]["total_memory_bytes"]
            or profile["cache_budgets"] != cache_budgets(foundation, 160000, 16)):
        raise ValueError("Literature full-population measured debug profile differs")
    validate_resources(profile["execution_site"], 16, 160000, 16)
    return digest


def check_submission_site(plan):
    """Read-only scheduler feasibility checks, with inherited requests removed."""
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))}
    cluster = subprocess.run(["scontrol", "show", "config"], capture_output=True,
                             text=True, check=True, env=env).stdout
    import re
    if not re.search(r"(?m)^\s*ClusterName\s*=\s*sporc\s*$", cluster):
        raise PermissionError("Submit this campaign from the SPORC cluster")
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
            raise PermissionError("SPORC rejected resource shape: " + result.stdout + result.stderr)


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


def create_campaign(*, gate_root, campaign_root):
    from .data import validate_foundation
    from .campaign import task_graph
    gate_root = Path(gate_root).resolve(strict=True)
    gate = load_json(gate_root / "gate_spec.json")
    validate_gate(gate, check_source=True)
    foundation_root = gate_root / "foundation"
    foundation = load_json(foundation_root / "foundation.json")
    validate_foundation(foundation, root=foundation_root)
    profile = load_json(gate_root / "evidence/runtime_profile.json")
    validate_profile(profile, foundation=foundation, spec=gate)
    _completed_gate(gate, foundation, profile)
    root = Path(campaign_root).resolve()
    protected = [gate_root, Path(gate["project_dir"]), Path(gate["request"]["study_root"]),
                 Path(gate["request"]["offline_root"])]
    if root.exists() or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise FileExistsError("Fresh separate literature campaign root required")
    plan = scientific_plan(foundation, foundation_root=foundation_root)
    spec = artifact("CAMPAIGN_SPEC", version=3,
        parents={"gate": gate["content_hash"], "foundation": foundation["content_hash"],
                 "profile": profile["content_hash"], "plan": plan["content_hash"]},
        gate_root=str(gate_root), campaign_root=str(root), project_dir=gate["project_dir"],
        source_commit=gate["source_commit"], source=gate["source"],
        foundation_root=str(foundation_root), foundation=foundation,
        runtime_profile=profile, scientific_plan=plan, tasks=task_graph(plan), model=model_contract(),
        fresh_fit_count=9, reducer_count=5, selected_branches=["DIRECT", "COARSE"],
        dataset_kind=KIND, full_views_persisted=False, existing_campaign_mutations=False)
    validate_campaign(spec, check_source=True)
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


def validate_campaign(spec, *, check_source=False):
    from .data import validate_foundation
    from .campaign import task_graph
    digest = validate(spec, "CAMPAIGN_SPEC", version=3, parents={
        "gate": spec["parents"]["gate"], "foundation": spec["foundation"]["content_hash"],
        "profile": spec["runtime_profile"]["content_hash"], "plan": spec["scientific_plan"]["content_hash"],
    })
    gate = load_json(Path(spec["gate_root"]) / "gate_spec.json")
    validate_gate(gate, check_source=check_source)
    foundation = spec["foundation"]
    validate_foundation(foundation, root=Path(spec["foundation_root"]))
    validate_profile(spec["runtime_profile"], foundation=foundation, spec=gate)
    _completed_gate(gate, foundation, spec["runtime_profile"])
    plan = scientific_plan(foundation, foundation_root=Path(spec["foundation_root"]))
    if (gate["content_hash"] != spec["parents"]["gate"] or spec["source"] != gate["source"]
            or spec["source_commit"] != gate["source_commit"] or spec["project_dir"] != gate["project_dir"]
            or Path(spec["foundation_root"]) != Path(spec["gate_root"]) / "foundation"
            or spec["scientific_plan"] != plan or spec["tasks"] != task_graph(plan)
            or spec["model"] != model_contract() or spec["fresh_fit_count"] != 9 or spec["reducer_count"] != 5
            or spec["selected_branches"] != ["DIRECT", "COARSE"] or spec["dataset_kind"] != KIND
            or "population_selection" in spec or spec["full_views_persisted"] is not False
            or spec["existing_campaign_mutations"] is not False):
        raise ValueError("Literature direct/coarse campaign differs")
    return digest
