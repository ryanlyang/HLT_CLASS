"""Portable formula provenance, immutable OSCAR graph and staged exact-ID launch."""
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.jetclass2_delphes.concat_k2_campaign import validate as old_validate
from hlt_classification.jetclass2_delphes.concat_k2_views import view_contract as old_views
from hlt_classification.jetclass2_delphes.production import _source
from hlt_classification.literature_proxy_consumer import RelocatedDataset, MANIFEST_SHA256
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, load_exact_dag_journal
from .contracts import artifact, validate, registration, require, publish, load, checked, AUTHORIZE
from .views import view_contract


def import_formula(path, digest):
    require(re.fullmatch(r"[0-9a-f]{64}", digest) and sha256_file(path) == digest,
            "Original K2 JSON byte hash differs; supply the recorded donor hash")
    value = load_json(path)
    require(value.get("contract") == "JETCLASS2_DELPHES_CONCAT_K2_CAMPAIGN_SPEC/v7",
            "Expected the original full K2 campaign, not a pilot or arbitrary formula")
    old_validate(value, "CAMPAIGN_SPEC")
    foundation = value["foundation"]
    old_validate(foundation, "FOUNDATION_SPEC")
    source = value["source_import"]
    old_validate(source, "SOURCE_IMPORT")
    candidate = foundation["candidate"]
    require(foundation["views"] == old_views(candidate)
        and foundation["source_import_sha256"] == source["content_hash"]
        and source["selected_candidate"] == candidate
        and source["salience_formula_only"] is True and source["assignments_imported"] is False
        and source["models_imported"] == [], "Donor formula lineage differs")
    # Intentionally do NOT resolve any RIT paths in this copied immutable JSON.
    return artifact("FORMULA_IMPORT", candidate=candidate, donor_spec_sha256=value["content_hash"],
        donor_file_sha256=digest, donor_commit=value["source_commit"],
        donor_foundation_sha256=foundation["content_hash"], donor_source_sha256=source["content_hash"],
        original_formula=foundation["views"]["salience"], portable_formula_only=True,
        assignments_imported=False, trained_models_imported=False)


def shards(data):
    return [dict(index=i, role=s["role"], source=s) for i, s in enumerate(
        s for s in data.study["shards"] if s["role"] in ("train", "validation"))]


def graph(ordinary_shards, node_rows):
    rows = []
    def add(name, kind, deps, resource="metadata", **extra):
        rows.append(dict(task_id=name, kind=kind, dependencies=deps, resource=resource, **extra))
    add("authenticate", "authenticate", [])
    add("select_population", "population", ["authenticate"])
    add("matcher_acceptance", "matcher_acceptance", ["select_population"], "assignment")
    for s in ordinary_shards:
        add(f"assign_{s['index']:04d}", "assign", ["matcher_acceptance"], "assignment", shard_index=s["index"])
    add("foundation", "foundation", [r["task_id"] for r in rows if r["kind"] == "assign"])
    add("preflight", "preflight", ["foundation"], "preflight")
    add("after_gate", "after_gate", ["preflight"])
    teachers = {n["teacher_distribution"] for n in node_rows} - {None}
    for n in node_rows:
        name, teacher = n["node_id"], n["teacher_distribution"]
        add("train_"+name, "train", ["reduce_"+teacher] if teacher else ["preflight"], "train", node_id=name)
        if name in teachers:
            add("reduce_"+name, "reduce", ["train_"+name], "reduce", node_id=name)
    add("aggregate", "aggregate", [r["task_id"] for r in rows if r["kind"] in ("train", "reduce")])
    add("complete", "complete", ["aggregate"])
    return rows


SCIENCE = {"train", "reduce", "aggregate", "complete"}


def create(*, dataset_root, formula_spec, formula_sha256, project_dir, source_commit, campaign_root):
    project, root, data_root = (Path(p).resolve() for p in (project_dir, campaign_root, dataset_root))
    _source(project, source_commit)
    require(project == Path(__file__).resolve().parents[3], "Run the CLI from its pinned checkout")
    require(not root.exists() and not root.is_relative_to(data_root)
            and not data_root.is_relative_to(root) and not root.is_relative_to(project), "Need a fresh, separate campaign root")
    formula = import_formula(formula_spec, formula_sha256)
    data = RelocatedDataset.from_oscar_copy(data_root, expected_manifest_sha256=MANIFEST_SHA256)
    settings = registration()
    ordinary = shards(data)
    value = artifact("CAMPAIGN_SPEC", **settings, dataset_root=str(data_root),
        manifest_sha256=MANIFEST_SHA256, dataset_metadata=data.describe(),
        formula=formula, candidate=formula["candidate"], views=view_contract(formula["candidate"]),
        shards=ordinary, tasks=graph(ordinary, settings["nodes"]), project_dir=str(project),
        source_commit=source_commit, campaign_root=str(root))
    root.mkdir(parents=True, exist_ok=False)
    # Copy the original JSON unchanged, not a rewritten producer or RIT artifact.
    from hlt_classification.data.cache_contracts import atomic_publish_bytes
    atomic_publish_bytes(root/"inputs"/"original_k2_campaign.json", Path(formula_spec).read_bytes())
    publish(root/"campaign_spec.json", value)
    validate_campaign(value)
    for stage in ("all", "gate", "science"):
        submit(value, stage=stage)
    return value


def validate_campaign(spec, *, check_source=True, check_dataset=True):
    digest = validate(spec, "CAMPAIGN_SPEC")
    require(all(spec.get(k) == v for k,v in registration().items()), "Scientific/execution registration differs")
    root = Path(spec["campaign_root"])
    formula = import_formula(root/"inputs"/"original_k2_campaign.json", spec["formula"]["donor_file_sha256"])
    require(spec["formula"] == formula and spec["candidate"] == formula["candidate"]
        and spec["views"] == view_contract(formula["candidate"])
        and spec["manifest_sha256"] == MANIFEST_SHA256
        and spec["tasks"] == graph(spec["shards"], spec["nodes"]), "Formula/graph identity differs")
    require(all(s["role"] in ("train", "validation") for s in spec["shards"]), "Forbidden role")
    if check_dataset:
        data = RelocatedDataset.from_oscar_copy(spec["dataset_root"], expected_manifest_sha256=MANIFEST_SHA256)
        require(spec["shards"] == shards(data) and spec["dataset_metadata"] == data.describe(), "Frozen dataset differs")
    if check_source:
        require(Path(spec["project_dir"]).resolve() == Path(__file__).resolve().parents[3],
                "Worker/imported code is not the registered pinned checkout")
        _source(Path(spec["project_dir"]), spec["source_commit"])
    return digest


def plan(spec, stage):
    require(stage in ("all", "gate", "science"), "Unknown stage")
    rows = [r for r in spec["tasks"] if stage == "all" or (r["kind"] in SCIENCE) == (stage == "science")]
    names = {r["task_id"] for r in rows}
    commands = []
    for row in rows:
        r, site = spec["resources"][row["resource"]], spec["execution_site"]
        deps = [d for d in row["dependencies"] if d in names]
        cmd = ["sbatch", "--parsable", "--account="+site["account"],
            "--partition="+(site["partition"] if r["gpu"] else spec["cpu_partition"]),
            "--nodes=1", "--ntasks=1", "--export=ALL", "--no-requeue",
            f"--cpus-per-task={r['cpus']}", f"--mem={r['memory_mb']}M", f"--time={r['minutes']}",
            "--job-name=noisek2_"+row["task_id"], "--chdir="+spec["project_dir"],
            "--output="+str(Path(spec["campaign_root"])/"slurm-%j.out")]
        if r["gpu"]:
            cmd += ["--qos="+site["qos"], "--gres="+site["gres"]]
        if deps:
            cmd += ["--dependency=afterok:"+":".join("${JOB_"+d+"}" for d in deps)]
        cmd += [str(Path(spec["project_dir"])/"sbatch"/"run_noise_k2.sh"), spec["project_dir"],
                str(Path(spec["campaign_root"])/"campaign_spec.json"), row["task_id"]]
        commands.append(dict(task_id=row["task_id"], dependencies=deps, command=cmd))
    return dict(commands=commands)


def submit(spec, *, stage, execute=False, authorization_phrase=None, auto_science=False):
    validate_campaign(spec)
    root = Path(spec["campaign_root"])
    proposed = plan(spec, stage)
    dry = root/f"{stage}_dry_run.json"
    if execute:
        require(stage != "all", "Live full-DAG submission forbidden; use gated stages")
        require(authorization_phrase == AUTHORIZE, "Exact authorization phrase required")
        # Full and stage dry runs must already exist and match exactly.
        require((root/"all_dry_run.json").is_file(), "Full canonical dry run is required")
        submit_exact_dag(identity=spec["content_hash"], plan=plan(spec, "all"),
            output=root/"all_dry_run.json", canonical_dry_run=root/"all_dry_run.json", execute=False)
        require(dry.is_file(), "Run this stage dry before live submission")
        if stage == "science":
            from .runtime import science_gate
            science_gate(spec)
        environment = {k:v for k,v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))}
        # Site/account limits must be checked on OSCAR; no guessed permission.
        tested = set()
        for row in proposed["commands"]:
            options = [v for v in row["command"][1:] if v.startswith("--") and not v.startswith("--dependency=")]
            shape = tuple(v for v in options if not v.startswith(("--job-name=", "--output=", "--chdir=")))
            if shape not in tested:
                result = subprocess.run(["sbatch", "--test-only", *options, "--wrap=true"],
                    text=True, capture_output=True, env=environment)
                require(result.returncode == 0, "Slurm resource test failed: "+result.stderr)
                tested.add(shape)
        publish(root/"authorization.json", artifact("AUTHORIZATION", campaign_sha256=spec["content_hash"],
            phrase=AUTHORIZE, auto_science=bool(auto_science) if stage == "gate" else
            load(root/"authorization.json", "AUTHORIZATION")["auto_science"]))
        destination = root/f"{stage}_submission_ledger.json"
        if not destination.exists():
            # Never silently retry an ambiguous sbatch acknowledgement.
            with (root/f"{stage}_submission.claim").open("x", encoding="utf-8") as handle:
                handle.write(spec["content_hash"]+"\n")
        return submit_exact_dag(identity=spec["content_hash"], plan=proposed, output=destination,
            canonical_dry_run=dry, execute=True, environment=environment)
    require(not auto_science, "Auto-science needs explicit live authorization")
    return submit_exact_dag(identity=spec["content_hash"], plan=proposed, output=dry,
        canonical_dry_run=dry, execute=False)


def task(spec, name):
    rows = [r for r in spec["tasks"] if r["task_id"] == name]
    require(len(rows) == 1, "Unknown task")
    return rows[0]


def completed(spec, name, *, recursive=True, seen=None):
    row = task(spec, name)
    root = Path(spec["campaign_root"])
    path = root/"tasks"/(name+".json")
    if not path.is_file():
        return None
    report = load(path, "TASK")
    require(report["campaign_sha256"] == spec["content_hash"] and report["task_id"] == name
        and report["source_commit"] == spec["source_commit"]
        and set(report["parents"]) == set(row["dependencies"]) and report["outputs"], "Task lineage differs")
    for ref in report["outputs"]:
        checked(root, ref)
    if recursive:
        seen = {} if seen is None else seen
        for parent, digest in report["parents"].items():
            if parent not in seen:
                seen[parent] = completed(spec, parent, seen=seen)
            require(seen[parent] and seen[parent]["content_hash"] == digest, "Dependency attestation differs")
    return report


def authenticate_job(spec, row):
    job = os.environ.get("SLURM_JOB_ID", "")
    require(re.fullmatch(r"[1-9][0-9]*", job), "A real Slurm job is required")
    stage = "science" if row["kind"] in SCIENCE else "gate"
    root = Path(spec["campaign_root"])
    for _ in range(30):
        _, jobs = load_exact_dag_journal(root/f"{stage}_submission_ledger_journal",
            identity=spec["content_hash"], plan=plan(spec, stage))
        if row["task_id"] in jobs:
            break
        time.sleep(1)
    require(jobs.get(row["task_id"]) == job, "Worker job is not the exact submitted campaign ID")
    site, res = spec["execution_site"], spec["resources"][row["resource"]]
    details = subprocess.run(["scontrol", "show", "job", "-o", job], check=True, text=True, capture_output=True).stdout
    fields = dict(t.split("=",1) for t in details.split() if "=" in t)
    expected_prefix = site["conda_base"]+"/envs/"+site["conda_env"]
    require(os.environ.get("SLURM_CLUSTER_NAME") == site["cluster"]
        and fields.get("Account") == site["account"] and fields.get("Partition") ==
        (site["partition"] if res["gpu"] else spec["cpu_partition"])
        and fields.get("NumNodes") == "1" and fields.get("NumTasks") == "1"
        and fields.get("NumCPUs") == str(res["cpus"])
        and os.environ.get("SLURM_CPUS_PER_TASK") == str(res["cpus"])
        and os.environ.get("SLURM_MEM_PER_NODE") == str(res["memory_mb"])
        and sys.prefix == expected_prefix and os.environ.get("CONDA_PREFIX") == expected_prefix
        and platform.machine() == site["architecture"] and os.environ.get("PYTHONNOUSERSITE") == "1",
        "Worker allocation/environment differs from registration")
    if res["gpu"]:
        from hlt_classification.jetclass2_delphes.execution import allocation
        allocation(site)
    return job
