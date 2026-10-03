"""Exactly one authorized CPU-only count38 pilot, preserving its v1 parent."""
import os
from pathlib import Path
import re
import subprocess

from hlt_classification.literature_proxy import campaign as old
from .contracts import artifact, load_json, reference, sha256_file, validate, write_immutable_json
from .inputs import authenticate
from .kernel import recipe

PROJECT = Path(__file__).resolve().parents[3]
RESOURCES = dict(old.RESOURCES)
AUTHORIZATION = "AUTHORIZE JC2 LITERATURE PROXY COUNT38 EXACT PLAN"
exclusive = old.exclusive


def source(project, commit, *, pushed=False):
    result = old.source(project, commit, pushed=pushed)
    project = Path(project)
    required = ["scripts/jetclass2_literature_proxy_count38.py",
                "scripts/queue_jetclass2_literature_proxy_count38.sh",
                "sbatch/run_jetclass2_literature_proxy_count38.sh",
                "docs/plans/JETCLASS2_LITERATURE_PROXY_COUNT38_PILOT_PLAN.md",
                "docs/contracts/JETCLASS2_LITERATURE_PROXY_COUNT38.md"]
    required += sorted(p.relative_to(project).as_posix() for p in (project / "src/hlt_classification/literature_proxy_v2").glob("*.py"))
    old.git(project, "ls-files", "--error-unmatch", "--", *required)
    result["file_sha256"].update({p: sha256_file(project / p) for p in required})
    return result


def create(*, project, commit, parent_spec, root):
    project, root = Path(project).resolve(), Path(root).resolve()
    if project != PROJECT.resolve():
        raise ValueError("Run from the requested pinned project")
    src = source(project, commit, pushed=True)
    parent_ref = reference(parent_spec)
    receipt_ref = reference(Path(parent_spec).parent / "receipt.json")
    parent, inputs = authenticate(parent_ref, receipt_ref)
    forbidden = [Path(parent["root"]).resolve(), Path(parent["data_root"]).resolve(), project]
    if root.exists() or any(root.is_relative_to(p) or p.is_relative_to(root) for p in forbidden):
        raise ValueError("Fresh output root outside parent data and execution worktree required")
    spec = artifact("SPEC", project_dir=str(project), root=str(root), source=src,
                    parent_spec=parent_ref, parent_receipt=receipt_ref, inputs=inputs,
                    population=parent["population"], recipe=recipe(), resources=RESOURCES,
                    native_hlt_particles_accessed=False, validation_accessed=False,
                    final_test_accessed=False, production_qualified=False)
    write_immutable_json(root / "study_spec.json", spec)
    write_immutable_json(root / "command_plan.json", plan(spec))
    return spec


def validate_spec(spec, *, authenticate_source=True):
    validate(spec, "SPEC")
    if spec["recipe"] != recipe() or spec["resources"] != RESOURCES:
        raise ValueError("Frozen recipe/resources differ")
    if Path(spec["project_dir"]).resolve() != PROJECT.resolve():
        raise ValueError("Loaded project differs")
    for key in ("native_hlt_particles_accessed", "validation_accessed", "final_test_accessed", "production_qualified"):
        if spec.get(key) is not False:
            raise PermissionError("Pilot scope differs")
    if authenticate_source and source(spec["project_dir"], spec["source"]["commit"]) != spec["source"]:
        raise ValueError("Source bytes differ")
    parent, inputs = authenticate(spec["parent_spec"], spec["parent_receipt"])
    if spec["inputs"] != inputs or spec["population"] != parent["population"]:
        raise ValueError("Frozen parent inputs/population differ")


def plan(spec):
    validate(spec, "SPEC")
    root, project = Path(spec["root"]), Path(spec["project_dir"])
    argv = ["sbatch", "--parsable", "--no-requeue", "--nodes=1", "--ntasks=1", "--cpus-per-task=16",
            "--mem=64G", "--time=02:00:00", "--partition=tigris", "--account=reu-aisocial", "--export=NONE",
            "--job-name=jc2lit2_pilot", f"--comment=jc2lit2:{spec['content_hash']}:pilot",
            f"--chdir={project}", f"--output={root}/slurm-%j.out",
            str(project / "sbatch/run_jetclass2_literature_proxy_count38.sh"), str(project), str(root / "study_spec.json")]
    return artifact("PLAN", parents=dict(spec=spec["content_hash"]), argv=argv, resources=RESOURCES,
                    jobs=1, automatic_followup=False, live_authorization_phrase=AUTHORIZATION)


def submit(spec, *, execute=False, reviewed_hash=None, authorization=None):
    validate_spec(spec)
    root, expected = Path(spec["root"]), plan(spec)
    if load_json(root / "command_plan.json") != expected:
        raise ValueError("Saved command plan differs")
    if not execute:
        return dict(dry_run=True, plan=expected, message="No jobs submitted")
    if reviewed_hash != expected["content_hash"] or authorization != AUTHORIZATION:
        raise PermissionError("Exact reviewed plan hash and authorization required")
    source(spec["project_dir"], spec["source"]["commit"], pushed=True)
    path = root / "submission_ledger.json"
    parents = dict(spec=spec["content_hash"], plan=expected["content_hash"])
    if path.exists():
        ledger = load_json(path)
        validate(ledger, "LEDGER")
        if ledger["parents"] != parents or not re.fullmatch(r"\d+", ledger["job"]):
            raise ValueError("Existing ledger differs")
        return ledger
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SBATCH_", "SLURM_"))}
    with exclusive(root / "submission.lock"):
        argv = expected["argv"]
        subprocess.run([argv[0], "--test-only", *argv[1:]], check=True, capture_output=True, text=True, env=env)
        write_immutable_json(root / "submission_intent.json", expected)
        result = subprocess.run(argv, check=True, capture_output=True, text=True, env=env)
        job = result.stdout.strip().split(";")[0]
        if not re.fullmatch(r"\d+", job):
            raise RuntimeError("Ambiguous sbatch output; inspect scheduler before retrying")
        ledger = artifact("LEDGER", parents=parents, job=job, final_test_accessed=False)
        write_immutable_json(path, ledger)
    return ledger
