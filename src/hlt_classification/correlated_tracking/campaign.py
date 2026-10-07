"""Original-OFFLINE training pilot, exact source and dry-first single submission."""
import os
from pathlib import Path
import re
import subprocess

from hlt_classification.literature_proxy.campaign import git, exclusive
from hlt_classification.literature_proxy_v2.inputs import authenticate
from .contracts import artifact, load_json, reference, sha256_file, validate, write_immutable_json
from .kernel import recipe

PROJECT = Path(__file__).resolve().parents[3]
RESOURCES = dict(cpus=16, memory_gib=64, hours=2, partition="tigris", account="reu-aisocial", gpus=0)
AUTHORIZATION = "AUTHORIZE JC2 CORRELATED TRACKING PILOT EXACT PLAN"
FLAGS = dict(native_hlt_particles_accessed=False, validation_accessed=False,
             final_test_accessed=False, production_qualified=False)


def source(project, commit, *, pushed=False):
    project = Path(project).resolve()
    if not re.fullmatch("[0-9a-f]{40}", commit) or git(project, "rev-parse", "HEAD") != commit:
        raise ValueError("Source commit differs")
    if git(project, "diff", "HEAD", "--name-only") or git(project, "status", "--porcelain", "--untracked-files=all", "--", "src", "scripts", "sbatch"):
        raise ValueError("Source must be a clean worktree")
    # Pin all Python dependencies, including transitive authentication/reader imports.
    required = sorted(p.relative_to(project).as_posix() for p in (project / "src").rglob("*.py"))
    required += ["scripts/jetclass2_correlated_tracking.py", "scripts/queue_jetclass2_correlated_tracking.sh",
                 "sbatch/run_jetclass2_correlated_tracking.sh",
                 "docs/plans/JETCLASS2_CORRELATED_TRACKING_PILOT_PLAN.md",
                 "docs/contracts/JETCLASS2_CORRELATED_TRACKING.md"]
    # Keep commands below Windows argv limits as well as Linux limits.
    for start in range(0, len(required), 100):
        git(project, "ls-files", "--error-unmatch", "--", *required[start:start+100])
    if pushed and not git(project, "branch", "-r", "--contains", commit, "origin/main"):
        raise PermissionError("Exact source not on origin/main; fetch first")
    return dict(commit=commit, file_sha256={p: sha256_file(project / p) for p in required})


def create(*, project, commit, parent_spec, root):
    project, root, parent_path = (Path(p).resolve() for p in (project, root, parent_spec))
    if project != PROJECT.resolve():
        raise ValueError("Run from the requested pinned project")
    src = source(project, commit, pushed=True)
    refs = dict(parent_spec=reference(parent_path), parent_receipt=reference(parent_path.parent / "receipt.json"))
    parent, records = authenticate(refs["parent_spec"], refs["parent_receipt"])
    if root.exists() or any(root.is_relative_to(p) or p.is_relative_to(root)
            for p in (project, parent_path.parent, Path(parent["data_root"]).resolve())):
        raise ValueError("Fresh output root outside source/project required")
    if not 1 <= parent["population"]["jets"] <= 20_000:
        raise ValueError("Diagnostic parent must contain at most 20,000 training jets")
    spec = artifact("SPEC", project_dir=str(project), root=str(root), source=src, **refs,
                    population=parent["population"], inputs=records, recipe=recipe(),
                    resources=dict(RESOURCES), **FLAGS)
    write_immutable_json(root / "study_spec.json", spec)
    write_immutable_json(root / "command_plan.json", plan(spec))
    return spec


def validate_spec(spec):
    validate(spec, "SPEC")
    if spec["recipe"] != recipe() or spec["resources"] != RESOURCES:
        raise ValueError("Frozen recipe/resources differ")
    if any(spec.get(k) is not False for k in FLAGS):
        raise PermissionError("Pilot scope differs")
    if Path(spec["project_dir"]).resolve() != PROJECT.resolve():
        raise ValueError("Loaded project differs")
    if source(spec["project_dir"], spec["source"]["commit"]) != spec["source"]:
        raise ValueError("Source implementation bytes differ")
    parent, records = authenticate(spec["parent_spec"], spec["parent_receipt"])
    if spec["inputs"] != records or spec["population"] != parent["population"] or not 1 <= spec["population"]["jets"] <= 20_000:
        raise ValueError("Frozen training population differs")


def plan(spec):
    validate(spec, "SPEC")
    root, project = Path(spec["root"]), Path(spec["project_dir"])
    argv = ["sbatch", "--parsable", "--no-requeue", "--nodes=1", "--ntasks=1", "--cpus-per-task=16",
            "--mem=64G", "--time=02:00:00", "--partition=tigris", "--account=reu-aisocial", "--export=NONE",
            "--job-name=jc2corr_pilot", f"--comment=jc2corr:{spec['content_hash']}:pilot",
            f"--chdir={project}", f"--output={root}/slurm-%j.out",
            str(project / "sbatch/run_jetclass2_correlated_tracking.sh"), str(project), str(root / "study_spec.json")]
    return artifact("PLAN", parents=dict(spec=spec["content_hash"]), argv=argv, resources=dict(RESOURCES),
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
    if path.exists():
        ledger = load_json(path)
        validate(ledger, "LEDGER")
        if ledger["parents"] != dict(spec=spec["content_hash"], plan=expected["content_hash"]) or not re.fullmatch(r"\d+", ledger["job"]):
            raise ValueError("Existing submission ledger differs")
        return ledger
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SBATCH_", "SLURM_"))}
    with exclusive(root / "submission.lock"):
        argv = expected["argv"]
        subprocess.run([argv[0], "--test-only", *argv[1:]], check=True, capture_output=True, text=True, env=env)
        write_immutable_json(root / "submission_intent.json", expected)
        result = subprocess.run(argv, check=True, capture_output=True, text=True, env=env)
        if not re.fullmatch(r"\d+(;[\w.-]+)?", result.stdout.strip()):
            raise RuntimeError("Ambiguous sbatch output; inspect scheduler before retry")
        ledger = artifact("LEDGER", parents=dict(spec=spec["content_hash"], plan=expected["content_hash"]),
                          job=result.stdout.strip().split(";")[0], final_test_accessed=False)
        write_immutable_json(path, ledger)
    return ledger
