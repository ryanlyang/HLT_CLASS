"""One-job train-only pilot, source-pinned and explicitly authorized."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import re
import subprocess

from .contracts import artifact, load_json, read_reference, reference, validate, write_immutable_json
from .kernel import recipe
from .population import select

AUTHORIZATION = "AUTHORIZE JC2 LITERATURE PROXY PILOT EXACT PLAN"
RESOURCES = dict(cpus=16, memory_gib=64, hours=2, partition="tigris", account="reu-aisocial", gpus=0)
PROJECT = Path(__file__).resolve().parents[3]


def git(project, *args):
    return subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True, text=True).stdout.strip()


def source(project, commit, *, pushed=False):
    project = Path(project).resolve()
    if not re.fullmatch("[0-9a-f]{40}", commit) or git(project, "rev-parse", "HEAD") != commit:
        raise ValueError("Source commit differs")
    # All tracked changes matter; untracked scientific/execution code cannot shadow pinned code.
    if git(project, "diff", "HEAD", "--name-only") or git(project, "status", "--porcelain", "--untracked-files=all", "--", "src", "scripts", "sbatch"):
        raise ValueError("Source must be a clean worktree")
    required = ["scripts/jetclass2_literature_proxy.py", "sbatch/run_jetclass2_literature_proxy_pilot.sh",
                "docs/plans/JETCLASS2_LITERATURE_PROXY_PILOT_PLAN.md", "docs/contracts/JETCLASS2_LITERATURE_PROXY.md"]
    required += sorted(p.relative_to(project).as_posix() for p in (project / "src/hlt_classification/literature_proxy").glob("*.py"))
    required += ["src/hlt_classification/cms2jc2_response/bridge.py",
                 "src/hlt_classification/data/cache_contracts.py",
                 "src/hlt_classification/jetclass2_delphes/split_registry.py",
                 "src/hlt_classification/jetclass2_delphes/inventory.py"]
    git(project, "ls-files", "--error-unmatch", "--", *required)
    if pushed and not git(project, "branch", "-r", "--contains", commit, "origin/main"):
        raise PermissionError("Exact source is not recorded on origin/main; fetch first")
    from .contracts import sha256_file
    return dict(commit=commit, file_sha256={p: sha256_file(project / p) for p in required})


def create(*, project, commit, data_root, inventory, profile, root, count=20_000):
    project, root, data_root = (Path(p).resolve() for p in (project, root, data_root))
    if project != PROJECT.resolve():
        raise ValueError("Run the CLI from the requested pinned project")
    src = source(project, commit, pushed=True)
    if root.exists() or root.is_relative_to(data_root) or data_root.is_relative_to(root):
        raise ValueError("Fresh output root outside the source dataset required")
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    refs = dict(inventory=reference(inventory), profile=reference(profile))
    inv, prof = (read_reference(refs[k]) for k in ("inventory", "profile"))
    population = select(inv, prof, count)
    spec = artifact("SPEC", project_dir=str(project), root=str(root), data_root=str(data_root),
                    source=src, **refs, population=population, recipe=recipe(), resources=dict(RESOURCES),
                    native_hlt_particles_accessed=False, validation_accessed=False, final_test_accessed=False,
                    production_qualified=False)
    write_immutable_json(root / "study_spec.json", spec)
    write_immutable_json(root / "command_plan.json", plan(spec))
    return spec


def validate_spec(spec, *, authenticate_source=True):
    validate(spec, "SPEC")
    if spec["recipe"] != recipe() or spec["resources"] != RESOURCES:
        raise ValueError("Frozen recipe/resources differ")
    for field in ("native_hlt_particles_accessed", "validation_accessed", "final_test_accessed", "production_qualified"):
        if spec.get(field) is not False:
            raise PermissionError("Pilot access/scope differs")
    if Path(spec["project_dir"]).resolve() != PROJECT.resolve():
        raise ValueError("Loaded project differs")
    if authenticate_source and source(spec["project_dir"], spec["source"]["commit"]) != spec["source"]:
        raise ValueError("Source implementation bytes differ")
    inv, prof = (read_reference(spec[k]) for k in ("inventory", "profile"))
    if spec["population"] != select(inv, prof, spec["population"]["jets"]):
        raise ValueError("Frozen training population differs")
    return inv, prof


def plan(spec):
    validate(spec, "SPEC")
    root, project = Path(spec["root"]), Path(spec["project_dir"])
    argv = ["sbatch", "--parsable", "--no-requeue", "--nodes=1", "--ntasks=1", "--cpus-per-task=16",
            "--mem=64G", "--time=02:00:00", "--partition=tigris", "--account=reu-aisocial", "--export=NONE",
            "--job-name=jc2lit_pilot", f"--comment=jc2lit:{spec['content_hash']}:pilot",
            f"--chdir={project}", f"--output={root}/slurm-%j.out",
            str(project / "sbatch/run_jetclass2_literature_proxy_pilot.sh"), str(project), str(root / "study_spec.json")]
    return artifact("PLAN", parents=dict(spec=spec["content_hash"]), argv=argv, resources=dict(RESOURCES),
                    live_authorization_phrase=AUTHORIZATION, jobs=1, automatic_followup=False)


@contextmanager
def exclusive(path):
    """Keep a failed lock as evidence; never guess whether a Slurm side effect happened."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()) + "\n")
    yield
    path.unlink()


def submit(spec, *, execute=False, reviewed_hash=None, authorization=None):
    validate_spec(spec)
    root = Path(spec["root"])
    expected = plan(spec)
    if load_json(root / "command_plan.json") != expected:
        raise ValueError("Saved command plan differs")
    if not execute:
        return dict(dry_run=True, plan=expected, message="No jobs submitted")
    if reviewed_hash != expected["content_hash"] or authorization != AUTHORIZATION:
        raise PermissionError("Exact reviewed plan hash and authorization required")
    source(spec["project_dir"], spec["source"]["commit"], pushed=True)
    ledger_path = root / "submission_ledger.json"
    if ledger_path.exists():
        ledger = load_json(ledger_path)
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
        job = result.stdout.strip().split(";")[0]
        if not re.fullmatch(r"\d+", job):
            raise RuntimeError("Ambiguous sbatch output; inspect scheduler before any retry")
        ledger = artifact("LEDGER", parents=dict(spec=spec["content_hash"], plan=expected["content_hash"]),
                          job=job, final_test_accessed=False)
        write_immutable_json(ledger_path, ledger)
    return ledger
