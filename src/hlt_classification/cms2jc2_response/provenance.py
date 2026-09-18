"""Read-only source and environment authentication; no implicit Git mutations."""
from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path
import re
import subprocess
import sys

from .contracts import PLAN, artifact, sha256_file, validate

DONORS = (
    "src/hlt_classification/data/cache_contracts.py",
    "src/hlt_classification/scouting/labels.py",
    "src/hlt_classification/scouting/schema.py",
    "src/hlt_classification/scouting/splits.py",
    "src/hlt_classification/jetclass2_delphes/inventory.py",
    "src/hlt_classification/jetclass2_delphes/split_registry.py",
    "src/hlt_classification/jetclass2_delphes/splits.py",
    "src/hlt_classification/jetclass2_delphes/contracts.py",
    "src/hlt_classification/jetclass2_delphes/schema.py",
)


def source_files(root: Path) -> dict[str, str]:
    paths = list((root / "src/hlt_classification/cms2jc2_response").glob("*.py"))
    paths += [root / p for p in (*DONORS, PLAN, "docs/contracts/CMS2JC2_RESPONSE_PREPARATION.md",
                                  "pyproject.toml", "scripts/cms2jc2_response.py",
                                  "sbatch/run_cms2jc2_response_cpu.sh")]
    if not paths or any(not p.is_file() for p in paths):
        raise FileNotFoundError("Incomplete response source surface")
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(paths)}


def source_record(root: Path, expected_commit: str | None = None, *, executable: bool = False) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    head = git("rev-parse", "HEAD")
    if not re.fullmatch("[0-9a-f]{40}", head) or (expected_commit is not None and head != expected_commit):
        raise ValueError("Response source commit differs")
    files = source_files(root)
    dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
    if executable:
        if dirty:
            raise PermissionError("Queue execution requires a clean dedicated worktree")
        for relative in files:
            git("ls-files", "--error-unmatch", relative)
        remotes = git("branch", "-r", "--contains", head).splitlines()
        if not remotes:
            raise PermissionError("Pinned commit has no recorded remote branch; fetch/push first")
    return artifact("SOURCE", commit=head, files=files, dirty=dirty, executable=executable)


def validate_source(record: dict, root: Path, *, executable: bool):
    validate(record, "SOURCE")
    current = source_record(root, record["commit"], executable=executable)
    if current["files"] != record["files"]:
        raise ValueError("Response source bytes changed")
    if executable and not record["executable"]:
        raise PermissionError("Local development evidence cannot authorize execution")


def environment() -> dict:
    versions = {}
    for name in ("numpy", "scipy", "awkward", "awkward-cpp", "uproot", "scikit-learn", "threadpoolctl", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return artifact("ENVIRONMENT", python=platform.python_version(), system=platform.system(),
                    machine=platform.machine(), packages=versions)


def numerical_environment() -> dict:
    """Hash actual installed numerical Python/native-library bytes, not RECORD claims."""
    from .contracts import canonical_sha256
    result = environment()
    packages = {}
    for name, version in result["packages"].items():
        if version is None:
            raise ImportError(f"Missing required response dependency: {name}")
        distribution = importlib.metadata.distribution(name)
        files = {}
        for relative in distribution.files or ():
            path = Path(distribution.locate_file(relative))
            if path.is_file() and (path.suffix in {".py", ".pyd", ".dll", ".so"} or ".so." in path.name):
                files[str(relative).replace("\\", "/")] = sha256_file(path)
        if not files:
            raise ValueError(f"No installed source/library bytes for {name}")
        packages[name] = dict(version=version, file_count=len(files), bytes_sha256=canonical_sha256(files))
    # BLAS/OpenMP libraries may live outside their importing distribution (for
    # example in conda's lib directory), so package versions alone are not enough.
    import numpy  # noqa: F401
    import scipy.linalg  # noqa: F401
    import sklearn.tree  # noqa: F401
    from threadpoolctl import threadpool_info
    native = sorted([dict(api=row["internal_api"], prefix=row["prefix"], version=row.get("version"),
                         sha256=sha256_file(Path(row["filepath"]))) for row in threadpool_info()],
                    key=lambda r: (r["api"], r["prefix"], r["sha256"]))
    return artifact("NUMERICAL_ENVIRONMENT", parents={"versions": result["content_hash"]},
                    versions=result, packages=packages, external_numerical_libraries=native,
                    interpreter_sha256=sha256_file(Path(sys.executable)))
