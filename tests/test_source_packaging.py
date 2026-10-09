"""Local development worktrees must not enter deployable source snapshots."""
from pathlib import Path
import subprocess

import pytest

from hlt_classification.provenance import (
    capture_source_snapshot, validate_source_snapshot,
)


PROJECT = Path(__file__).resolve().parents[1]


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True,
        capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()


def init(root):
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Packaging Test")
    git(root, "config", "user.email", "packaging@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    git(root, "config", "core.autocrlf", "false")
    (root / "source.txt").write_text("tracked scientific source\n", encoding="utf-8")
    git(root, "add", "source.txt")
    git(root, "commit", "-q", "-m", "fixture")


def test_repository_index_has_no_gitlinks_or_local_worktrees():
    rows = git(PROJECT, "ls-files", "--stage", "-z").split("\0")
    for row in filter(None, rows):
        metadata, path = row.split("\t", 1)
        assert metadata.split()[0] != "160000", f"Non-file gitlink in deployable source: {path}"
        assert not path.startswith(".worktrees/"), f"Local worktree must remain untracked: {path}"


def test_local_worktrees_are_ignored_but_scientific_source_is_not():
    for path in (".worktrees/example/README.md", ".worktrees/another/src/module.py"):
        assert git(PROJECT, "check-ignore", "--no-index", "--", path) == path
    result = subprocess.run(
        ["git", "-C", str(PROJECT), "check-ignore", "--no-index", "--",
         "src/hlt_classification/provenance.py", "tests/test_source_packaging.py"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1 and not result.stdout


def test_index_only_cleanup_preserves_local_worktree_and_fixes_fresh_clone(tmp_path):
    original = tmp_path / "original"
    init(original)
    original_head = git(original, "rev-parse", "HEAD")
    relative = ".worktrees/developer-work"
    # A gitlink without submodule metadata reproduces the accidental packaging.
    git(original, "update-index", "--add", "--cacheinfo", f"160000,{original_head},{relative}")
    git(original, "commit", "-q", "-m", "accidental gitlink")
    before = tmp_path / "before"
    git(tmp_path, "clone", "-q", "--no-hardlinks", "-c", "core.autocrlf=false",
        str(original), str(before))
    assert git(before, "status", "--porcelain") == ""
    with pytest.raises(ValueError, match="tracked source file is absent: .worktrees/developer-work"):
        capture_source_snapshot(before)

    local = original / relative
    local.parent.mkdir()
    init(local)
    local_head = git(local, "rev-parse", "HEAD")
    (local / "source.txt").write_text("uncommitted work: preserve me\n", encoding="utf-8")
    (local / "notes.txt").write_text("untracked work: preserve me\n", encoding="utf-8")
    contents = {name: (local / name).read_bytes() for name in ("source.txt", "notes.txt")}
    status = git(local, "status", "--porcelain", "--untracked-files=all")
    (original / ".gitignore").write_bytes((PROJECT / ".gitignore").read_bytes())
    git(original, "rm", "--cached", "--", relative)
    git(original, "add", ".gitignore")
    git(original, "commit", "-q", "-m", "untrack local development worktree")
    assert local.is_dir() and (local / ".git").is_dir()
    assert git(local, "rev-parse", "HEAD") == local_head
    assert git(local, "status", "--porcelain", "--untracked-files=all") == status
    assert {name: (local / name).read_bytes() for name in contents} == contents
    fixed_snapshot = capture_source_snapshot(original)
    assert fixed_snapshot["worktree_clean"] is True
    assert fixed_snapshot["tracked_file_count"] == 2

    after = tmp_path / "after"
    git(tmp_path, "clone", "-q", "--no-hardlinks", "-c", "core.autocrlf=false",
        str(original), str(after))
    assert not (after / ".worktrees").exists()
    assert capture_source_snapshot(after) == fixed_snapshot
    validate_source_snapshot(fixed_snapshot, repository=after)
    (after / "source.txt").write_text("real source drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        validate_source_snapshot(fixed_snapshot, repository=after)


def test_missing_ordinary_tracked_file_still_fails_closed(tmp_path):
    root = tmp_path / "missing_source"
    init(root)
    (root / "source.txt").unlink()
    with pytest.raises(ValueError, match="tracked source file is absent: source.txt"):
        capture_source_snapshot(root, require_clean=False)
