"""Concurrent publication must not invalidate another auxiliary shard's audit."""
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from hlt_classification.jetclass2_delphes.offline_aux import contracts


def listed_paths(monkeypatch, paths):
    # Freeze the directory enumeration, just as a filesystem iterator can lag
    # behind a sibling writer's atomic link/unlink publication.
    monkeypatch.setattr(Path, "rglob", lambda self, pattern: iter(paths))


@pytest.mark.parametrize("name", ["0001_hlt_pt.npz", "result.json"])
def test_publication_unlinks_temporary_after_metadata_read(tmp_path, monkeypatch, name):
    temporary = tmp_path / f".{name}.s9d1hofv.tmp"
    temporary.write_bytes(b"payload")
    listed_paths(monkeypatch, [temporary])
    original_stat = Path.stat
    original_unlink = os.unlink
    observed = False

    def read_then_finish(path, *args, **kwargs):
        nonlocal observed
        metadata = original_stat(path, *args, **kwargs)
        if path == temporary and not observed:
            observed = True
            os.link(temporary, tmp_path / name)
            original_unlink(temporary)
        return metadata

    monkeypatch.setattr(Path, "stat", read_then_finish)
    assert contracts.storage_audit(tmp_path) == dict(files=1, bytes=7, largest_bytes=7)
    assert observed


@pytest.mark.parametrize("destination_listed", [False, True])
def test_vanished_temporary_counts_published_destination_once(tmp_path, monkeypatch, destination_listed):
    final = tmp_path / "0001_hlt_pt.npz"
    temporary = tmp_path / ".0001_hlt_pt.npz.s9d1hofv.tmp"
    contracts.atomic_publish_bytes(final, b"payload")
    listed_paths(monkeypatch, [temporary] + ([final] if destination_listed else []))
    assert contracts.storage_audit(tmp_path) == dict(files=1, bytes=7, largest_bytes=7)


def test_aborted_publication_cleanup_is_tolerated(tmp_path, monkeypatch):
    listed_paths(monkeypatch, [tmp_path / ".result.json.s9d1hofv.tmp"])
    assert contracts.storage_audit(tmp_path) == dict(files=0, bytes=0, largest_bytes=0)


def test_real_publisher_counts_live_temporary_and_final(tmp_path, monkeypatch):
    original_unlink = os.unlink
    observations = []

    def audit_then_unlink(path, *args, **kwargs):
        if Path(path).parent == tmp_path:
            observations.append(contracts.storage_audit(tmp_path))
            result = original_unlink(path, *args, **kwargs)
            with monkeypatch.context() as listing:
                listed_paths(listing, [Path(path)])
                observations.append(contracts.storage_audit(tmp_path))
            return result
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", audit_then_unlink)
    contracts.atomic_publish_bytes(tmp_path / "payload.npz", b"payload")
    assert observations == [dict(files=2, bytes=14, largest_bytes=7),
                            dict(files=1, bytes=7, largest_bytes=7)]
    assert contracts.storage_audit(tmp_path) == dict(files=1, bytes=7, largest_bytes=7)


def test_submitter_claim_cleanup_is_tolerated_only_in_attempt_directory(tmp_path, monkeypatch):
    claim = tmp_path / "stages/PREPARE/attempts/initial/submission_in_progress.claim"
    listed_paths(monkeypatch, [claim])
    assert contracts.storage_audit(tmp_path) == dict(files=0, bytes=0, largest_bytes=0)
    listed_paths(monkeypatch, [tmp_path / claim.name])
    with pytest.raises(FileNotFoundError):
        contracts.storage_audit(tmp_path)


@pytest.mark.parametrize("name", ["result.json", "notes.tmp", ".unknown.tmp"])
def test_missing_nonpublication_file_still_fails(tmp_path, monkeypatch, name):
    listed_paths(monkeypatch, [tmp_path / name])
    with pytest.raises(FileNotFoundError):
        contracts.storage_audit(tmp_path)


@pytest.mark.parametrize("temporary", [False, True])
@pytest.mark.parametrize("limit", ["MAX_FILE", "MAX_TOTAL"])
def test_live_payloads_and_temporaries_remain_size_limited(tmp_path, monkeypatch, temporary, limit):
    name = ".payload.npz.s9d1hofv.tmp" if temporary else "payload.npz"
    (tmp_path / name).write_bytes(b"12345678")
    monkeypatch.setattr(contracts, limit, 7)
    with pytest.raises(ValueError, match="storage envelope"):
        contracts.storage_audit(tmp_path)


def test_total_limit_includes_multiple_files_and_directories(tmp_path, monkeypatch):
    nested = tmp_path / "shard"
    nested.mkdir()
    (nested / "a.npz").write_bytes(b"1234")
    (tmp_path / "b.json").write_bytes(b"5678")
    assert contracts.storage_audit(tmp_path) == dict(files=2, bytes=8, largest_bytes=4)
    monkeypatch.setattr(contracts, "MAX_TOTAL", 7)
    with pytest.raises(ValueError, match="storage envelope"):
        contracts.storage_audit(tmp_path)


def test_recovered_destination_is_still_size_limited(tmp_path, monkeypatch):
    (tmp_path / "payload.npz").write_bytes(b"12345678")
    listed_paths(monkeypatch, [tmp_path / ".payload.npz.s9d1hofv.tmp"])
    monkeypatch.setattr(contracts, "MAX_FILE", 7)
    with pytest.raises(ValueError, match="storage envelope"):
        contracts.storage_audit(tmp_path)


@pytest.mark.parametrize("published_fallback", [False, True])
def test_symlinks_including_dangling_links_fail(tmp_path, monkeypatch, published_fallback):
    final = tmp_path / "payload.npz"
    temporary = tmp_path / ".payload.npz.s9d1hofv.tmp"
    listed_paths(monkeypatch, [temporary if published_fallback else final])
    original_stat = Path.stat

    def symlink_stat(path, *args, **kwargs):
        if path == final:
            if kwargs.get("follow_symlinks", True):
                raise FileNotFoundError(str(path))
            return SimpleNamespace(st_mode=stat.S_IFLNK, st_size=0)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", symlink_stat)
    with pytest.raises(ValueError, match="Symlinks forbidden"):
        contracts.storage_audit(tmp_path)


def test_permission_errors_are_not_swallowed(tmp_path, monkeypatch):
    path = tmp_path / ".payload.npz.s9d1hofv.tmp"
    listed_paths(monkeypatch, [path])

    def denied(*args, **kwargs):
        raise PermissionError("unreadable output")

    monkeypatch.setattr(Path, "stat", denied)
    with pytest.raises(PermissionError, match="unreadable output"):
        contracts.storage_audit(tmp_path)
