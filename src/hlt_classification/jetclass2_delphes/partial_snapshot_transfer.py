"""Deterministic archive transport for a planned JetClass2 partial snapshot."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tarfile

from hlt_classification.data.cache_contracts import sha256_file, write_immutable_json

from .contracts import artifact, relative_file, validate
from .inventory import validate_inventory
from .partial_snapshot import validate_partial_snapshot_plan


ARCHIVE_ROOT = "jetclass2"
ARCHIVE_POLICY = "ustar_normalized_uid_gid_names_mtime_zero_mode_0600_v1"


def _selected_records(inventory: dict, plan: dict) -> list[dict]:
    validate_partial_snapshot_plan(plan, inventory)
    rows = {row["path"]: row for row in inventory["files"]}
    if set(plan["selected_paths"]) - set(rows):
        raise ValueError("Partial snapshot plan escapes parent inventory")
    return [rows[path] for path in plan["selected_paths"]]


def build_transfer_archive(
    *, data_root: Path, inventory: dict, plan: dict, archive_path: Path,
) -> dict:
    """Write one normalized uncompressed tar after authenticating source files."""
    validate_inventory(inventory)
    records = _selected_records(inventory, plan)
    root = Path(data_root).resolve(strict=True)
    archive = Path(archive_path).resolve()
    if archive.exists() or archive.is_relative_to(root):
        raise FileExistsError("Transfer archive must be new and outside raw data")
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, mode="w", format=tarfile.USTAR_FORMAT) as handle:
            for index, row in enumerate(records, 1):
                source = relative_file(root, row["path"])
                before = source.stat()
                if before.st_size != row["size_bytes"] or sha256_file(source) != row["sha256"]:
                    raise ValueError(f"Selected source file differs: {row['path']}")
                info = tarfile.TarInfo(f"{ARCHIVE_ROOT}/{row['path']}")
                info.size = row["size_bytes"]
                info.mode = 0o600
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with source.open("rb") as stream:
                    handle.addfile(info, stream)
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f"Selected source changed during archive: {row['path']}")
                print(
                    f"JC2 phase=partial_archive file={index}/{len(records)} path={row['path']}",
                    flush=True,
                )
    except BaseException:
        # A failed archive has no scientific meaning. Leave no plausible final
        # artifact, but do not touch the source or any pre-existing path.
        if archive.exists():
            archive.unlink()
        raise
    result = artifact(
        "PARTIAL_SNAPSHOT_TRANSFER",
        parent_inventory_sha256=inventory["content_hash"],
        partial_snapshot_plan_sha256=plan["content_hash"],
        archive_policy=ARCHIVE_POLICY,
        archive_name=archive.name,
        archive_sha256=sha256_file(archive),
        archive_size_bytes=archive.stat().st_size,
        archive_root=ARCHIVE_ROOT,
        selected_file_count=len(records),
        selected_size_bytes=sum(row["size_bytes"] for row in records),
        selected_paths=[row["path"] for row in records],
        destination_reinventory_required=True,
        final_test_accessed=False,
    )
    validate_transfer(result, inventory, plan)
    return result


def validate_transfer(value: dict, inventory: dict, plan: dict) -> str:
    digest = validate(value, "PARTIAL_SNAPSHOT_TRANSFER")
    records = _selected_records(inventory, plan)
    if (
        value["parent_inventory_sha256"] != inventory["content_hash"]
        or value["partial_snapshot_plan_sha256"] != plan["content_hash"]
        or value["archive_policy"] != ARCHIVE_POLICY
        or value["archive_root"] != ARCHIVE_ROOT
        or value["selected_file_count"] != len(records)
        or value["selected_size_bytes"] != sum(row["size_bytes"] for row in records)
        or value["selected_paths"] != [row["path"] for row in records]
        or value["destination_reinventory_required"] is not True
        or value["final_test_accessed"] is not False
    ):
        raise ValueError("Partial snapshot transfer contract differs")
    return digest


def extract_transfer_archive(
    *, archive_path: Path, transfer: dict, inventory: dict, plan: dict,
    output_parent: Path, receipt_path: Path,
) -> dict:
    """Safely extract exact registered members and verify their content hashes."""
    transfer_hash = validate_transfer(transfer, inventory, plan)
    records = _selected_records(inventory, plan)
    archive = Path(archive_path).resolve(strict=True)
    parent = Path(output_parent).resolve()
    destination = parent / ARCHIVE_ROOT
    receipt = Path(receipt_path).resolve()
    if destination.exists() or receipt.exists() or receipt.is_relative_to(destination):
        raise FileExistsError("Extraction destination and receipt must be new and disjoint")
    if (
        archive.name != transfer["archive_name"]
        or archive.stat().st_size != transfer["archive_size_bytes"]
        or sha256_file(archive) != transfer["archive_sha256"]
    ):
        raise ValueError("Transferred archive bytes differ")
    expected_names = [f"{ARCHIVE_ROOT}/{row['path']}" for row in records]
    parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:") as handle:
        members = handle.getmembers()
        if [member.name for member in members] != expected_names:
            raise ValueError("Transfer archive member coverage/order differs")
        for index, (member, row) in enumerate(zip(members, records), 1):
            if (
                not member.isfile()
                or member.size != row["size_bytes"]
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
                or member.mtime != 0
                or member.mode != 0o600
            ):
                raise ValueError(f"Transfer archive metadata differs: {member.name}")
            target = relative_file(parent, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"Transfer archive member is unreadable: {member.name}")
            digest = hashlib.sha256()
            with target.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != row["sha256"]:
                raise ValueError(f"Extracted ROOT content differs: {row['path']}")
            print(
                f"JC2 phase=partial_extract file={index}/{len(records)} path={row['path']}",
                flush=True,
            )
    result = artifact(
        "PARTIAL_SNAPSHOT_EXTRACTION",
        transfer_sha256=transfer_hash,
        parent_inventory_sha256=inventory["content_hash"],
        partial_snapshot_plan_sha256=plan["content_hash"],
        archive_sha256=transfer["archive_sha256"],
        data_root=str(destination),
        selected_file_count=len(records),
        selected_size_bytes=sum(row["size_bytes"] for row in records),
        destination_reinventory_required=True,
        extraction_complete=True,
        final_test_accessed=False,
    )
    write_immutable_json(receipt, result)
    return result
