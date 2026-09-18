from pathlib import Path
import tarfile

import pytest

from hlt_classification.data.cache_contracts import sha256_file, with_content_hash
from hlt_classification.jetclass2_delphes.partial_snapshot import (
    build_partial_snapshot_plan,
)
from hlt_classification.jetclass2_delphes.partial_snapshot_transfer import (
    build_transfer_archive,
    extract_transfer_archive,
)
from test_jetclass2_delphes_partial_snapshot import synthetic_inventory


def _files(tmp_path: Path):
    inventory = synthetic_inventory(files_per_source=24, rows_per_class=100)
    root = tmp_path / "source"
    for row in inventory["files"]:
        path = root / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (row["path"] + "\n").encode().ljust(10_000, b"x")
        path.write_bytes(payload)
        row["sha256"] = row["group_id"] = sha256_file(path)
    # Rebuild the inventory hash/totals after replacing synthetic digests.
    inventory = with_content_hash(inventory)
    plan = build_partial_snapshot_plan(
        inventory, training_sizes=(2_000,), validation_size=1_000,
        test_size=1_000, headroom_fraction=0,
    )
    return root, inventory, plan


def test_deterministic_archive_and_safe_extraction(tmp_path):
    root, inventory, plan = _files(tmp_path)
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    transfer = build_transfer_archive(
        data_root=root, inventory=inventory, plan=plan, archive_path=first,
    )
    other = build_transfer_archive(
        data_root=root, inventory=inventory, plan=plan, archive_path=second,
    )
    assert transfer["archive_sha256"] == other["archive_sha256"]
    assert transfer["archive_size_bytes"] == other["archive_size_bytes"]
    receipt = extract_transfer_archive(
        archive_path=first, transfer=transfer, inventory=inventory, plan=plan,
        output_parent=tmp_path / "destination", receipt_path=tmp_path / "receipt.json",
    )
    assert receipt["extraction_complete"] is True
    for path in plan["selected_paths"]:
        assert sha256_file(tmp_path / "destination" / "jetclass2" / path) == next(
            row["sha256"] for row in inventory["files"] if row["path"] == path
        )


def test_archive_rejects_changed_source_and_archive(tmp_path):
    root, inventory, plan = _files(tmp_path)
    selected = root / plan["selected_paths"][0]
    selected.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source file differs"):
        build_transfer_archive(
            data_root=root, inventory=inventory, plan=plan,
            archive_path=tmp_path / "bad-source.tar",
        )

    root, inventory, plan = _files(tmp_path / "fresh")
    archive = tmp_path / "good.tar"
    transfer = build_transfer_archive(
        data_root=root, inventory=inventory, plan=plan, archive_path=archive,
    )
    with archive.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="archive bytes differ"):
        extract_transfer_archive(
            archive_path=archive, transfer=transfer, inventory=inventory, plan=plan,
            output_parent=tmp_path / "bad-destination", receipt_path=tmp_path / "bad-receipt.json",
        )


def test_archive_rejects_member_tampering(tmp_path):
    root, inventory, plan = _files(tmp_path)
    archive = tmp_path / "good.tar"
    transfer = build_transfer_archive(
        data_root=root, inventory=inventory, plan=plan, archive_path=archive,
    )
    # Repack one extra member, then update only the transport byte fields. The
    # member-coverage gate must still reject it.
    tampered = tmp_path / "tampered.tar"
    with tarfile.open(archive, "r:") as source, tarfile.open(tampered, "w") as target:
        for member in source.getmembers():
            target.addfile(member, source.extractfile(member))
        info = tarfile.TarInfo("jetclass2/extra.root")
        info.size = 0
        target.addfile(info)
    changed = dict(transfer)
    changed["archive_name"] = tampered.name
    changed["archive_sha256"] = sha256_file(tampered)
    changed["archive_size_bytes"] = tampered.stat().st_size
    changed = with_content_hash(changed)
    with pytest.raises(ValueError, match="member coverage"):
        extract_transfer_archive(
            archive_path=tampered, transfer=changed, inventory=inventory, plan=plan,
            output_parent=tmp_path / "bad-members", receipt_path=tmp_path / "bad-members.json",
        )
