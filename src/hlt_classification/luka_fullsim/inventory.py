"""Authenticate the transfer and scan scalar eligibility, never constituents."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import numpy as np
import uproot

from hlt_classification.data.cache_contracts import canonical_sha256, require_sha256, sha256_file
from hlt_classification.jetclass2_delphes.inventory import latest_tree
from hlt_classification.jetclass2_delphes.schema import CLASS_NAMES, RAW_TO_CLASS, map_labels, validate_tree_schema
from .contracts import SOURCE, SCALARS, artifact, selection, source_path, validate

CUTFLOW = ("excluded_qcd_source", "excluded_unmatched", "excluded_empty", "excluded_offline_pt", "selected")


def transfer_manifest(container: Path, expected_files: int) -> tuple[dict, list[dict]]:
    container = Path(container).resolve(strict=True)
    if type(expected_files) is not int or expected_files < 1:
        raise ValueError("Expected file count must be positive")
    marker, manifest = container / "source.txt", container / "source.sha256"
    if marker.read_text(encoding="utf-8").strip() != SOURCE:
        raise ValueError("Transfer source marker differs")
    records, names, hashes = [], set(), set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if not match:
            raise ValueError("Malformed SHA256 manifest line")
        digest, name = match.groups()
        if name.startswith("./"):
            name = name[2:]
        source_path(container / "raw", name)
        if name.casefold() in names or digest in hashes:
            raise ValueError("Duplicate source path or content in transfer")
        names.add(name.casefold())
        hashes.add(digest)
        records.append(dict(path=name, sha256=digest))
    records.sort(key=lambda r: r["path"])
    actual = sorted(p.relative_to(container / "raw").as_posix()
                    for p in (container / "raw").rglob("*.root"))
    if len(records) != expected_files or actual != [r["path"] for r in records]:
        raise ValueError("Transfer ROOT coverage/count differs (missing or extra files)")
    if {r["path"].split("/")[1] for r in records} != {"train_higgs2p", "train_qcd"}:
        raise ValueError("Both signal and QCD source groups are required")
    return dict(source=SOURCE, source_marker_sha256=sha256_file(marker),
                source_manifest_sha256=sha256_file(manifest)), records


def eligible(batch: dict, source: str) -> tuple[np.ndarray, np.ndarray, dict]:
    arrays = {key: np.asarray(batch[key]) for key in SCALARS}
    raw = arrays["jet_label"]
    labels = map_labels(raw)  # Validate even excluded/unmatched raw labels.
    if any(a.ndim != 1 or len(a) != len(raw) for a in arrays.values()):
        raise ValueError("Scalar branch shapes differ")
    if arrays["hlt_matched"].dtype.kind != "b":
        raise ValueError("hlt_matched must be boolean")
    for key in ("jet_nparticles", "hlt_jet_nparticles"):
        if arrays[key].dtype.kind not in "iu" or np.any(arrays[key] < 0):
            raise ValueError("Particle counts must be nonnegative integers")
    if arrays["jet_pt"].dtype.kind != "f" or not np.isfinite(arrays["jet_pt"]).all():
        raise ValueError("Nonfinite or invalid offline pT")
    if source not in ("train_higgs2p", "train_qcd"):
        raise ValueError("Unknown source category")
    masks = (
        ("excluded_qcd_source", (labels != 0) | (source == "train_qcd")),
        ("excluded_unmatched", arrays["hlt_matched"]),
        ("excluded_empty", (arrays["jet_nparticles"] > 0) & (arrays["hlt_jet_nparticles"] > 0)),
        ("excluded_offline_pt", arrays["jet_pt"] > 200),
    )
    keep = np.ones(len(raw), bool)
    counts = {}
    for name, mask in masks:
        counts[name] = int(np.count_nonzero(keep & ~mask))
        keep &= mask
    counts["selected"] = int(keep.sum())
    return keep, labels, counts


def build_inventory(container: Path, *, expected_files: int = 890, step_size: int = 100000) -> dict:
    if type(step_size) is not int or step_size < 1:
        raise ValueError("Invalid scalar chunk size")
    container = Path(container).resolve(strict=True)
    transfer, manifest = transfer_manifest(container, expected_files)
    common_schema, files = None, []
    for index, item in enumerate(manifest, 1):
        path, source = source_path(container / "raw", item["path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Source checksum differs: {item['path']}")
        classes = [[] for _ in CLASS_NAMES]
        raw_counts, exclusions = Counter(), Counter({name: 0 for name in CUTFLOW})
        with uproot.open(path) as handle:
            key, tree = latest_tree(handle)
            schema = validate_tree_schema(tree)
            if schema.get("jet_pt") not in ("float", "double", "float32_t", "float64_t"):
                raise ValueError("Missing/invalid offline jet_pt scalar schema")
            if common_schema is None:
                common_schema = schema
            if schema != common_schema:
                raise ValueError(f"ROOT schema differs: {item['path']}")
            start = 0
            for batch in tree.iterate(list(SCALARS), step_size=step_size, library="np"):
                keep, labels, counts = eligible(batch, source)
                raw_counts.update(map(int, batch["jet_label"]))
                exclusions.update(counts)
                for label in range(len(CLASS_NAMES)):
                    classes[label].extend((start + np.flatnonzero(keep & (labels == label))).tolist())
                start += len(labels)
            if start != tree.num_entries:
                raise ValueError("Scalar scan entry count differs")
        if sha256_file(path) != item["sha256"]:
            raise ValueError("Source changed during inventory")
        files.append(dict(**item, source=source, bytes=path.stat().st_size,
                          tree_key=key, entries=start,
                          raw_label_counts={str(k): int(v) for k, v in sorted(raw_counts.items())},
                          cutflow=dict(exclusions), entries_by_class=classes,
                          selected_class_counts=list(map(len, classes))))
        if index % 25 == 0 or index == len(manifest):
            print(f"FullSim scalar inventory: {index}/{len(manifest)} files", flush=True)
    if transfer_manifest(container, expected_files) != (transfer, manifest):
        raise ValueError("Transfer manifest changed during inventory")
    result = artifact("INVENTORY", parents={}, transfer=transfer, selection=selection(),
                      schema=common_schema, schema_sha256=canonical_sha256(common_schema),
                      tree_policy="latest_cycle_only", files=files, file_count=len(files),
                      raw_entries=sum(r["entries"] for r in files),
                      selected_class_counts=np.sum([r["selected_class_counts"] for r in files], axis=0).tolist(),
                      particle_arrays_decoded=False, pre_split_scalar_metadata_accessed=True,
                      final_test_accessed=False)
    validate_inventory(result)
    return result


def validate_inventory(value: dict) -> str:
    digest = validate(value, "INVENTORY")
    if (value["selection"] != selection() or value["transfer"]["source"] != SOURCE
            or value["schema_sha256"] != canonical_sha256(value["schema"])
            or value["particle_arrays_decoded"] is not False
            or value["final_test_accessed"] is not False
            or value["pre_split_scalar_metadata_accessed"] is not True
            or value["parents"] != {} or value["tree_policy"] != "latest_cycle_only"):
        raise ValueError("Inventory semantics differ")
    for key in ("source_marker_sha256", "source_manifest_sha256"):
        require_sha256(value["transfer"][key], name=key)
    files = value["files"]
    if not files or len(files) != value["file_count"]:
        raise ValueError("Inventory file coverage differs")
    paths, digests = [], []
    for r in files:
        _, source = source_path(Path.cwd(), r["path"])
        require_sha256(r["sha256"], name="source file")
        paths.append(r["path"])
        digests.append(r["sha256"])
        if (source != r["source"] or not re.fullmatch(r"tree;[1-9][0-9]*", r["tree_key"])
                or type(r["entries"]) is not int or r["entries"] < 0
                or type(r["bytes"]) is not int or r["bytes"] < 1):
            raise ValueError("Invalid source record")
        classes = r["entries_by_class"]
        if len(classes) != len(CLASS_NAMES) or list(map(len, classes)) != r["selected_class_counts"]:
            raise ValueError("Selected class counts differ")
        seen = set()
        for entries in classes:
            if (any(type(n) is not int or not 0 <= n < r["entries"] for n in entries)
                    or entries != sorted(set(entries)) or seen.intersection(entries)):
                raise ValueError("Duplicate/out-of-range source entries")
            seen.update(entries)
        flow = r["cutflow"]
        raw_counts = r["raw_label_counts"]
        if (any(k not in {str(n) for n in RAW_TO_CLASS} for k in raw_counts)
                or any(type(n) is not int or n < 0 for n in raw_counts.values())):
            raise ValueError("Invalid raw label counts")
        mapped_counts = [0] * len(CLASS_NAMES)
        for k, n in raw_counts.items():
            mapped_counts[RAW_TO_CLASS[int(k)]] += n
        if any(len(es) > n for es, n in zip(classes, mapped_counts)):
            raise ValueError("Selected class counts exceed raw label capacity")
        if (set(flow) != set(CUTFLOW)
                or any(type(n) is not int or n < 0 for n in flow.values())
                or sum(flow.values()) != r["entries"] or flow["selected"] != len(seen)
                or sum(r["raw_label_counts"].values()) != r["entries"]
                or (source == "train_higgs2p" and classes[0])):
            raise ValueError("Selection cutflow differs")
    if (paths != sorted(paths) or len(set(p.casefold() for p in paths)) != len(paths)
            or len(set(digests)) != len(digests)):
        raise ValueError("Duplicate/unordered source paths or contents")
    if (sum(r["entries"] for r in files) != value["raw_entries"]
            or np.sum([r["selected_class_counts"] for r in files], axis=0).tolist()
            != value["selected_class_counts"]):
        raise ValueError("Inventory totals differ")
    return digest


def verify_source_bytes(container: Path, inventory: dict) -> None:
    """Reauthenticate the entire snapshot without decoding even scalar arrays."""
    validate_inventory(inventory)
    transfer, records = transfer_manifest(container, inventory["file_count"])
    if (transfer != inventory["transfer"] or records != [dict(path=r["path"], sha256=r["sha256"])
                                                        for r in inventory["files"]]):
        raise ValueError("Transfer parents differ from frozen inventory")
    for record in inventory["files"]:
        path, _ = source_path(Path(container) / "raw", record["path"])
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Source checksum differs: {record['path']}")
