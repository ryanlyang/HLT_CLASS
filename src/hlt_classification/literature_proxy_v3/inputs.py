"""Read only sealed v1 OFFLINE and v2 control blocks, never raw datasets."""
import hashlib
from pathlib import Path

import numpy as np

from hlt_classification.cms2jc2_response.bridge import Particles
from hlt_classification.literature_proxy_v2 import contracts as v2, inputs as old, worker
from hlt_classification.literature_proxy_v2.kernel import recipe, validate_calibration
from .contracts import load_json, read_reference, reference, sha256_file

FIELDS = old.FIELDS


def file_key(record):
    return hashlib.sha256(record["v1"]["selected"]["path"].encode()).hexdigest()[:20]


def authenticate(parent_ref, receipt_ref, calibration_ref):
    parent = read_reference(parent_ref)
    v2.validate(parent, "SPEC")
    root = Path(parent["root"])
    for ref, name in ((parent_ref, "study_spec.json"), (receipt_ref, "receipt.json"),
                      (calibration_ref, "calibration.json")):
        if Path(ref["path"]).resolve() != (root / name).resolve():
            raise ValueError("Parent artifact path differs")
    receipt = read_reference(receipt_ref)
    v2.validate(receipt, "RECEIPT")
    if parent["recipe"] != recipe():
        raise ValueError("Expected unchanged count38 v2 recipe")
    for key in ("native_hlt_particles_accessed", "validation_accessed", "final_test_accessed", "production_qualified"):
        if parent.get(key) is not False:
            raise PermissionError("Parent pilot scope differs")
    original, records = old.authenticate(parent["parent_spec"], parent["parent_receipt"])
    if parent["population"] != original["population"] or parent["inputs"] != records:
        raise ValueError("Parent input/population lineage differs")
    report = worker.results(parent)  # Verify every sealed v2 output, independent of old worktree.
    rates = read_reference(calibration_ref)
    validate_calibration(rates, parent["content_hash"])
    if rates != report["calibration"] or not report["replay"]["exact"] or rates["jets"] != parent["population"]["jets"]:
        raise ValueError("Parent calibration/replay differs")
    inputs, counts = [], []
    for record in records:
        key = hashlib.sha256(record["selected"]["path"].encode()).hexdigest()[:20]
        file = load_json(root / "files" / f"{key}.json")
        v2.validate(file, "FILE_REPORT")
        expected = {f"blocks/{key}.npz", f"lineage/{key}.json"}
        if (file["parents"] != dict(spec=parent["content_hash"], calibration=rates["content_hash"])
                or file["input"] != record or set(file["outputs"]) != expected
                or any(receipt["outputs"].get(p) != h for p, h in file["outputs"].items())):
            raise ValueError("Parent file lineage differs")
        inputs.append(dict(v1=record, block=reference(root / f"blocks/{key}.npz"),
                           lineage=reference(root / f"lineage/{key}.json"), counts=file["counts"]))
        counts.append(file["counts"])
    if worker.sum_counts(counts) != report["counts"]:
        raise ValueError("Parent aggregate counts differ")
    return parent, rates, inputs


def read_rows(record):
    path = record["block"]["path"]
    if sha256_file(path) != record["block"]["sha256"]:
        raise ValueError("Parent block checksum differs")
    trace = read_reference(record["lineage"])
    identities = record["v1"]["selected"]["identities"]
    if trace.get("purpose") != "diagnostic_only_forbidden_model_input" or len(trace["rows"]) != len(identities):
        raise ValueError("Parent trace scope/coverage differs")
    with np.load(path, allow_pickle=False) as data:
        if data["identity"].tolist() != identities:
            raise ValueError("Parent block identities differ")
        offsets = data["COUNT38_V2_offsets"]
        if (offsets.dtype.kind not in "iu" or offsets.ndim != 1 or len(offsets) != len(identities)+1
                or offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1])):
            raise ValueError("Invalid parent offsets")
        arrays = {field: data[f"COUNT38_V2_{field}"] for field in FIELDS}
        if any(len(a) != offsets[-1] for a in arrays.values()):
            raise ValueError("Parent physical coverage differs")
    if sha256_file(path) != record["block"]["sha256"]:
        raise ValueError("Parent block changed during read")
    for i, (identity, offline, _, _) in enumerate(old.read_rows(record["v1"])):
        row = trace["rows"][i]
        ancestry = tuple(tuple(a) for a in row["descendants"])
        flattened = [k for a in ancestry for k in a]
        lo, hi = map(int, offsets[i:i+2])
        if (row["identity"] != identity or tuple(row["offline_keys"]) != offline.keys
                or len(ancestry) != hi-lo or any(len(a) not in (1, 2) for a in ancestry)
                or len(set(flattened)) != len(flattened) or not set(flattened) <= set(offline.keys)):
            raise ValueError("Parent trace identity/ancestry differs")
        control = Particles(**{k: a[lo:hi] for k, a in arrays.items()},
                            keys=tuple("+".join(a) for a in ancestry))
        yield identity, offline, control, ancestry
