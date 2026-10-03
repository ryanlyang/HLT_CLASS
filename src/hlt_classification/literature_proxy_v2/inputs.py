"""Authenticated v1 training blocks; never read raw ROOT or labels."""
from pathlib import Path
import hashlib

import numpy as np

from hlt_classification.cms2jc2_response.bridge import Particles
from hlt_classification.jetclass2_delphes.contracts import relative_file
from hlt_classification.literature_proxy import contracts as v1, worker as old_worker
from hlt_classification.literature_proxy.kernel import recipe as old_recipe
from hlt_classification.literature_proxy.population import select
from .contracts import read_reference, reference, sha256_file

FIELDS = ("p4", "charge", "category", "tracking", "valid")


def authenticate(parent_ref, receipt_ref):
    parent = read_reference(parent_ref)
    v1.validate(parent, "SPEC")
    root = Path(parent["root"])
    if Path(parent_ref["path"]).resolve() != (root / "study_spec.json").resolve():
        raise ValueError("Parent spec path differs")
    if Path(receipt_ref["path"]).resolve() != (root / "receipt.json").resolve():
        raise ValueError("Parent receipt path differs")
    receipt = read_reference(receipt_ref)
    v1.validate(receipt, "RECEIPT")
    required = {"report.json", "statistics.csv", "overlays.pdf", "examples.json"}
    for selected in parent["population"]["files"]:
        key = hashlib.sha256(selected["path"].encode()).hexdigest()[:20]
        required.update((f"files/{key}.json", f"blocks/{key}.npz", f"lineage/{key}.json"))
    if set(receipt["outputs"]) != required:
        raise ValueError("Parent completion manifest differs")
    if parent["recipe"] != old_recipe():
        raise ValueError("Expected unchanged literature v1 recipe")
    for key in ("native_hlt_particles_accessed", "validation_accessed", "final_test_accessed", "production_qualified"):
        if parent.get(key) is not False:
            raise PermissionError("Parent pilot scope differs")
    inv, profile = (read_reference(parent[k]) for k in ("inventory", "profile"))
    if parent["population"] != select(inv, profile, parent["population"]["jets"]):
        raise ValueError("Parent train population differs")
    report = old_worker.results(parent)  # All saved physical/diagnostic bytes, not raw ROOT.
    if not report["replay"]["exact"] or report.get("final_test_accessed") is not False:
        raise ValueError("Parent replay/access evidence differs")
    expected, found = parent["population"]["files"], {}
    for name in receipt["outputs"]:
        if not name.startswith("files/") or not name.endswith(".json"):
            continue
        file = v1.load_json(relative_file(root, name))
        v1.validate(file, "FILE_REPORT")
        selected = file["selected"]
        if file["parents"] != dict(spec=parent["content_hash"]) or selected not in expected or selected["path"] in found:
            raise ValueError("Parent per-file lineage/coverage differs")
        for output, digest in file["outputs"].items():
            if receipt["outputs"].get(output) != digest:
                raise ValueError("Parent per-file output not sealed by receipt")
        block = [n for n in file["outputs"] if n.startswith("blocks/") and n.endswith(".npz")]
        lineage = [n for n in file["outputs"] if n.startswith("lineage/") and n.endswith(".json")]
        if len(block) != 1 or len(lineage) != 1:
            raise ValueError("Parent physical/lineage outputs differ")
        found[selected["path"]] = dict(selected=selected,
                                      block=reference(relative_file(root, block[0])),
                                      lineage=reference(relative_file(root, lineage[0])))
    if set(found) != {r["path"] for r in expected}:
        raise ValueError("Incomplete parent training coverage")
    return parent, [found[r["path"]] for r in expected]


def read_rows(record):
    """Validate physical shapes/offsets/identity and trace before yielding a file."""
    path = record["block"]["path"]
    if sha256_file(path) != record["block"]["sha256"]:
        raise ValueError("Parent block checksum differs")
    trace = read_reference(record["lineage"])
    identities = record["selected"]["identities"]
    if trace.get("purpose") != "diagnostic_only_forbidden_model_input" or len(trace["rows"]) != len(identities):
        raise ValueError("Parent lineage scope/coverage differs")
    with np.load(path, allow_pickle=False) as data:
        if data["identity"].tolist() != identities:
            raise ValueError("Parent block identities differ")
        arrays, offsets = {}, {}
        for side in ("OFFLINE", "NOMINAL"):
            offset = data[f"{side}_offsets"]
            if offset.dtype.kind not in "iu" or offset.ndim != 1 or len(offset) != len(identities)+1 or offset[0] != 0 or np.any(offset[1:] < offset[:-1]):
                raise ValueError("Invalid parent offsets")
            arrays[side] = {field: data[f"{side}_{field}"] for field in FIELDS}
            if any(len(a) != offset[-1] for a in arrays[side].values()):
                raise ValueError("Parent offsets/physical coverage differ")
            offsets[side] = offset
    if sha256_file(path) != record["block"]["sha256"]:
        raise ValueError("Parent block changed during read")
    for i, identity in enumerate(identities):
        row = trace["rows"][i]
        if row["identity"] != identity:
            raise ValueError("Parent trace identity differs")
        offline_keys = tuple(f"part:{j:08d}" for j in range(int(offsets["OFFLINE"][i+1]-offsets["OFFLINE"][i])))
        ancestry = tuple(tuple(a) for a in row["descendants"]["NOMINAL"])
        flattened = [k for a in ancestry for k in a]
        if (tuple(row["offline_keys"]) != offline_keys or any(len(a) not in (1, 2) for a in ancestry)
                or len(set(flattened)) != len(flattened) or not set(flattened) <= set(offline_keys)):
            raise ValueError("Parent trace/native keys differ")
        physical = []
        for side, keys in (("OFFLINE", offline_keys), ("NOMINAL", tuple("+".join(a) for a in ancestry))):
            lo, hi = map(int, offsets[side][i:i+2])
            if len(keys) != hi-lo:
                raise ValueError("Parent ancestry/physical coverage differs")
            physical.append(Particles(**{f: a[lo:hi] for f, a in arrays[side].items()}, keys=keys))
        yield identity, *physical, ancestry
