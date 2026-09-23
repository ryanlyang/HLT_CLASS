"""Bounded real-train parity before K2's expensive full-population caches.

This supplements, rather than replaces, the later population-derived parity,
batch probes and execution gate. It never recomputes matching or reads test.
"""
from itertools import islice
from pathlib import Path

import numpy as np

from hlt_classification.data.cache_contracts import write_immutable_json
from .acceptance import installed_parity
from .cache import RamBlock, RamCache
from .concat_k2_campaign import artifact
from .concat_k2_data import load_assignment
from .concat_k2_model import storage_parity, parity_backend
from .concat_k2_views import build_view
from .inputs import build_inputs
from .reader import DatasetReader, Jet


def parity_sample(spec):
    """First four registered train rows, with authenticated original K2 maps."""
    foundation = spec["foundation"]
    from .concat_k2_pilot import is_pilot
    splits = foundation.get("splits")
    tasks = foundation["assignment_tasks"]
    if is_pilot(spec):
        from .concat_k2_pilot_data import population, selected_tasks, map_rows
        splits = population(spec)
        tasks = selected_tasks(spec, splits)
    columns = {c: dict(features=[], vectors=[], offsets=[0]) for c in ("D100", "D000")}
    identities, labels, files = [], [], []
    for task in tasks:
        if task["role"] != "train" or task["rows"] == 0:
            continue
        _, arrays = load_assignment(spec, task["file_index"])
        lookup = map_rows(foundation,splits,task) if is_pilot(spec) else np.arange(task["rows"])
        reader = iter(DatasetReader(Path(spec["data_root"]), foundation["inventory"],
            splits, role="train", include_offline=True,
            file_paths=(task["path"],), step_size=64))
        try:
            for row, jet in enumerate(islice(reader, 4-len(labels))):
                row = lookup[row]
                identity = np.frombuffer(bytes.fromhex(jet.identity), np.uint8)
                start, end = arrays["offsets"][row:row+2]
                if (not np.array_equal(identity, arrays["identities"][row])
                        or len(jet.offline) != arrays["offline_counts"][row]
                        or end-start != len(jet.hlt)):
                    raise ValueError("Early K2 parity map/native identity join differs")
                mapping = arrays["mapping"][start:end]
                for coordinate, values in columns.items():
                    # Exercise the deployable endpoint with no offline capability.
                    source = jet if coordinate == "D100" else Jet(jet.identity, jet.label, jet.hlt, None)
                    view = build_view(source, coordinate, foundation["candidate"], mapping)
                    inputs = build_inputs(view, capacity=foundation["inputs"]["capacity"])
                    values["features"].append(inputs.features[:, :len(view)].T.copy())
                    values["vectors"].append(inputs.vectors[:, :len(view)].T.copy())
                    values["offsets"].append(values["offsets"][-1]+len(view))
                identities.append(identity)
                labels.append(jet.label)
                files.append(task["file_index"])
        finally:
            reader.close()
        if len(labels) == 4:
            break
    if len(labels) != 4:
        raise ValueError("Early K2 parity needs four distinct registered training rows")
    identities = np.asarray(identities, np.uint8)
    if len(np.unique(identities, axis=0)) != 4:
        raise ValueError("Early K2 parity identities are not unique")
    caches = {c: RamCache([RamBlock(0, np.asarray(v["offsets"], np.int64),
        np.concatenate(v["features"]), np.concatenate(v["vectors"]), identities,
        np.asarray(labels, np.int64))], role="train", foundation_sha256=foundation["content_hash"],
        coordinate_name=c) for c, v in columns.items()}
    return caches, dict(role="train", rows=4, file_indices=files,
        identities=[bytes(row).hex() for row in identities], final_test_accessed=False)


def early_parity(spec, directory, device):
    # Delayed import avoids a cycle and shares the existing cleanup policy.
    from .concat_k2_runtime import _cuda_clear
    print("JC2-K2 phase=early_parity_sample rows=4 role=train before_full_cache=True", flush=True)
    caches, sample = parity_sample(spec)
    reports = []
    for coordinate in ("D100", "D000"):
        cache = caches[coordinate]
        print(f"JC2-K2 phase=early_native_parity view={coordinate}", flush=True)
        with parity_backend(device):
            native = installed_parity(cache, device=device)
        _cuda_clear()
        for bf16 in (False, True):
            precision = "bf16" if bf16 else "fp32"
            print(f"JC2-K2 phase=early_storage_parity view={coordinate} precision={precision}", flush=True)
            result = storage_parity(cache.batch(np.arange(4)), device=device, bf16=bf16)
            report = artifact("EARLY_PARITY", campaign_sha256=spec["content_hash"],
                node_id="CONCAT_K2_"+coordinate, sample=sample, native_parity=native,
                storage_parity=result, acceptance_only=True, final_test_accessed=False)
            write_immutable_json(Path(directory)/f"early_parity_{coordinate}_{precision}.json", report)
            reports.append(report)
            _cuda_clear()
    print("JC2-K2 phase=early_parity_complete passed=True full_cache_next=True", flush=True)
    return reports
