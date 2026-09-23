"""Bounded real-training parity before expensive full-population fusion caches.

Adapted from concat_k2_parity at 91be019, using this campaign's salience maps
and view builder, not K2 inputs. This is diagnostic evidence, never a GPU gate
substitute or a scientific training artifact.
"""
from itertools import islice
from pathlib import Path

import numpy as np

from hlt_classification.data.cache_contracts import write_immutable_json
from .cache import RamBlock, RamCache
from .dzfix_fusion_chain import artifact
from .dzfix_fusion_model import native_mask_parity, native_offload_parity
from .inputs import build_inputs
from .reader import DatasetReader, Jet
from .salience_foundation import load_assignments
from .salience_learned_graph import coordinate
from .salience_views import build_view


EARLY_PAIRS = (("U050", "U000"), ("D000", "D000"))


def parity_sample(spec):
    """Four distinct registered train jets; read and authenticate existing maps."""
    foundation = spec["foundation"]
    columns = {c: dict(features=[], vectors=[], offsets=[0]) for c in ("U000", "U050", "D000")}
    identities, labels, files = [], [], []
    for task in foundation["assignment_tasks"]:
        if task["role"] != "train" or task["rows"] == 0:
            continue
        _, arrays = load_assignments(foundation,
            root=Path(spec["source_import"]["foundation_root"]), file_index=task["file_index"])
        reader = iter(DatasetReader(Path(spec["data_root"]), foundation["inventory"],
            foundation["splits"], role="train", include_offline=True,
            file_paths=(task["path"],), step_size=64))
        try:
            for row, jet in enumerate(islice(reader, 4-len(labels))):
                identity = np.frombuffer(bytes.fromhex(jet.identity), np.uint8)
                start, end = arrays["offsets"][row:row+2]
                if (not np.array_equal(identity, arrays["identities"][row])
                        or end-start != len(jet.hlt)):
                    raise ValueError("Early fusion parity map/native identity join differs")
                mapping = arrays["mapping"][int(start):int(end)]
                for name, values in columns.items():
                    # Explicitly remove offline capability for the HLT endpoint.
                    source = jet if name != "D000" else Jet(jet.identity, jet.label, jet.hlt, None)
                    u, f = coordinate(name)
                    view = build_view(source, u=u, f=f, candidate=foundation["candidate"],
                                      mapping=mapping if name != "D000" else None)
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
        raise ValueError("Early fusion parity needs four distinct registered training rows")
    identities = np.asarray(identities, np.uint8)
    if len(np.unique(identities, axis=0)) != 4:
        raise ValueError("Early fusion parity identities are not unique")
    caches = {c: RamCache([RamBlock(0, np.asarray(v["offsets"], np.int64),
        np.concatenate(v["features"]), np.concatenate(v["vectors"]), identities,
        np.asarray(labels, np.int64))], role="train", foundation_sha256=foundation["content_hash"],
        coordinate_name=c) for c, v in columns.items()}
    return caches, dict(role="train", rows=4, file_indices=files,
        identities=[bytes(row).hex() for row in identities], final_test_accessed=False)


def early_parity(spec, directory, device):
    from .dzfix_fusion_runtime import _cuda_clear
    print("JC2-FUSION phase=early_parity_sample rows=4 role=train before_full_cache=True", flush=True)
    caches, sample = parity_sample(spec)
    reports = []
    for primary, context in EARLY_PAIRS:
        raw = caches[primary].batch(np.arange(4))
        context_raw = caches[context].batch(np.arange(4))
        print(f"JC2-FUSION phase=early_native_parity primary={primary} context={context}", flush=True)
        mask_parity = native_mask_parity(raw, device=device)
        _cuda_clear()
        for bf16 in (False, True):
            precision = "bf16" if bf16 else "fp32"
            print(f"JC2-FUSION phase=early_storage_parity primary={primary} context={context} precision={precision}", flush=True)
            result = native_offload_parity(raw, context_raw, device=device, bf16=bf16)
            report = artifact("EARLY_PARITY", campaign_sha256=spec["content_hash"],
                primary=primary, context=context, sample=sample, compact_mask_native_parity=mask_parity,
                storage_parity=result, acceptance_only=True, final_test_accessed=False)
            write_immutable_json(Path(directory)/f"early_parity_{primary}_{context}_{precision}.json", report)
            reports.append(report)
            _cuda_clear()
    print("JC2-FUSION phase=early_parity_complete passed=True full_cache_next=True", flush=True)
    return reports
