"""Source-local assignments and bounded, process-parallel CMS RAM caches."""
from pathlib import Path
import hashlib
import time
import numpy as np

from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json, canonical_sha256
from hlt_classification.cms_salience_learned.data import raw_chunks, native_path, bounded_map, Cache as NativeCache
from hlt_classification.cms_salience_learned.storage import publish_npz, arrays_from, fingerprint
from hlt_classification.scouting.selective_assignment import build_row_selection, validate_row_selection
from hlt_classification.scouting.identity import ScoutingJetIdentity
from hlt_classification.cms_proxy_ladder.views import match_particles
from hlt_classification.scouting.hcwdl_fullcard_salience_matcher import validate_pairing
from .contracts import artifact, validate, CAPACITY
from .campaign import validate_spec, require
from .inputs import endpoints, view, features, audit


class Cache(NativeCache):
    """Same safe model keys, with the JC2 minimum-padding convention."""
    def _view(self, name, ix):
        f, v, m = self.views[name]
        width = max(16, int(m[ix, 0].sum(1).max()))
        return dict(features=np.ascontiguousarray(f[ix, :, :width]),
                    vectors=np.ascontiguousarray(v[ix, :, :width]),
                    mask=np.ascontiguousarray(m[ix, :, :width]), labels=self.labels[ix])

    def subset(self, ix):
        return Cache({name: tuple(np.ascontiguousarray(a[ix]) for a in arrays) for name, arrays in self.views.items()},
                     self.labels[ix], self.identities[ix], self.role, self.foundation_sha256, self.primary)


def path(spec, name):
    return Path(spec["campaign_root"]) / "foundation" / name


def selection(spec):
    value = load_json(path(spec, "selection.json"))
    validate_row_selection(value, split_manifest_sha256=spec["parents"]["split"])
    if (set(value["roles"]) != {"train", "validation"} or value["seed"] != 1337
        or {r: x["rows"] for r, x in value["roles"].items()} != spec["scientific"]["budgets"]):
        raise ValueError("CMS selection budget/roles differ")
    return value


def select(spec):
    split = validate_spec(spec)
    for row in spec["sources"]:
        if sha256_file(native_path(spec, row)) != row["sha256"]:
            raise ValueError("CMS raw source checksum differs")
        print(f"CMS-FEATURE authenticated {row['role']} {row['path']}", flush=True)
    value = build_row_selection(split, data_root=spec["data_root"], role_budgets=spec["scientific"]["budgets"], seed=1337)
    destination = path(spec, "selection.json")
    write_immutable_json(destination, value)
    return [destination]


def entries_for(spec, source):
    pop = selection(spec)
    entries = next(r["entries"] for r in pop["roles"][source["role"]]["sources"] if r["path"] == source["path"])
    return np.asarray(entries, np.int64)


def identities(source, entries):
    return [hashlib.sha256(ScoutingJetIdentity(source["path"], int(e)).key.encode()).hexdigest() for e in entries]


def _match(raw):
    maps, lengths, counts = [], [], {side: {} for side in ("hlt", "offline")}
    for h, o in endpoints(raw):
        mapping = match_particles(h.particles, o.particles)
        for side, value in (("hlt", h), ("offline", o)):
            for k, v in audit(value).items():
                counts[side][k] = counts[side].get(k, 0) + v
        lengths.append((len(h.particles), len(o.particles)))
        maps.append(mapping)
        # U000 is the support envelope for every registered view.
        if max(len(h.particles), len(o.particles)) > CAPACITY:
            raise ValueError("CMS support exceeds capacity; no truncation allowed")
    return maps, lengths, counts


def match_source(spec, index):
    source = spec["sources"][index]
    entries = entries_for(spec, source)
    if sha256_file(native_path(spec, source)) != source["sha256"]:
        raise ValueError("CMS source changed before matching")
    mappings, offsets, sizes, labels = [], [0], [], []
    counts = {side: {} for side in ("hlt", "offline")}
    def arguments():
        for _, _, target, raw in raw_chunks(spec, source, entries, size=128):
            labels.extend(target.tolist())
            yield raw
    for maps, lens, local in bounded_map(_match, arguments(), spec["resources"]["workers"]):
        for m in maps:
            mappings.extend(m.tolist()); offsets.append(len(mappings))
        sizes.extend(lens)
        for side in counts:
            for k, v in local[side].items():
                counts[side][k] = counts[side].get(k, 0) + v
        print(f"CMS-FEATURE match source={index} jets={len(sizes)}/{len(entries)}", flush=True)
    destination = path(spec, f"match_{index:04d}.json")
    payload = publish_npz(destination.with_suffix(".npz"), entries=entries, labels=np.asarray(labels, np.int64),
        offsets=np.asarray(offsets, np.int64), mapping=np.asarray(mappings, np.int32), sizes=np.asarray(sizes, np.int32).reshape(-1, 2))
    result = artifact("ASSIGNMENT", parents={"spec": spec["content_hash"], "selection": selection(spec)["content_hash"]},
        source=source, payload=payload, diagnostic=counts, rows=len(entries))
    write_immutable_json(destination, result)
    return [destination, destination.with_suffix(".npz")]


def assignment(spec, index):
    source = spec["sources"][index]
    value = load_json(path(spec, f"match_{index:04d}.json"))
    validate(value, "ASSIGNMENT", parents={"spec": spec["content_hash"], "selection": selection(spec)["content_hash"]})
    data = arrays_from(value["payload"])
    entries = entries_for(spec, source)
    n = len(entries)
    if (value["source"] != source or value["rows"] != n or not np.array_equal(entries, data["entries"])
        or data["labels"].shape != (n,) or not np.isin(data["labels"], range(15)).all()
        or data["sizes"].shape != (n, 2) or data["offsets"].shape != (n + 1,)
        or data["offsets"][0] != 0 or data["offsets"][-1] != len(data["mapping"])
        or np.any(np.diff(data["offsets"]) != data["sizes"][:, 0])):
        raise ValueError("Assignment population/layout differs")
    for i, (nh, no) in enumerate(data["sizes"]):
        validate_pairing(data["mapping"][data["offsets"][i]:data["offsets"][i + 1]], nh=int(nh), no=int(no))
    return value, data


def foundation(spec):
    records, role_ids, counts = [], {r: [] for r in ("train", "validation")}, {r: {} for r in ("hlt", "offline")}
    for i, source in enumerate(spec["sources"]):
        require(spec, f"match_{i:04d}")
        value, data = assignment(spec, i)
        records.append(fingerprint(path(spec, f"match_{i:04d}.json")))
        role_ids[source["role"]].extend(identities(source, data["entries"]))
        for side in counts:
            for k, v in value["diagnostic"][side].items():
                counts[side][k] = counts[side].get(k, 0) + v
    for role, ids in role_ids.items():
        if len(ids) != spec["scientific"]["budgets"][role] or len(set(ids)) != len(ids):
            raise ValueError("Foundation role population differs")
    if set(role_ids["train"]) & set(role_ids["validation"]):
        raise ValueError("Foundation train/validation overlap")
    value = artifact("FOUNDATION", parents={"spec": spec["content_hash"], "selection": selection(spec)["content_hash"]},
        assignments=records, role_identity_sha256={r: canonical_sha256(v) for r, v in role_ids.items()},
        diagnostic=counts, budgets=spec["scientific"]["budgets"])
    destination = path(spec, "foundation.json")
    write_immutable_json(destination, value)
    print("CMS-FEATURE foundation diagnostics:", counts, flush=True)
    return [destination]


def _view_chunk(args):
    raw, maps, keys, coordinate, arm = args
    result = []
    for (hlt, offline), mapping, key in zip(endpoints(raw), maps, keys, strict=True):
        value = view(identity=key, hlt=hlt, offline=offline, coordinate=coordinate, mapping=mapping)
        result.append(features(value, arm))
    return result


def build_cache(spec, role, coordinate, arm):
    if role not in ("train", "validation") or arm not in spec["scientific"]["arms"]:
        raise PermissionError("Unregistered feature arm or sealed role")
    require(spec, "foundation")
    lock = load_json(path(spec, "foundation.json"))
    validate(lock, "FOUNDATION", parents={"spec": spec["content_hash"], "selection": selection(spec)["content_hash"]})
    n = spec["scientific"]["budgets"][role]
    dimensions = 17 if arm == "SHARED17" else 21
    # One active view per job; batch padding is cropped, not constituent trimming.
    f = np.zeros((n, dimensions, CAPACITY), np.float32)
    v = np.zeros((n, 4, CAPACITY), np.float32)
    m = np.zeros((n, 1, CAPACITY), bool)
    labels, all_keys = [], []
    started = time.monotonic()
    def arguments():
        for i, source in enumerate(spec["sources"]):
            if source["role"] != role:
                continue
            if fingerprint(path(spec, f"match_{i:04d}.json")) != lock["assignments"][i]:
                raise ValueError("Foundation assignment changed")
            _, data = assignment(spec, i)
            if sha256_file(native_path(spec, source)) != source["sha256"]:
                raise ValueError("CMS raw file changed")
            for start, entries, target, raw in raw_chunks(spec, source, data["entries"], size=128):
                if not np.array_equal(target, data["labels"][start:start + len(entries)]):
                    raise ValueError("CMS cache label join differs")
                keys = identities(source, entries)
                labels.extend(target.tolist()); all_keys.extend(keys)
                maps = [data["mapping"][data["offsets"][j]:data["offsets"][j + 1]] for j in range(start, start + len(entries))]
                yield raw, maps, keys, coordinate, arm
    cursor = 0
    for chunk in bounded_map(_view_chunk, arguments(), spec["resources"]["workers"]):
        for fx, vx in chunk:
            length = len(fx)
            f[cursor, :, :length], v[cursor, :, :length], m[cursor, 0, :length] = fx.T, vx.T, True
            cursor += 1
        if cursor % 4096 == 0:
            print(f"CMS-FEATURE cache {arm} {coordinate} {role} {cursor}/{n}", flush=True)
    if cursor != n or canonical_sha256(all_keys) != lock["role_identity_sha256"][role]:
        raise ValueError("CMS cache ordered identities differ")
    ids = np.frombuffer(b"".join(bytes.fromhex(k) for k in all_keys), np.uint8).reshape(-1, 32).copy()
    print(f"CMS-FEATURE cache ready {role} {arm} {coordinate} seconds={time.monotonic()-started:.2f}", flush=True)
    return Cache({coordinate: (f, v, m)}, np.asarray(labels, np.int64), ids, role, lock["content_hash"], coordinate)
