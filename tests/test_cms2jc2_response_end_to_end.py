"""Actual ROOT/bridge/calibration/evaluation/transfer integration on tiny fixtures.

Only capacity minima and the synthetic JC2 metadata are substituted. These tests
cannot produce, or stand in for, a production execution lock.
"""
from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
import uproot

from hlt_classification.cms2jc2_response.audit import cms_particle_branches
from hlt_classification.cms2jc2_response.association import policy
from hlt_classification.cms2jc2_response.assumptions import provisional_compatibility
from hlt_classification.cms2jc2_response.bridge import JC2_FIELDS
from hlt_classification.cms2jc2_response.contracts import artifact, candidates, sha256_file
from hlt_classification.cms2jc2_response.evaluation import fit_metric_lock, evaluate
from hlt_classification.cms2jc2_response.readers import iter_cms, iter_jc2_offline
from hlt_classification.cms2jc2_response.response import collect, fit_response
from hlt_classification.cms2jc2_response.selection import family_finalists, select
from hlt_classification.cms2jc2_response import splits
from hlt_classification.cms2jc2_response.transfer import evaluate_transfer
from test_cms2jc2_response import particles


def cms_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, "MINIMA", (1, 1, 1))
    rows = []
    branches = cms_particle_branches("offline")+cms_particle_branches("hlt")
    for file in range(8):
        path = tmp_path/f"cms{file}.root"
        columns = {name: [] for name in branches}
        for i in range(6):
            p = particles((10.+file+.1*i, 20.+file+.2*i), (0., .3), (0., .2))
            for prefix in ("cpfcandlt", "npfcand", "scoutpfcand"):
                n = 0 if prefix == "npfcand" else 2
                vector = p.p4[:n]*(.9 if prefix == "scoutpfcand" else 1.)
                for name in branches:
                    if not name.startswith(prefix+"_"):
                        continue
                    field = name[len(prefix)+1:]
                    if field in ("px", "py", "pz", "energy"):
                        value = vector[:, ("px", "py", "pz", "energy").index(field)]
                    elif field in ("charge", "isChargedHad"):
                        value = np.ones(n)
                    elif field in ("dxy", "dz"):
                        value = np.full(n, .01+.001*i)
                    elif field in ("dxysig", "dzsig"):
                        value = np.full(n, 2.)
                    else:
                        value = np.zeros(n)
                    columns[name].append(value)
        with uproot.recreate(path) as handle:
            tree = handle.mktree("tree", {**{k: "var * float64" for k in branches},
                                          "scoutfj_label": "int32", "scoutfj_gen_pid": "int32", "fj_label": "int32"})
            tree.extend({**{k: ak.Array(v) for k, v in columns.items()},
                         "scoutfj_label": np.zeros(6, np.int32), "scoutfj_gen_pid": np.zeros(6, np.int32),
                         "fj_label": np.full(6, 309, np.int32)})
        rows.append(dict(path=path.name, sha256=sha256_file(path), tree_key="tree;1", raw_entries=6,
                         selected_entries=6, entry_mask=splits.pack_entries(np.arange(6), 6),
                         source="source"+str(file % 2), original_role="train"))
    inventory = artifact("CMS_INVENTORY", parents={"historical_split": "a"*64}, files=rows, selected_entries=48)
    roles = splits.build_roles(inventory)
    review = provisional_compatibility(inventory_hash=inventory["content_hash"])
    return inventory, roles, review


def test_root_to_33_fits_selection_claimed_confirmation_and_offline_transfer(tmp_path, monkeypatch):
    inventory, roles, review = cms_fixture(tmp_path, monkeypatch)
    def reader(role, **kwargs):
        return iter_cms(tmp_path, inventory, roles, review, role=role, **kwargs)
    loc = list(reader("response_fit", fit_role="fit_location"))
    res = list(reader("response_fit", fit_role="fit_residual"))
    assert not {p.source_group for p in loc} & {p.source_group for p in res}
    assert all(p.diagnostic_class is None for p in loc+res)
    lock = fit_metric_lock(loc, policy(), membership_hash=roles["content_hash"], bins=32, cap=32)
    primaries, fitted = [], {}
    for candidate in candidates()["candidates"]:
        for budget, count in (("250K", 4), ("1M", 8), ("FULL", len(loc))):
            # Tiny nested memberships test scientific plumbing only; production
            # exact-budget validators are tested separately and are not loosened.
            l, lr = collect(loc[:count], policy(), cap=720)
            r, rr = collect(res[:count], policy(), cap=720)
            response = fit_response(l, r, location_report=lr, residual_report=rr, candidate_id=candidate["id"],
                                    review=review, rules=policy(), budget=budget, source_hash="b"*64)
            report = evaluate(reader("response_select", diagnostic_labels=True), response, lock,
                              membership_hash=roles["content_hash"], role="response_select")
            assert report["labelled_diagnostic_jets"] == report["comparison"]["real_jets"]
            primaries.append(report["comparison"])
            if budget == "FULL":
                fitted[candidate["id"]] = response
    finalists, sensitivities = family_finalists(primaries), []
    for candidate in finalists.values():
        for gate in (.05, .2):
            rules = policy(gate=gate)
            l, lr = collect(loc, rules, cap=720); r, rr = collect(res, rules, cap=720)
            response = fit_response(l, r, location_report=lr, residual_report=rr, candidate_id=candidate,
                                    review=review, rules=rules, budget="FULL", source_hash="b"*64)
            sensitivities.append(evaluate(reader("response_select"), response, lock,
                membership_hash=roles["content_hash"], role="response_select")["comparison"])
    selected = select(primaries, sensitivities, registry_hash=candidates()["content_hash"])
    assert len(primaries) == 27 and len(sensitivities) == 6
    response = fitted[selected["selected_candidate"]]
    assert response["content_hash"] == selected["response_hash"]
    with pytest.raises(PermissionError, match="claim"):
        list(reader("response_confirm"))
    claim = artifact("CONFIRMATION_CLAIM", parents={"roles": roles["content_hash"],
                     "selection": selected["content_hash"], "compatibility": review["content_hash"]})
    confirmed = evaluate(reader("response_confirm", confirmation_claim=claim, selection=selected), response, lock,
                         membership_hash=roles["content_hash"], role="response_confirm")
    assert confirmed["comparison"]["replicas"] == [0, 1, 2, 3, 4]
    assert selected["response_hash"] == response["content_hash"]
    assert not confirmed["nonselecting_diagnostics"]["two_sample"]["available"]

    # Real offline-only ROOT file deliberately has no labels or native-HLT fields.
    p = particles()
    columns = dict(zip(("px", "py", "pz", "energy"), p.p4.T))
    columns.update(charge=np.ones(2), isChargedHadron=np.ones(2), isNeutralHadron=np.zeros(2),
                   isPhoton=np.zeros(2), isElectron=np.zeros(2), isMuon=np.zeros(2),
                   d0val=np.zeros(2), dzval=np.zeros(2), d0err=np.zeros(2), dzerr=np.zeros(2))
    path = tmp_path/"offline.root"
    with uproot.recreate(path) as handle:
        tree = handle.mktree("tree", {"jet_nparticles": "int32", **{"part_"+f: "var * float64" for f in JC2_FIELDS}})
        tree.extend({"jet_nparticles": np.array([2], np.int32), **{"part_"+f: ak.Array([columns[f]]) for f in JC2_FIELDS}})
    from hlt_classification.jetclass2_delphes import inventory as jc_inv, splits as jc_splits
    from hlt_classification.jetclass2_delphes.split_registry import pack_entries
    jc2 = {"files": [dict(path=path.name, tree_key="tree;1", entries=1, sha256=sha256_file(path))]}
    profile = {"profile": "TRAIN_500K", "groups": [dict(path=path.name, role="train")],
               "memberships": {"train": {"files": [dict(path=path.name, entry_mask=pack_entries([0], 1))]}}}
    monkeypatch.setattr(jc_inv, "validate_inventory", lambda value: "d"*64)
    monkeypatch.setattr(jc_inv, "verify_file", lambda *args: path)
    monkeypatch.setattr(jc_splits, "validate_splits", lambda *args: None)
    monkeypatch.setattr(jc_splits, "is_subset_profile", lambda *args: True)
    transfer_claim = artifact("TRANSFER_CLAIM", parents={"selection": selected["content_hash"],
                             "response": response["content_hash"], "profile": "e"*64}, role="train", separately_authorized=True)
    result = evaluate_transfer(iter_jc2_offline(tmp_path, jc2, profile, review, role="train"), response,
                               selection=selected, claim=transfer_claim, profile_hash="e"*64, role="train")
    assert result["counts"]["jets"] == 1 and result["scientific_completion"]
    assert not result["native_jc2_hlt_accessed"] and not result["physically_qualified"]


def test_probe_is_hash_selected_and_cannot_read_held_out_roles(tmp_path, monkeypatch):
    inventory, roles, review = cms_fixture(tmp_path, monkeypatch)
    args = dict(root=tmp_path, inventory=inventory, roles=roles, review=review, role="response_fit", fit_role="fit_location")
    full = list(iter_cms(**args))
    first = list(iter_cms(**args, fit_probe_limit=3, chunk_rows=1))
    second = list(iter_cms(**args, fit_probe_limit=3, chunk_rows=4))
    assert [p.identity for p in first] == [p.identity for p in second]
    assert len(first)*2 == len(full)
    with pytest.raises(PermissionError, match="fit-only|fit-only|internal-fit-only"):
        list(iter_cms(tmp_path, inventory, roles, review, role="response_select", fit_probe_limit=1))
