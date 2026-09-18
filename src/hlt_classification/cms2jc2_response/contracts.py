"""Source-bound contracts for the isolated CMS2JC2 response study."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping
import re

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, require_sha256, sha256_file,
    validate_content_hash, with_content_hash, write_immutable_json,
)

PREFIX = "CMS2JC2_RESPONSE_"
SEED = 20260917
OUTER_ROLES = ("response_fit", "response_select", "response_confirm")
FIT_ROLES = ("fit_location", "fit_residual")
BUDGETS = ("250K", "1M", "FULL")
MINIMA = (2_000_000, 250_000, 250_000)
QUANTILES = (.001, .005, .01, .025, .05, .1, .25, .5, .75, .9, .95, .975, .99, .995, .999)
PLAN = "docs/plans/CMS_CALIBRATED_JETCLASS2_HLT_RESPONSE_THREE_FAMILY_IMPLEMENTATION_PLAN.md"
DOCUMENTARY_KEYS = (
    "cms_p4_weighting", "jc2_p4_weighting", "momentum_units",
    "cms_dxy_units_sign_reference", "cms_dz_units_reference",
    "jc2_d0_units_sign_reference", "jc2_dz_units_reference",
    "cms_significance_definition", "jc2_uncertainty_definition",
    "pid_charge_categories", "cms_regular_pf_excludes_lost_tracks",
)


def artifact(kind: str, *, parents: Mapping[str, str] | None = None, **fields) -> dict:
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", kind):
        raise ValueError("Invalid response artifact kind")
    forbidden = {"contract", "schema_version", "content_hash", "final_test_accessed"}
    if forbidden.intersection(fields):
        raise ValueError("Artifact metadata cannot be overridden")
    for name, digest in (parents or {}).items():
        require_sha256(digest, name=name)
    return with_content_hash(dict(
        contract=f"{PREFIX}{kind}/v1", schema_version=1,
        parents=dict(parents or {}), final_test_accessed=False, **fields,
    ))


def validate(value: dict, kind: str, *, parents: Mapping[str, str] | None = None) -> str:
    digest = validate_content_hash(value, expected_contract=f"{PREFIX}{kind}/v1")
    if (type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or value.get("final_test_accessed") is not False
            or not isinstance(value.get("parents"), dict)):
        raise ValueError("Response role/parent registry differs")
    for name, parent in value["parents"].items():
        require_sha256(parent, name=name)
    if parents is not None and value["parents"] != dict(parents):
        raise ValueError("Response artifact lineage differs")
    return digest


def candidates() -> dict:
    rows = []
    for i, level in enumerate(("L", "M", "H")):
        rows.append(dict(id=f"A_{level}", family="table", pt_bins=(6, 12, 24)[i],
                         eta_bins=(3, 6, 8)[i], pseudocount=(2000, 1000, 500)[i]))
    for i, level in enumerate(("L", "M", "H")):
        rows.append(dict(id=f"B_{level}", family="smooth", interior_knots=(4, 6, 8)[i],
                         ridge=(10., 1., .1)[i]))
    for i, level in enumerate(("L", "M", "H")):
        rows.append(dict(id=f"C_{level}", family="tree", depth=(2, 3, 4)[i],
                         trees=(200, 400, 600)[i], min_effective_leaf=(5000, 2000, 1000)[i],
                         learning_rate=.05, l2=1., early_stopping=False))
    return artifact("CANDIDATES", candidates=rows, budgets=list(BUDGETS),
                    primary_fits=27, sensitivity_fits=6, seed=SEED,
                    selection_budget="FULL", pruning=False)


def safe_relative(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("Expected canonical POSIX relative path")
    if not relative or relative.startswith("/") or ":" in relative or any(
        part in {"", ".", ".."} for part in relative.split("/")
    ):
        raise ValueError("Unsafe source-relative path")
    target = (root / relative).resolve()
    target.relative_to(root.resolve())
    return target


def compatibility_template(*, inventory_hash: str) -> dict:
    return artifact("COMPATIBILITY_REVIEW", parents={"inventory": inventory_hash},
                    decisions={key: dict(status="unresolved", evidence=[], decision="")
                               for key in DOCUMENTARY_KEYS},
                    cms_length_to_mm=None, jc2_length_to_mm=None,
                    cms_d0_sign=None, jc2_d0_sign=None,
                    momentum_to_gev=None, reviewer=None)


def validate_compatibility(value: dict, *, inventory_hash: str | None = None) -> str:
    provisional = value.get("contract") == f"{PREFIX}PROVISIONAL_COMPATIBILITY/v1"
    digest = validate(value, "PROVISIONAL_COMPATIBILITY" if provisional else "COMPATIBILITY_REVIEW")
    if provisional:
        from .assumptions import validate_provisional
        validate_provisional(value)
    if inventory_hash is not None and value["parents"] != {"inventory": inventory_hash}:
        raise ValueError("Compatibility inventory differs")
    if set(value.get("decisions", {})) != set(DOCUMENTARY_KEYS):
        raise ValueError("Missing common-field decisions")
    unresolved = []
    for key, decision in value["decisions"].items():
        allowed_status = {"assumed"} if provisional else {"verified"}
        if decision.get("status") not in allowed_status or not decision.get("decision"):
            unresolved.append(key)
            continue
        evidence = decision.get("evidence", [])
        if not evidence:
            unresolved.append(key)
        for row in evidence:
            require_sha256(row.get("sha256"), name="producer evidence")
            if not row.get("source") or not row.get("locator"):
                raise ValueError("Producer evidence needs source and precise locator")
    if unresolved:
        raise PermissionError("Unresolved shared physical semantics: " + ", ".join(unresolved))
    if not isinstance(value.get("reviewer"), str) or not value["reviewer"].strip():
        raise PermissionError("Human compatibility review is required")
    for key in ("cms_length_to_mm", "jc2_length_to_mm", "momentum_to_gev"):
        number = value.get(key)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not 0 < number < 1e6:
            raise ValueError(f"Invalid documented conversion: {key}")
    for key in ("cms_d0_sign", "jc2_d0_sign"):
        if value.get(key) not in (-1, 1) or isinstance(value.get(key), bool):
            raise ValueError("Invalid documented displacement sign")
    return digest


def publish(path: Path, value: dict, kind: str) -> str:
    validate(value, kind)
    return write_immutable_json(path, value)


def read(path: Path, kind: str) -> dict:
    value = load_json(path)
    validate(value, kind)
    return value
