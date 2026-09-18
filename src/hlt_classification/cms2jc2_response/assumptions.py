"""User-authorized provisional physical interface, never a verified review.

Do not overwrite this protocol when producer information arrives. Publish a
new version and new downstream artifacts, retaining all old assumptions.
"""
from .contracts import DOCUMENTARY_KEYS, artifact, canonical_sha256, validate

AUTHORITY = (
    "2026-09-17 user authorization: proceed with explicit most-likely producer "
    "assumptions while awaiting Luka's reply; revise if contradicted. This is "
    "not permission to claim confirmed physical equivalence or submit jobs."
)
DECISIONS = {
    "cms_p4_weighting": "Preserve native regular-PF four-vectors; assume unweighted CMS writer, no export rescaling.",
    "jc2_p4_weighting": "Preserve stored PUPPI-processed four-vectors. Not asserted equivalent to unweighted CMS PF.",
    "momentum_units": "Both stored p4 collections use GeV, no additional numeric scale.",
    "cms_dxy_units_sign_reference": "Stored signed dxy in cm; canonical sign is CMS dxy. Retain producer vertex reference.",
    "cms_dz_units_reference": "Stored dz in cm; retain producer reference, do not invent an unavailable vertex correction.",
    "jc2_d0_units_sign_reference": "Delphes D0 in mm; multiply by -1 for CMS dxy sign. Retain beamline reference.",
    "jc2_dz_units_reference": "Delphes DZ in mm with exporter PV-z subtraction for nonzero values; no further correction.",
    "cms_significance_definition": "Assume signed value/error; recover abs(value/significance) only with both nonzero and finite.",
    "jc2_uncertainty_definition": "Positive errors are standard deviations in mm; zero means unavailable, not exact measurement.",
    "pid_charge_categories": "Five exclusive physical categories, ambiguous flags become explicit unknown; charge is -1,0,1.",
    "cms_regular_pf_excludes_lost_tracks": "Exclude cpfcandlt_isLostTrack entries; never use that flag as a response predictor.",
}
LIMITATIONS = [
    "Exact dataset producer revisions are unconfirmed.",
    "CMS unweighted PF versus JC2 PUPPI-processed offline inputs are a known domain mismatch, not undone by this bridge.",
    "CMS/Delphes vertex references are retained, not asserted identical; no missing event-level correction is fabricated.",
    "CMS scouting impact-parameter upstream definitions remain provisionally assumed.",
]
CONVERSIONS = dict(cms_length_to_mm=10., jc2_length_to_mm=1., momentum_to_gev=1.,
                   cms_d0_sign=1, jc2_d0_sign=-1)


def provisional_compatibility(*, inventory_hash: str) -> dict:
    evidence = dict(source="user_authorized_assumption_protocol", locator="assumptions.py:AUTHORITY",
                    sha256=canonical_sha256(AUTHORITY))
    return artifact("PROVISIONAL_COMPATIBILITY", parents={"inventory": inventory_hash},
                    assumption_protocol="CMS2JC2_NATIVE_PROVISIONAL/v1", authority=AUTHORITY,
                    reviewer="user-authorized provisional implementation; producer confirmation pending",
                    decisions={k: dict(status="assumed", decision=v, evidence=[evidence])
                               for k, v in DECISIONS.items()}, **CONVERSIONS,
                    limitations=LIMITATIONS, producer_confirmed=False,
                    physically_qualified_transfer_allowed=False,
                    stored_four_vectors_policy="preserve_native_no_invented_unweighting_v1")


def validate_provisional(value: dict) -> str:
    digest = validate(value, "PROVISIONAL_COMPATIBILITY")
    expected = provisional_compatibility(inventory_hash=value["parents"].get("inventory", ""))
    if value != expected or set(value["decisions"]) != set(DOCUMENTARY_KEYS):
        raise ValueError("Unregistered provisional physical assumptions; publish a new version")
    return digest


def physical_status(review: dict) -> dict:
    from .contracts import validate_compatibility
    validate_compatibility(review)
    provisional = review["contract"].endswith("PROVISIONAL_COMPATIBILITY/v1")
    return dict(producer_confirmed=not provisional, provisional=provisional,
                physically_qualified_transfer_allowed=not provisional,
                limitations=review.get("limitations", []))
