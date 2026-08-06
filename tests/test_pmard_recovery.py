from __future__ import annotations

import pytest

from hlt_classification.scouting.recovery import (
    validate_argv_string_normalization, validate_assignment_root_resolution,
    validate_selective_identity_scope,
)


OLD = """
class Workflow:
    def _teacher_command(self):
        command = ["python", 0.5]
        return command

    def _student_command(self):
        command = ["python", 0.5]
        return command
"""

NEW = """
class Workflow:
    def _teacher_command(self):
        command = ["python", 0.5]
        return [str(value) for value in command]

    def _student_command(self):
        command = ["python", 0.5]
        return [str(value) for value in command]
"""


def test_prefix_recovery_accepts_only_complete_argv_string_normalization():
    validate_argv_string_normalization(OLD, NEW)
    with pytest.raises(ValueError, match="beyond argv string normalization"):
        validate_argv_string_normalization(
            OLD,
            NEW.replace('command = ["python", 0.5]', 'command = ["python", 0.25]', 1),
        )
    with pytest.raises(ValueError, match="not complete argv string normalization"):
        validate_argv_string_normalization(
            OLD,
            NEW.replace(
                "return [str(value) for value in command]",
                "return command",
                1,
            ),
        )


OLD_ASSIGNMENT = '''
from pathlib import Path

class PersistentAssignmentStore:
    def __init__(self, manifest_path):
        self.path = Path(manifest_path); self.root = self.path.parent
'''

NEW_ASSIGNMENT = '''
from pathlib import Path

def _assignment_root_for_manifest(path: Path) -> Path:
    """Resolve the canonical shard root for a workflow manifest location."""
    if path.name == "assignment_manifest.json":
        return path.parent / "assignments"
    if path.name == "final_assignment_manifest.json":
        return path.parent / "final_assignments"
    return path.parent

class PersistentAssignmentStore:
    def __init__(self, manifest_path):
        self.path = Path(manifest_path); self.root = _assignment_root_for_manifest(self.path)
'''


def test_prefix_recovery_accepts_only_canonical_assignment_root_resolution():
    validate_assignment_root_resolution(OLD_ASSIGNMENT, NEW_ASSIGNMENT)
    with pytest.raises(ValueError, match="resolver differs"):
        validate_assignment_root_resolution(
            OLD_ASSIGNMENT,
            NEW_ASSIGNMENT.replace('return path.parent / "assignments"', 'return path.parent / "wrong"'),
        )
    with pytest.raises(ValueError, match="beyond manifest root resolution"):
        validate_assignment_root_resolution(
            OLD_ASSIGNMENT,
            NEW_ASSIGNMENT.replace("class PersistentAssignmentStore:", "EXTRA = 1\n\nclass PersistentAssignmentStore:"),
        )


OLD_REPAIR = '''
def _hlt_charged_mask(raw: Mapping[str, Sequence[np.ndarray]], *, row: int, visible: int) -> np.ndarray:
    flags = np.stack([
        np.asarray(raw[HLT_FEATURE_SPECS[channel].branch][row][:visible], np.float64)
        for channel in range(2, 7)
    ], axis=1)
    if not (((flags == 0) | (flags == 1)).all() and np.all(flags.sum(axis=1) == 1)):
        raise ValueError(f"invalid HLT particle identity in row {row}")
    return np.argmax(flags, axis=1) < 3

def _apply_full_endpoint_repair():
    hlt_charged = _hlt_charged_mask(raw, row=row, visible=visible)[matched_tokens]
'''

NEW_REPAIR = '''
def _hlt_charged_mask(
    raw: Mapping[str, Sequence[np.ndarray]], *, row: int, visible: int,
    tokens: np.ndarray,
) -> np.ndarray:
    flags = np.stack([
        np.asarray(raw[HLT_FEATURE_SPECS[channel].branch][row][:visible], np.float64)
        for channel in range(2, 7)
    ], axis=1)[tokens]
    if not (((flags == 0) | (flags == 1)).all() and np.all(flags.sum(axis=1) == 1)):
        raise ValueError(f"invalid matched HLT particle identity in row {row}")
    return np.argmax(flags, axis=1) < 3

def _apply_full_endpoint_repair():
    hlt_charged = _hlt_charged_mask(
        raw, row=row, visible=visible, tokens=matched_tokens,
    )
'''


def test_prefix_recovery_accepts_only_matched_token_identity_scope():
    validate_selective_identity_scope(OLD_REPAIR, NEW_REPAIR)
    with pytest.raises(ValueError, match="authorized correction"):
        validate_selective_identity_scope(
            OLD_REPAIR,
            NEW_REPAIR.replace("flags = np.stack", "flags = 2 * np.stack"),
        )
    with pytest.raises(ValueError, match="beyond matched-token identity validation"):
        validate_selective_identity_scope(
            OLD_REPAIR,
            NEW_REPAIR + "\nEXTRA = 1\n",
        )
