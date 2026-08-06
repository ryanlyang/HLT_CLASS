from __future__ import annotations

import pytest

from hlt_classification.scouting.recovery import (
    validate_argv_string_normalization, validate_assignment_root_resolution,
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
