from __future__ import annotations

import pytest

from hlt_classification.scouting.recovery import validate_argv_string_normalization


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
