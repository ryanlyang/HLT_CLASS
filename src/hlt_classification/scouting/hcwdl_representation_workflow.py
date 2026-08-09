"""Thin immutable-task dispatcher for HCWDL-RKD campaign workers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import re
from typing import Any

from .hcwdl_representation_campaign import (
    DETERMINISTIC_KINDS, CampaignTask, validate_campaign_spec,
)


ARRAY_PATTERN = re.compile(r"^(\d+)-(\d+)$")


def array_indices(specification: str | None) -> tuple[int | None, ...]:
    if specification is None:
        return (None,)
    match = ARRAY_PATTERN.fullmatch(specification)
    if match is None:
        raise ValueError("representation task array syntax differs")
    start, stop = map(int, match.groups())
    if start < 0 or stop < start:
        raise ValueError("representation task array range differs")
    return tuple(range(start, stop + 1))


class RepresentationWorkflow:
    """Validate a task row and delegate it without adding scientific defaults."""

    def __init__(
        self,
        spec: Mapping[str, Any],
        *,
        handlers: Mapping[str, Callable[[Mapping[str, Any], CampaignTask, int | None], Any]],
        executable: bool = False,
    ) -> None:
        validate_campaign_spec(spec, executable=executable)
        self.spec = dict(spec)
        self.handlers = dict(handlers)
        self.tasks = {
            row["task_key"]: CampaignTask(
                **{
                    **row,
                    "dependencies": tuple(row["dependencies"]),
                    "registered_inputs": tuple(row["registered_inputs"]),
                    "registered_outputs": tuple(row["registered_outputs"]),
                }
            )
            for row in spec["tasks"]
        }

    def run(
        self, task_key: str, *, array_index: int | None = None,
        deterministic_worker: bool = False,
    ) -> Any:
        task = self.tasks.get(task_key)
        if task is None:
            raise KeyError(f"unregistered representation task {task_key!r}")
        allowed = array_indices(task.array)
        if array_index not in allowed:
            raise IndexError("representation array index is not registered")
        expected_deterministic = task.kind in DETERMINISTIC_KINDS
        if deterministic_worker != expected_deterministic:
            raise PermissionError("representation task was routed through the wrong worker")
        if task.deterministic_worker != expected_deterministic:
            raise ValueError("representation task worker declaration differs")
        handler = self.handlers.get(task.kind)
        if handler is None:
            raise KeyError(f"no representation workflow handler for {task.kind!r}")
        return handler(self.spec, task, array_index)


def exercise_registered_rows(
    spec: Mapping[str, Any], *,
    handlers: Mapping[str, Callable[[Mapping[str, Any], CampaignTask, int | None], Any]],
) -> list[dict[str, Any]]:
    workflow = RepresentationWorkflow(spec, handlers=handlers)
    rows = []
    for task in workflow.tasks.values():
        for index in array_indices(task.array):
            result = workflow.run(
                task.task_key,
                array_index=index,
                deterministic_worker=task.deterministic_worker,
            )
            rows.append(
                {"task_key": task.task_key, "array_index": index, "result": result}
            )
    return rows


__all__ = ["RepresentationWorkflow", "array_indices", "exercise_registered_rows"]
