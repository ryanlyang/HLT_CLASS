"""Exact-stage restart-zero recovery. Never cancels or modifies any job."""
from pathlib import Path
from .contracts import artifact, write_immutable_json
from .campaign import completed, validate_stage
from .submission import previous_attempts_terminal, command_plan


def prepare(stage, study, attempt_root):
    validate_stage(stage, study)
    previous_attempts_terminal(stage)
    root = Path(attempt_root).resolve()
    if root.exists():
        raise FileExistsError("Recovery needs a new attempt directory")
    remaining = [t["task_id"] for t in stage["tasks"] if completed(stage, t["task_id"]) is None]
    plan = command_plan(stage, study, root, selected=remaining)
    record = artifact("RECOVERY", stage_sha256=stage["content_hash"], command_plan_sha256=plan["content_hash"],
                      remaining=remaining, restart_from_zero=True, jobs_cancelled=[],
                      partial_outputs_preserved=True, optimizer_resume=False)
    write_immutable_json(root / "recovery.json", record)
    write_immutable_json(root / "command_plan.json", plan)
    return record
