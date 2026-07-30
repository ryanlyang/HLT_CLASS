from __future__ import annotations

import re
from pathlib import Path

import hlt_classification


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def test_package_import_reports_current_transfer_status() -> None:
    assert hlt_classification.__version__ == "0.1.0"
    assert hlt_classification.FOUNDATION_STATUS == {
        "transfer_block": 4,
        "implemented_out_of_order_blocks": (5,),
        "scientific_pipeline_implemented": False,
        "next_transfer_block": 5,
        "authoritative_weaver_parity_passed": False,
    }


def test_required_scaffold_exists() -> None:
    required = (
        "AGENTS.md",
        "README.md",
        "REPOSITORY_TRANSFER_PLAN.md",
        "pyproject.toml",
        "configs/tigris.yaml",
        "docs/HANDOFF.md",
        "docs/RESEARCH_COMPUTE_RUNBOOK.md",
        "docs/DATA_CONTRACT.md",
        "docs/EXPERIMENT_CONTRACT.md",
        "docs/TESTING.md",
        "docs/LEGACY_SOURCE_MAP.md",
        "docs/decisions/README.md",
        "docs/plans/README.md",
        "scripts/README.md",
        "sbatch/README.md",
    )
    missing = [relative for relative in required if not (ROOT / relative).is_file()]
    assert missing == []


def test_repository_relative_markdown_links_resolve() -> None:
    failures: list[str] = []
    for markdown in sorted(ROOT.rglob("*.md")):
        if any(part.startswith(".") for part in markdown.relative_to(ROOT).parts):
            continue
        text = markdown.read_text(encoding="utf-8")
        for target in MARKDOWN_LINK.findall(text):
            target = target.strip()
            if (
                not target
                or target.startswith(("http://", "https://", "mailto:", "#"))
            ):
                continue
            path_text = target.split("#", 1)[0].replace("\\", "/")
            if not path_text:
                continue
            resolved = (markdown.parent / path_text).resolve()
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                failures.append(f"{markdown.relative_to(ROOT)} -> {target} escapes root")
                continue
            if not resolved.exists():
                failures.append(f"{markdown.relative_to(ROOT)} -> {target} is missing")
    assert failures == []


def test_agent_document_names_active_task_and_tigris_rules() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    handoff = (ROOT / "docs" / "HANDOFF.md").read_text(encoding="utf-8")
    runbook = (ROOT / "docs" / "RESEARCH_COMPUTE_RUNBOOK.md").read_text(
        encoding="utf-8"
    )
    assert "Transfer Block 5" in handoff
    assert "reu-aisocial" in agents and "reu-aisocial" in runbook
    assert "atlas_kd_tigris" in agents and "atlas_kd_tigris" in runbook
    assert "PYTHONNOUSERSITE=1" in agents and "PYTHONNOUSERSITE=1" in runbook
    assert "Poor accuracy" in agents
