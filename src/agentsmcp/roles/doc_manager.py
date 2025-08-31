from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from typing import Iterable


def _retention_count() -> int:
    try:
        return max(1, int(os.getenv("AGENTS_ROLES_RETENTION", "20")))
    except Exception:
        return 20


def _write_versioned_doc(base_dir: Path, name: str, content: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = base_dir / f"{ts}.md"
    path.write_text(content, encoding="utf-8")
    latest = base_dir / "latest.md"
    latest.write_text(content, encoding="utf-8")
    versions = sorted([p for p in base_dir.glob("*.md") if p.name != "latest.md"], reverse=True)
    keep = _retention_count()
    for old in versions[keep:]:
        try:
            old.unlink()
        except Exception:
            pass
    return path


def update_role_doc(role_name: str, default_prompt: str, responsibilities: Iterable[str], improvements: str = "") -> Path:
    base = Path("docs/roles") / role_name
    lines = [
        f"# Role: {role_name}",
        "",
        "## Responsibilities",
    ]
    for r in responsibilities:
        lines.append(f"- {r}")
    lines += [
        "",
        "## Default Prompt",
        "",
        default_prompt,
    ]
    if improvements and improvements.strip():
        lines += ["", "## Recent Improvements", "", improvements.strip()]
    content = "\n".join(lines) + "\n"
    return _write_versioned_doc(base, role_name, content)


def update_team_instructions(improvements: str) -> Path:
    base = Path("docs/roles") / "team_instructions"
    content = "\n".join([
        "# Team Instructions",
        "",
        "These instructions are continuously improved after each task.",
        "",
        "## Recent Improvements",
        "",
        improvements.strip(),
        "",
    ])
    return _write_versioned_doc(base, "team_instructions", content)

