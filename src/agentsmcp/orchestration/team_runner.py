import asyncio
from typing import Iterable, List, Dict

from ..agent_manager import AgentManager
from ..events import EventBus
from ..models import TaskEnvelopeV1
from ..runtime_config import Config


DEFAULT_TEAM: List[str] = [
    "business_analyst",
    "backend_engineer",
    "api_engineer",
    "web_frontend_engineer",
    "tui_frontend_engineer",
    "backend_qa_engineer",
    "web_frontend_qa_engineer",
    "tui_frontend_qa_engineer",
]


async def run_team(objective: str, roles: Iterable[str] | None = None, progress_callback=None) -> Dict[str, str]:
    """Run a team of role agents in parallel and return a mapping role->output.

    Uses the in-process SelfAgent path (AgentManager) and the shared provider caps.
    """
    cfg = Config.load()
    bus = EventBus()
    mgr = AgentManager(cfg, events=bus)

    results: Dict[str, str] = {}
    roles = list(roles or DEFAULT_TEAM)

    async def _run(role: str):
        task = TaskEnvelopeV1(objective=objective)
        job_id = await mgr.spawn_agent(role, task.objective)
        if progress_callback:
            try:
                await progress_callback("job.spawned", {"agent": role})
            except Exception:
                pass
        status = await mgr.wait_for_completion(job_id)
        results[role] = status.output or status.error or ""
        if progress_callback:
            try:
                await progress_callback("job.completed", {"agent": role})
            except Exception:
                pass

    await asyncio.gather(*[_run(r) for r in roles])

    # Review gate: if there are staged changes, run chief_qa_engineer to review and approve
    from pathlib import Path as _Path
    staged_root = _Path.cwd().resolve() / 'build' / 'staging'
    if staged_root.exists() and any(p.is_file() for p in staged_root.rglob('*')):
        review_objective = (
            "Review the staged changes and either approve or discard them. "
            "Use list_staged_changes, git_diff, and approve_changes(commit_message='feat: reviewed changes') if acceptable; "
            "otherwise call discard_staged_changes with a short explanation."
        )
        # Run chief QA explicitly with review instruction
        job_id = await mgr.spawn_agent('chief_qa_engineer', review_objective)
        status = await mgr.wait_for_completion(job_id)
        results['chief_qa_engineer'] = status.output or status.error or ''

    return results
