from typing import Dict, List, Optional

from ..models import JobStatus
from .base import BaseStorage


class MemoryStorage(BaseStorage):
    """In-memory storage backend for development and testing."""

    def __init__(self):
        self._jobs: Dict[str, JobStatus] = {}

    async def store_job_status(self, status: JobStatus) -> None:
        """Store a job status in memory."""
        self._jobs[status.job_id] = status

    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Retrieve a job status from memory."""
        return self._jobs.get(job_id)

    async def list_job_statuses(self, limit: int = 100) -> List[JobStatus]:
        """List job statuses from memory."""
        statuses = list(self._jobs.values())
        # Sort by created_at descending
        statuses.sort(key=lambda x: x.created_at, reverse=True)
        return statuses[:limit]

    async def delete_job_status(self, job_id: str) -> bool:
        """Delete a job status from memory."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False

    async def close(self) -> None:
        """Close the memory storage (no-op)."""
        self._jobs.clear()

    async def ping(self) -> bool:
        return True
