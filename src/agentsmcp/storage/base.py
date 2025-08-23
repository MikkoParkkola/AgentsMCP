from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from ..models import JobStatus


class BaseStorage(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    async def store_job_status(self, status: JobStatus) -> None:
        """Store a job status."""
        pass
    
    @abstractmethod
    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Retrieve a job status by ID."""
        pass
    
    @abstractmethod
    async def list_job_statuses(self, limit: int = 100) -> List[JobStatus]:
        """List job statuses with optional limit."""
        pass
    
    @abstractmethod
    async def delete_job_status(self, job_id: str) -> bool:
        """Delete a job status by ID."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the storage connection."""
        pass