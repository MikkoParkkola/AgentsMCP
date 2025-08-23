from datetime import datetime

import pytest

from agentsmcp.models import JobState, JobStatus
from agentsmcp.storage.memory import MemoryStorage


@pytest.fixture
def memory_storage():
    """Create memory storage for testing."""
    return MemoryStorage()


@pytest.fixture
def sample_job_status():
    """Create sample job status for testing."""
    return JobStatus(
        job_id="test-123",
        state=JobState.PENDING,
        output=None,
        error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )


@pytest.mark.asyncio
async def test_store_and_get_job_status(memory_storage, sample_job_status):
    """Test storing and retrieving job status."""
    # Store job status
    await memory_storage.store_job_status(sample_job_status)
    
    # Retrieve job status
    retrieved = await memory_storage.get_job_status("test-123")
    
    assert retrieved is not None
    assert retrieved.job_id == "test-123"
    assert retrieved.state == JobState.PENDING


@pytest.mark.asyncio
async def test_get_nonexistent_job_status(memory_storage):
    """Test retrieving non-existent job status."""
    result = await memory_storage.get_job_status("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_list_job_statuses(memory_storage):
    """Test listing job statuses."""
    # Add multiple job statuses
    for i in range(3):
        status = JobStatus(
            job_id=f"job-{i}",
            state=JobState.PENDING
        )
        await memory_storage.store_job_status(status)
    
    # List all statuses
    statuses = await memory_storage.list_job_statuses()
    assert len(statuses) == 3
    
    # List with limit
    limited = await memory_storage.list_job_statuses(limit=2)
    assert len(limited) == 2


@pytest.mark.asyncio
async def test_delete_job_status(memory_storage, sample_job_status):
    """Test deleting job status."""
    # Store job status
    await memory_storage.store_job_status(sample_job_status)
    
    # Delete job status
    success = await memory_storage.delete_job_status("test-123")
    assert success is True
    
    # Verify it's gone
    retrieved = await memory_storage.get_job_status("test-123")
    assert retrieved is None
    
    # Try to delete non-existent
    success = await memory_storage.delete_job_status("nonexistent")
    assert success is False


@pytest.mark.asyncio
async def test_close_storage(memory_storage, sample_job_status):
    """Test closing storage."""
    # Store some data
    await memory_storage.store_job_status(sample_job_status)
    
    # Close storage
    await memory_storage.close()
    
    # Verify data is cleared
    retrieved = await memory_storage.get_job_status("test-123")
    assert retrieved is None


def test_job_status_post_init():
    """Test JobStatus post_init method."""
    # Test with no timestamps
    status = JobStatus(job_id="test", state=JobState.PENDING)
    assert status.created_at is not None
    assert status.updated_at is not None
    
    # Test with custom timestamp
    custom_time = datetime.utcnow()
    status = JobStatus(
        job_id="test", 
        state=JobState.PENDING,
        created_at=custom_time
    )
    assert status.created_at == custom_time
    assert status.updated_at == custom_time