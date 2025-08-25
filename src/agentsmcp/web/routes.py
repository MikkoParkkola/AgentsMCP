"""
Web Dashboard Routes for AgentsMCP

Provides real-time web interface with Server-Sent Events, cost tracking,
model management, and agent orchestration controls.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

# Cost Intelligence Integration
try:
    from ..cost.tracker import CostTracker
    from ..cost.optimizer import ModelOptimizer  
    from ..cost.budget import BudgetManager
    COST_AVAILABLE = True
except ImportError:
    COST_AVAILABLE = False

# Agent Integration
try:
    from ..agent_manager import AgentManager
    from ..config import Config
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False

# Orchestration Integration  
try:
    from ..orchestration.orchestration_manager import OrchestrationManager
    ORCHESTRATION_AVAILABLE = True
except ImportError:
    ORCHESTRATION_AVAILABLE = False


# Global instances
cost_tracker = CostTracker() if COST_AVAILABLE else None
event_queue: asyncio.Queue = asyncio.Queue()


# Request/Response Models
class SpawnRequest(BaseModel):
    model: str
    task: str
    timeout: int = 300
    cost_limit: Optional[float] = None


class ModelInfo(BaseModel):
    name: str
    provider: str
    cost_per_token: float
    capabilities: List[str]
    performance_score: Optional[float] = None


class CostData(BaseModel):
    total_cost: float
    daily_cost: float
    monthly_cost: float
    breakdown: Dict[str, Dict[str, float]]
    budget_status: Dict[str, Any]


# Create router
router = APIRouter(prefix="/api", tags=["dashboard"])


# Server-Sent Events endpoint
@router.get("/events")
async def stream_events(request: Request):
    """
    Stream real-time events to web dashboard clients.
    
    Events include:
    - cost_update: Real-time cost changes
    - agent_spawned: New agent started
    - agent_completed: Agent finished task
    - budget_alert: Budget threshold exceeded
    """
    
    async def generate_events():
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break
                
            try:
                # Get event from queue (with timeout to send heartbeat)
                event = await asyncio.wait_for(event_queue.get(), timeout=5.0)
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                heartbeat = {
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                    "server_time": datetime.utcnow().strftime("%H:%M:%S")
                }
                yield f"data: {json.dumps(heartbeat)}\n\n"
            except Exception as e:
                error_event = {
                    "type": "error", 
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                yield f"data: {json.dumps(error_event)}\n\n"
                break
    
    return StreamingResponse(
        generate_events(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


@router.get("/costs", response_model=CostData)
async def get_costs():
    """Get current cost data and budget status."""
    
    if not COST_AVAILABLE or not cost_tracker:
        raise HTTPException(status_code=503, detail="Cost tracking not available")
    
    try:
        # Get current costs
        total_cost = cost_tracker.total_cost
        daily_cost = cost_tracker.get_daily_cost()
        
        # Get monthly cost
        now = datetime.utcnow()
        monthly_cost = cost_tracker.get_monthly_cost(now.year, now.month)
        
        # Get breakdown
        breakdown = cost_tracker.get_breakdown()
        
        # Budget status (using default $100 budget for demo)
        budget_manager = BudgetManager(cost_tracker, 100.0)
        budget_status = {
            "budget_limit": 100.0,
            "within_budget": budget_manager.check_budget(),
            "remaining": budget_manager.remaining_budget(),
            "usage_percentage": (total_cost / 100.0) * 100
        }
        
        return CostData(
            total_cost=total_cost,
            daily_cost=daily_cost,
            monthly_cost=monthly_cost,
            breakdown=breakdown,
            budget_status=budget_status
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get costs: {e}")


@router.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Get available models with cost and performance data."""
    
    # Mock model data - in real implementation would fetch from providers
    models = [
        ModelInfo(
            name="gpt-4",
            provider="openai", 
            cost_per_token=0.00003,
            capabilities=["reasoning", "coding", "creative"],
            performance_score=95.0
        ),
        ModelInfo(
            name="gpt-3.5-turbo",
            provider="openai",
            cost_per_token=0.0000015,
            capabilities=["general", "coding"],
            performance_score=85.0
        ),
        ModelInfo(
            name="gpt-oss:20b", 
            provider="ollama",
            cost_per_token=0.0,  # Free local model
            capabilities=["general", "coding"],
            performance_score=80.0
        ),
        ModelInfo(
            name="claude-3-sonnet",
            provider="anthropic",
            cost_per_token=0.000003,
            capabilities=["reasoning", "creative", "analysis"],
            performance_score=92.0
        )
    ]
    
    return models


@router.post("/spawn")
async def spawn_agent(request: SpawnRequest, background_tasks: BackgroundTasks):
    """Spawn a new agent with cost prediction and monitoring."""
    
    if not AGENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent manager not available")
    
    try:
        # Cost prediction
        estimated_tokens = len(request.task) * 2  # Rough estimate
        model_cost = 0.000003  # Mock cost per token
        predicted_cost = estimated_tokens * model_cost
        
        # Check cost limit
        if request.cost_limit and predicted_cost > request.cost_limit:
            raise HTTPException(
                status_code=400,
                detail=f"Predicted cost ${predicted_cost:.6f} exceeds limit ${request.cost_limit:.6f}"
            )
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Publish spawn event
        spawn_event = {
            "type": "agent_spawned",
            "job_id": job_id,
            "model": request.model,
            "task": request.task[:100] + "..." if len(request.task) > 100 else request.task,
            "predicted_cost": predicted_cost,
            "timestamp": datetime.utcnow().isoformat()
        }
        await event_queue.put(spawn_event)
        
        # Add background task to simulate agent work
        background_tasks.add_task(simulate_agent_work, job_id, request.model, predicted_cost)
        
        return {
            "job_id": job_id,
            "status": "spawned",
            "predicted_cost": predicted_cost,
            "model": request.model,
            "estimated_duration": request.timeout
        }
        
    except Exception as e:
        error_event = {
            "type": "error",
            "message": f"Failed to spawn agent: {e}",
            "timestamp": datetime.utcnow().isoformat()
        }
        await event_queue.put(error_event)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_system_status():
    """Get overall system status and health metrics."""
    
    status = {
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "cost_tracking": COST_AVAILABLE,
            "agent_manager": AGENT_AVAILABLE, 
            "orchestration": ORCHESTRATION_AVAILABLE,
        },
        "metrics": {
            "uptime_seconds": 3600,  # Mock uptime
            "active_agents": 0,  # Mock count
            "total_requests": 42,  # Mock counter
        }
    }
    
    if COST_AVAILABLE and cost_tracker:
        status["cost_metrics"] = {
            "total_cost": cost_tracker.total_cost,
            "daily_cost": cost_tracker.get_daily_cost()
        }
    
    return status


@router.post("/budget")
async def set_budget(budget_data: dict):
    """Set monthly budget limit."""
    
    if not COST_AVAILABLE:
        raise HTTPException(status_code=503, detail="Budget management not available")
    
    try:
        amount = float(budget_data.get("amount", 100.0))
        
        # Create budget manager with new limit
        budget_manager = BudgetManager(cost_tracker, amount)
        
        # Check if already over budget
        if not budget_manager.check_budget():
            alert_event = {
                "type": "budget_alert",
                "message": f"Already over new budget of ${amount:.2f}!",
                "current_cost": cost_tracker.total_cost,
                "budget_limit": amount,
                "timestamp": datetime.utcnow().isoformat()
            }
            await event_queue.put(alert_event)
        
        return {
            "budget_limit": amount,
            "current_cost": cost_tracker.total_cost,
            "within_budget": budget_manager.check_budget(),
            "remaining": budget_manager.remaining_budget()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid budget data: {e}")


# Background task to simulate agent work
async def simulate_agent_work(job_id: str, model: str, predicted_cost: float):
    """Simulate agent work and publish completion event."""
    
    # Simulate work duration
    await asyncio.sleep(5.0)
    
    # Record actual cost (slightly different from prediction)
    actual_cost = predicted_cost * (0.8 + 0.4 * asyncio.get_event_loop().time() % 1)
    
    if COST_AVAILABLE and cost_tracker:
        from ..cost.models import CostRecord
        record = CostRecord(
            call_id=job_id,
            provider=model.split("-")[0] if "-" in model else "unknown",
            model=model,
            task="simulated_task",
            input_tokens=100,
            output_tokens=200,
            cost=actual_cost
        )
        cost_tracker.record_call(record, actual_cost / 300)  # Token price
    
    # Publish completion event
    completion_event = {
        "type": "agent_completed", 
        "job_id": job_id,
        "model": model,
        "actual_cost": actual_cost,
        "predicted_cost": predicted_cost,
        "cost_accuracy": abs(actual_cost - predicted_cost) / predicted_cost * 100,
        "timestamp": datetime.utcnow().isoformat()
    }
    await event_queue.put(completion_event)
    
    # Publish cost update
    if COST_AVAILABLE and cost_tracker:
        cost_event = {
            "type": "cost_update",
            "total_cost": cost_tracker.total_cost,
            "latest_cost": actual_cost,
            "timestamp": datetime.utcnow().isoformat()
        }
        await event_queue.put(cost_event)


# Background task to periodically update costs
async def cost_update_publisher():
    """Background task to publish periodic cost updates."""
    
    while True:
        await asyncio.sleep(10.0)  # Update every 10 seconds
        
        if COST_AVAILABLE and cost_tracker:
            cost_event = {
                "type": "cost_update",
                "total_cost": cost_tracker.total_cost,
                "daily_cost": cost_tracker.get_daily_cost(),
                "timestamp": datetime.utcnow().isoformat()
            }
            await event_queue.put(cost_event)


# Initialize background tasks
def init_background_tasks():
    """Initialize background tasks for the web dashboard."""
    asyncio.create_task(cost_update_publisher())