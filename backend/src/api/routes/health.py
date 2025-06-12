"""Health check endpoints."""

from typing import Dict

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import get_db

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "Smart PDF Splitter API"}


@router.get("/health/db")
async def database_health(db: AsyncSession = Depends(get_db)) -> Dict[str, str]:
    """Check database connectivity."""
    try:
        # Execute a simple query
        result = await db.execute(text("SELECT 1"))
        result.scalar()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}


@router.get("/health/ready")
async def readiness_check() -> Dict[str, str]:
    """Readiness check for Kubernetes/Docker."""
    # Add more checks here (Redis, Celery, etc.)
    return {"status": "ready", "service": "Smart PDF Splitter API"}