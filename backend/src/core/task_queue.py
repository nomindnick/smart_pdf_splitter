"""Task queue abstraction for async processing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Callable
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class TaskPriority(str, Enum):
    """Task priority levels."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class TaskStatus(str, Enum):
    """Task status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class OCRTask:
    """Represents an OCR processing task."""
    task_id: str
    document_id: str
    pdf_path: Path
    priority: TaskPriority = TaskPriority.NORMAL
    purpose: str = "boundary_detection"
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class TaskQueue(ABC):
    """Abstract base for task queue implementations."""
    
    @abstractmethod
    async def enqueue(self, task: OCRTask) -> str:
        """Add task to queue and return task ID."""
        pass
    
    @abstractmethod
    async def dequeue(self) -> Optional[OCRTask]:
        """Get next task from queue."""
        pass
    
    @abstractmethod
    async def get_status(self, task_id: str) -> TaskStatus:
        """Get task status."""
        pass
    
    @abstractmethod
    async def update_status(self, task_id: str, status: TaskStatus, result: Any = None):
        """Update task status."""
        pass


class InMemoryTaskQueue(TaskQueue):
    """
    Simple in-memory task queue for standalone splitter.
    
    This is sufficient for the standalone app where we process
    synchronously and don't need persistence.
    """
    
    def __init__(self):
        self.tasks = {}
        self.pending = []
        self.results = {}
    
    async def enqueue(self, task: OCRTask) -> str:
        """Add task to queue."""
        self.tasks[task.task_id] = task
        self.pending.append(task.task_id)
        logger.info(f"Enqueued task {task.task_id} for {task.purpose}")
        return task.task_id
    
    async def dequeue(self) -> Optional[OCRTask]:
        """Get next task from queue."""
        if not self.pending:
            return None
        
        # Sort by priority
        self.pending.sort(
            key=lambda tid: (
                self.tasks[tid].priority.value,
                self.tasks[tid].created_at
            )
        )
        
        task_id = self.pending.pop(0)
        return self.tasks.get(task_id)
    
    async def get_status(self, task_id: str) -> TaskStatus:
        """Get task status."""
        if task_id not in self.tasks:
            return TaskStatus.FAILED
        
        if task_id in self.results:
            return TaskStatus.COMPLETED
        elif task_id in self.pending:
            return TaskStatus.PENDING
        else:
            return TaskStatus.PROCESSING
    
    async def update_status(self, task_id: str, status: TaskStatus, result: Any = None):
        """Update task status."""
        if status == TaskStatus.COMPLETED and result is not None:
            self.results[task_id] = result
        
        logger.info(f"Task {task_id} status updated to {status}")


class CeleryTaskQueue(TaskQueue):
    """
    Celery-based task queue for RAG application (future).
    
    This is a placeholder showing how you'd implement a production
    task queue for the RAG pipeline.
    """
    
    def __init__(self, celery_app=None):
        self.celery_app = celery_app
        logger.info("CeleryTaskQueue is a placeholder for RAG integration")
    
    async def enqueue(self, task: OCRTask) -> str:
        """
        In RAG app, this would:
        1. Serialize task
        2. Send to Celery
        3. Return task ID for tracking
        """
        # TODO: Implement when integrating with RAG
        raise NotImplementedError("Celery queue for RAG app not implemented")
    
    async def dequeue(self) -> Optional[OCRTask]:
        """Workers automatically process from Celery queue."""
        raise NotImplementedError("Celery queue for RAG app not implemented")
    
    async def get_status(self, task_id: str) -> TaskStatus:
        """Query Celery for task status."""
        # TODO: self.celery_app.AsyncResult(task_id).status
        raise NotImplementedError("Celery queue for RAG app not implemented")
    
    async def update_status(self, task_id: str, status: TaskStatus, result: Any = None):
        """Celery handles status updates automatically."""
        raise NotImplementedError("Celery queue for RAG app not implemented")


class TaskQueueFactory:
    """Factory for creating appropriate task queue."""
    
    @staticmethod
    def create(queue_type: str = "memory", **kwargs) -> TaskQueue:
        """
        Create task queue based on type.
        
        Args:
            queue_type: "memory" for standalone, "celery" for RAG
            **kwargs: Additional arguments for queue initialization
        """
        if queue_type == "memory":
            return InMemoryTaskQueue()
        elif queue_type == "celery":
            # For RAG app
            celery_app = kwargs.get("celery_app")
            return CeleryTaskQueue(celery_app)
        else:
            raise ValueError(f"Unknown queue type: {queue_type}")


# Placeholder for future OCR worker
class OCRWorker:
    """
    Worker that processes OCR tasks from queue.
    
    In standalone app: Runs synchronously
    In RAG app: Runs as Celery worker
    """
    
    def __init__(self, queue: TaskQueue, processor_factory: Callable):
        self.queue = queue
        self.processor_factory = processor_factory
    
    async def process_next(self):
        """Process next task from queue."""
        task = await self.queue.dequeue()
        if not task:
            return None
        
        await self.queue.update_status(task.task_id, TaskStatus.PROCESSING)
        
        try:
            # Get appropriate processor
            processor = self.processor_factory(task.purpose)
            
            # Process document
            result = processor.process_document(task.pdf_path)
            
            # Update status
            await self.queue.update_status(
                task.task_id,
                TaskStatus.COMPLETED,
                result
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            await self.queue.update_status(
                task.task_id,
                TaskStatus.FAILED,
                {"error": str(e)}
            )
            return None