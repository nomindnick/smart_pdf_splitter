"""Enhanced document processing service with pipeline architecture."""

import logging
import time
import asyncio
from pathlib import Path
from typing import Dict, Optional
import uuid

from ...core.models import Document, ProcessingStatus, DocumentMetadata, Boundary
from ...core.processing_pipeline import SmartPDFPipeline, ProcessingPurpose, ProcessingResult
from ...core.task_queue import TaskQueueFactory, OCRTask, TaskPriority, TaskStatus
from ...core.parallel_processor import AdaptiveParallelProcessor
from ...core.hybrid_boundary_detector import HybridBoundaryDetector, VisualProcessingConfig

logger = logging.getLogger(__name__)


class DocumentServiceV2:
    """
    Enhanced document service using pipeline architecture.
    
    This service is designed for the standalone splitter but includes
    hooks for future RAG integration.
    """
    
    def __init__(
        self,
        enable_parallel_processing: bool = True,
        queue_type: str = "memory",
        processing_mode: str = "smart"
    ):
        """
        Initialize enhanced document service.
        
        Args:
            enable_parallel_processing: Enable parallel OCR processing
            queue_type: Task queue type ("memory" for standalone)
            processing_mode: Default processing mode
        """
        # Initialize pipeline
        self.pipeline = SmartPDFPipeline()
        
        # Initialize task queue
        self.task_queue = TaskQueueFactory.create(queue_type)
        
        # Initialize parallel processor if enabled
        self.parallel_processor = AdaptiveParallelProcessor() if enable_parallel_processing else None
        
        # Initialize boundary detector with visual features
        self.boundary_detector = HybridBoundaryDetector(
            config=VisualProcessingConfig(
                enable_visual_features=True,
                visual_confidence_threshold=0.5
            )
        )
        
        self.processing_mode = processing_mode
        
        logger.info(
            f"Initialized DocumentServiceV2 with parallel={enable_parallel_processing}, "
            f"queue={queue_type}, mode={processing_mode}"
        )
    
    async def process_document(
        self,
        document: Document,
        progress_callback: Optional[callable] = None,
        use_parallel: Optional[bool] = None
    ) -> Document:
        """
        Process document through the pipeline.
        
        Args:
            document: Document to process
            progress_callback: Progress callback
            use_parallel: Override parallel processing setting
            
        Returns:
            Processed document with boundaries
        """
        try:
            start_time = time.time()
            logger.info(f"Starting pipeline processing for document {document.id}")
            
            # Update status
            document.status = ProcessingStatus.PROCESSING
            
            # Get file path
            pdf_path = Path(document.original_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Create OCR task
            task = OCRTask(
                task_id=str(uuid.uuid4()),
                document_id=document.id,
                pdf_path=pdf_path,
                priority=TaskPriority.NORMAL,
                purpose="boundary_detection"
            )
            
            # Queue task
            await self.task_queue.enqueue(task)
            
            # Process through pipeline
            if use_parallel and self.parallel_processor:
                result = await self._process_parallel(pdf_path, progress_callback)
            else:
                result = self.pipeline.process_for_splitting(pdf_path, progress_callback)
            
            # Detect boundaries using hybrid detector
            boundaries = self.boundary_detector.detect_boundaries(pdf_path)
            
            # Update document
            document.total_pages = result.document.total_pages
            document.page_info = result.document.page_info
            document.metadata = result.document.metadata
            document.detected_boundaries = boundaries
            document.processing_time = time.time() - start_time
            document.status = ProcessingStatus.COMPLETED
            
            # Update task status
            await self.task_queue.update_status(
                task.task_id,
                TaskStatus.COMPLETED,
                {"boundaries": len(boundaries)}
            )
            
            # Log summary
            stats = result.metadata.get("detection_stats", {})
            logger.info(
                f"Document {document.id} processed successfully. "
                f"Pages: {document.total_pages}, "
                f"Boundaries: {len(boundaries)}, "
                f"Time: {document.processing_time:.2f}s, "
                f"OCR pages: {stats.get('ocr_performed', 0)}"
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error processing document {document.id}: {str(e)}", exc_info=True)
            document.status = ProcessingStatus.FAILED
            document.error_message = str(e)
            document.processing_time = time.time() - start_time
            
            # Update task status
            if 'task' in locals():
                await self.task_queue.update_status(
                    task.task_id,
                    TaskStatus.FAILED,
                    {"error": str(e)}
                )
            
            return document
    
    async def _process_parallel(
        self,
        pdf_path: Path,
        progress_callback: Optional[callable]
    ) -> ProcessingResult:
        """
        Process document using parallel OCR.
        
        This demonstrates how parallel processing can be integrated
        while maintaining compatibility with the pipeline.
        """
        # Use parallel processor for OCR
        config = self.pipeline.detection_config
        results = self.parallel_processor.process_document(
            pdf_path,
            config,
            purpose="detection"
        )
        
        # Convert parallel results to pipeline format
        # This is a simplified conversion - full implementation would
        # properly integrate with the pipeline
        from ...core.models import PageInfo
        pages = []
        for page_num, result in sorted(results.items()):
            pages.append(PageInfo(
                page_number=page_num + 1,
                text_content=result.text,
                word_count=result.word_count,
                ocr_confidence=result.confidence
            ))
        
        # Create document
        from ...core.models import Document as DocModel
        doc = DocModel(
            id=str(uuid.uuid4()),
            filename=pdf_path.name,
            total_pages=len(pages),
            page_info=pages,
            metadata=DocumentMetadata(
                page_count=len(pages),
                custom_fields={"parallel_processing": True}
            )
        )
        
        # Return as processing result
        return ProcessingResult(
            document=doc,
            boundaries=[],
            processing_time=0.0,
            purpose=ProcessingPurpose.BOUNDARY_DETECTION
        )
    
    async def get_task_status(self, task_id: str) -> Dict:
        """
        Get status of a processing task.
        
        This allows checking on background processing status,
        which will be useful for the RAG pipeline.
        """
        status = await self.task_queue.get_status(task_id)
        return {
            "task_id": task_id,
            "status": status.value
        }
    
    def prepare_for_rag_export(self, document: Document) -> Dict:
        """
        Prepare document data for export to RAG pipeline.
        
        This method demonstrates how the splitter can prepare data
        for the future RAG integration.
        """
        # Extract key information for RAG
        rag_data = {
            "document_id": document.id,
            "filename": document.filename,
            "total_pages": document.total_pages,
            "boundaries": [
                {
                    "start_page": b.start_page,
                    "end_page": b.end_page,
                    "confidence": b.confidence,
                    "document_type": b.document_type.value if b.document_type else "unknown"
                }
                for b in document.detected_boundaries
            ],
            "ocr_cache": {
                # Include any OCR results that can be reused
                "pages_processed": [],
                "detection_config": self.pipeline.detection_config.model_dump()
            },
            "metadata": document.metadata.model_dump() if document.metadata else {}
        }
        
        # Include OCR results for pages that were processed
        if document.page_info:
            for page in document.page_info:
                if page.text_content:
                    rag_data["ocr_cache"]["pages_processed"].append({
                        "page_number": page.page_number,
                        "text_preview": page.text_content[:200],
                        "word_count": page.word_count,
                        "confidence": page.ocr_confidence
                    })
        
        return rag_data


# Factory function for easy switching between versions
def create_document_service(version: str = "v2", **kwargs) -> Union[DocumentService, DocumentServiceV2]:
    """
    Create document service instance.
    
    Args:
        version: Service version ("v1" or "v2")
        **kwargs: Additional configuration
        
    Returns:
        Document service instance
    """
    if version == "v1":
        from .document_service import DocumentService
        return DocumentService(**kwargs)
    else:
        return DocumentServiceV2(**kwargs)