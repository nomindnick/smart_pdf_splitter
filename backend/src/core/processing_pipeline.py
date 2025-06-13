"""Processing pipeline with hooks for future RAG integration."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import logging

from .unified_document_processor import UnifiedDocumentProcessor, ProcessingMode
from .models import Document, Boundary
from .ocr_config import OCRConfig

logger = logging.getLogger(__name__)


class ProcessingPurpose(str, Enum):
    """Purpose of processing - determines quality/speed tradeoffs."""
    BOUNDARY_DETECTION = "boundary_detection"  # Fast, minimal OCR
    FULL_EXTRACTION = "full_extraction"       # Complete OCR for RAG (future)


@dataclass
class ProcessingResult:
    """Result of document processing."""
    document: Document
    boundaries: List[Boundary]
    split_documents: Optional[List[Path]] = None
    processing_time: float = 0.0
    purpose: ProcessingPurpose = ProcessingPurpose.BOUNDARY_DETECTION
    metadata: Dict[str, Any] = None


class PostProcessor(ABC):
    """Abstract base for post-processing hooks."""
    
    @abstractmethod
    async def process(self, result: ProcessingResult) -> None:
        """Process the result after main processing is complete."""
        pass


class PlaceholderRAGProcessor(PostProcessor):
    """Placeholder for future RAG processing."""
    
    async def process(self, result: ProcessingResult) -> None:
        """
        Future implementation will:
        1. Queue full OCR processing
        2. Extract embeddings
        3. Store in vector database
        4. Update document metadata
        """
        logger.info(f"RAG processing placeholder for document: {result.document.id}")
        # TODO: Implement when integrating with RAG pipeline
        pass


class SmartPDFPipeline:
    """
    Main processing pipeline for smart PDF splitting.
    
    Designed with extension points for future RAG integration.
    """
    
    def __init__(
        self,
        detection_config: Optional[OCRConfig] = None,
        extraction_config: Optional[OCRConfig] = None,
        post_processors: Optional[List[PostProcessor]] = None
    ):
        """
        Initialize pipeline with separate configs for different purposes.
        
        Args:
            detection_config: Config for fast boundary detection
            extraction_config: Config for full text extraction (future)
            post_processors: List of post-processing hooks
        """
        # Detection config - optimized for speed
        self.detection_config = detection_config or OCRConfig(
            enable_ocr=True,
            ocr_engine="tesseract-cli",  # Faster
            target_dpi=150,              # Lower resolution
            preprocessing_steps=["deskew"],  # Minimal preprocessing
            confidence_threshold=0.5,    # Lower threshold acceptable
            enable_postprocessing=False, # Skip corrections for speed
            force_full_page_ocr=False,   # Only OCR text regions
            bitmap_area_threshold=0.2    # More selective
        )
        
        # Extraction config - optimized for quality (future use)
        self.extraction_config = extraction_config or OCRConfig(
            enable_ocr=True,
            ocr_engine="easyocr",        # Better quality
            target_dpi=300,              # Higher resolution
            preprocessing_steps=["deskew", "denoise", "contrast"],
            confidence_threshold=0.7,
            enable_postprocessing=True,
            apply_aggressive_corrections=True,
            force_full_page_ocr=True     # Process everything
        )
        
        # Post-processors (including RAG placeholder)
        self.post_processors = post_processors or []
        
        # Initialize processors
        self._init_processors()
    
    def _init_processors(self):
        """Initialize document processors for different purposes."""
        # Fast processor for boundary detection
        self.detection_processor = UnifiedDocumentProcessor(
            mode=ProcessingMode.SMART,
            ocr_config=self.detection_config,
            enable_adaptive=True,
            max_ocr_pages=10  # Limit for speed
        )
        
        # Full processor for extraction (future use)
        # This will be activated when needed for RAG
        self.extraction_processor = None  # Lazy initialization
    
    def process_for_splitting(
        self,
        pdf_path: Path,
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """
        Process PDF for boundary detection and splitting.
        
        This is the main method for the standalone splitter app.
        """
        import time
        start_time = time.time()
        
        logger.info(f"Processing PDF for splitting: {pdf_path}")
        
        # Step 1: Fast boundary detection
        document = self.detection_processor.process_document(
            pdf_path,
            progress_callback=progress_callback,
            return_quality_report=True
        )
        
        # Step 2: Extract boundaries from document
        # (This would use your boundary detection logic)
        boundaries = self._extract_boundaries(document)
        
        # Step 3: Create result
        result = ProcessingResult(
            document=document,
            boundaries=boundaries,
            processing_time=time.time() - start_time,
            purpose=ProcessingPurpose.BOUNDARY_DETECTION,
            metadata={
                "detection_stats": self.detection_processor.get_processing_stats(),
                "detection_config": self.detection_config.model_dump()
            }
        )
        
        # Step 4: Run post-processors (async-friendly)
        self._run_post_processors(result)
        
        return result
    
    def process_for_extraction(
        self,
        pdf_path: Path,
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """
        Process PDF for full text extraction (future RAG use).
        
        This method is a placeholder for future implementation.
        """
        raise NotImplementedError(
            "Full extraction not implemented in standalone splitter. "
            "This will be implemented in the RAG application."
        )
    
    def _extract_boundaries(self, document: Document) -> List[Boundary]:
        """
        Extract boundaries from processed document.
        
        TODO: Integrate with HybridBoundaryDetector
        """
        # Placeholder - integrate with your boundary detection
        boundaries = []
        logger.warning("Boundary extraction not yet integrated with pipeline")
        return boundaries
    
    def _run_post_processors(self, result: ProcessingResult):
        """Run all registered post-processors."""
        for processor in self.post_processors:
            try:
                # Support both sync and async processors
                import asyncio
                if asyncio.iscoroutinefunction(processor.process):
                    # Create new event loop if needed
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    loop.run_until_complete(processor.process(result))
                else:
                    processor.process(result)
            except Exception as e:
                logger.error(f"Post-processor {processor.__class__.__name__} failed: {e}")
    
    def add_post_processor(self, processor: PostProcessor):
        """Add a post-processing hook."""
        self.post_processors.append(processor)
    
    def get_extraction_processor(self) -> UnifiedDocumentProcessor:
        """
        Get or create extraction processor (lazy initialization).
        
        This allows the RAG app to access the full extraction processor
        when needed without initializing it in the standalone splitter.
        """
        if self.extraction_processor is None:
            self.extraction_processor = UnifiedDocumentProcessor(
                mode=ProcessingMode.ENHANCED,
                ocr_config=self.extraction_config,
                enable_adaptive=True
            )
        return self.extraction_processor


class CacheManager:
    """
    Manages OCR result caching between detection and extraction phases.
    
    This allows the RAG pipeline to reuse OCR results from boundary detection.
    """
    
    def __init__(self, cache_dir: Path = Path("./ocr_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
    
    def store_detection_results(self, document_id: str, results: Dict[str, Any]):
        """Store detection phase results for potential reuse."""
        cache_file = self.cache_dir / f"{document_id}_detection.json"
        # TODO: Implement caching logic
        logger.info(f"Would cache detection results for {document_id}")
    
    def get_detection_results(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached detection results if available."""
        cache_file = self.cache_dir / f"{document_id}_detection.json"
        # TODO: Implement retrieval logic
        return None


# Example usage for standalone splitter
def create_splitter_pipeline() -> SmartPDFPipeline:
    """Create pipeline configured for standalone splitter app."""
    # Just the basics for splitting
    pipeline = SmartPDFPipeline()
    
    # Add placeholder for future RAG processing
    # This won't do anything now but shows the extension point
    pipeline.add_post_processor(PlaceholderRAGProcessor())
    
    return pipeline


# Example usage for future RAG app
def create_rag_pipeline() -> SmartPDFPipeline:
    """
    Create pipeline configured for RAG application (future).
    
    This would be implemented in your RAG app, reusing the
    SmartPDFPipeline but with different configuration.
    """
    # Custom configs for RAG quality requirements
    extraction_config = OCRConfig(
        enable_ocr=True,
        ocr_engine="easyocr",
        target_dpi=300,
        # ... other quality settings
    )
    
    # Create pipeline with RAG-specific processors
    pipeline = SmartPDFPipeline(
        extraction_config=extraction_config
    )
    
    # Add RAG-specific post-processors
    # pipeline.add_post_processor(EmbeddingProcessor())
    # pipeline.add_post_processor(VectorDBWriter())
    # pipeline.add_post_processor(MetadataExtractor())
    
    return pipeline