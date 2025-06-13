"""Document processing service with enhanced OCR."""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

from ...core.enhanced_document_processor_v2 import EnhancedDocumentProcessorV2 as EnhancedDocumentProcessor
from ...core.boundary_detector import BoundaryDetector
from ...core.models import Document, ProcessingStatus, DocumentMetadata, Boundary

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for processing documents with enhanced OCR."""
    
    def __init__(self, enable_adaptive_ocr: bool = True):
        """Initialize document service.
        
        Args:
            enable_adaptive_ocr: Enable adaptive OCR configuration
        """
        self.enable_adaptive_ocr = enable_adaptive_ocr
        self.enhanced_processor = EnhancedDocumentProcessor(
            enable_adaptive=enable_adaptive_ocr,
            language="en"
        )
        self.boundary_detector = BoundaryDetector()
    
    async def process_document(
        self,
        document: Document,
        progress_callback: Optional[callable] = None
    ) -> Document:
        """Process a document with enhanced OCR and boundary detection.
        
        Args:
            document: Document to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processed document with boundaries and metadata
        """
        try:
            start_time = time.time()
            logger.info(f"Starting processing for document {document.id}")
            
            # Update status
            document.status = ProcessingStatus.PROCESSING
            
            # Get file path
            pdf_path = Path(document.original_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Process with enhanced OCR
            processed_doc = self.enhanced_processor.process_document(
                pdf_path,
                progress_callback=progress_callback,
                return_quality_report=True
            )
            
            # Update document with processed data
            document.total_pages = processed_doc.total_pages
            document.page_info = processed_doc.page_info
            document.metadata = processed_doc.metadata
            
            # Detect boundaries
            boundaries = self.boundary_detector.detect_boundaries(processed_doc)
            document.detected_boundaries = boundaries
            
            # Calculate processing time
            processing_time = time.time() - start_time
            document.processing_time = processing_time
            
            # Update status
            document.status = ProcessingStatus.COMPLETED
            
            # Get OCR statistics
            stats = self.enhanced_processor.get_processing_stats()
            
            logger.info(
                f"Document {document.id} processed successfully. "
                f"Pages: {document.total_pages}, "
                f"Boundaries: {len(boundaries)}, "
                f"Time: {processing_time:.2f}s, "
                f"OCR confidence: {stats['average_confidence']:.2%}"
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error processing document {document.id}: {str(e)}", exc_info=True)
            document.status = ProcessingStatus.FAILED
            document.error_message = str(e)
            document.processing_time = time.time() - start_time
            return document
    
    def get_document_quality_summary(self, document: Document) -> Dict:
        """Get quality summary for a processed document.
        
        Args:
            document: Processed document
            
        Returns:
            Quality summary dict
        """
        if not document.metadata or not document.metadata.custom_fields:
            return {"error": "No quality data available"}
        
        custom_fields = document.metadata.custom_fields
        
        summary = {
            "overall_confidence": custom_fields.get("overall_confidence", 0),
            "quality_assessment": custom_fields.get("quality_assessment", "unknown"),
            "processing_stats": custom_fields.get("processing_stats", {}),
            "ocr_config": custom_fields.get("ocr_config", {}),
        }
        
        # Add page-level summary
        if document.page_info:
            low_confidence_pages = []
            pages_needing_review = []
            
            for page in document.page_info:
                if page.ocr_confidence and page.ocr_confidence < 0.7:
                    low_confidence_pages.append({
                        "page": page.page_number,
                        "confidence": page.ocr_confidence,
                        "issues": page.ocr_issues
                    })
                
                if page.needs_review:
                    pages_needing_review.append(page.page_number)
            
            summary["low_confidence_pages"] = low_confidence_pages
            summary["pages_needing_review"] = pages_needing_review
            summary["total_pages_with_issues"] = len(low_confidence_pages)
        
        return summary
    
    def get_page_ocr_details(self, document: Document, page_number: int) -> Dict:
        """Get detailed OCR information for a specific page.
        
        Args:
            document: Processed document
            page_number: Page number (1-indexed)
            
        Returns:
            Page OCR details
        """
        if not document.page_info or page_number < 1 or page_number > len(document.page_info):
            return {"error": "Invalid page number"}
        
        page = document.page_info[page_number - 1]
        
        details = {
            "page_number": page.page_number,
            "word_count": page.word_count,
            "ocr_confidence": page.ocr_confidence,
            "ocr_quality_assessment": page.ocr_quality_assessment,
            "ocr_issues": page.ocr_issues,
            "preprocessing_applied": page.preprocessing_applied,
            "corrections_made": page.corrections_made,
            "needs_review": page.needs_review,
            "has_images": page.has_images,
            "has_tables": page.has_tables,
        }
        
        # Add quality report if available
        if (document.metadata and 
            document.metadata.custom_fields and 
            "page_quality_reports" in document.metadata.custom_fields):
            
            reports = document.metadata.custom_fields["page_quality_reports"]
            if page_number <= len(reports):
                details["quality_report"] = reports[page_number - 1]
        
        return details