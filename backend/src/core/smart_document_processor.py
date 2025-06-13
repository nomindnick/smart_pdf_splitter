"""Smart document processor with optimized OCR handling."""

import logging
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, List, Tuple
import fitz  # PyMuPDF

from .document_processor import DocumentProcessor
from .models import PageInfo, Document
from .ocr_optimizer import check_page_needs_ocr

logger = logging.getLogger(__name__)


class SmartDocumentProcessor:
    """
    Enhanced document processor with smart OCR handling.
    
    Features:
    - Auto-detects if pages need OCR
    - Processes text and image pages differently
    - Optimizes memory usage for large documents
    - Provides progress tracking
    """
    
    def __init__(
        self,
        enable_ocr: bool = True,
        ocr_languages: List[str] = None,
        memory_limit_mb: int = 4096,
        ocr_page_limit: int = 100,  # Max pages to OCR in one document
        text_confidence_threshold: int = 50  # Min chars to consider page has text
    ):
        """
        Initialize smart document processor.
        
        Args:
            enable_ocr: Whether to enable OCR for image pages
            ocr_languages: Languages for OCR
            memory_limit_mb: Memory limit in MB
            ocr_page_limit: Maximum pages to OCR per document
            text_confidence_threshold: Minimum characters to consider page has text
        """
        self.enable_ocr = enable_ocr
        self.ocr_languages = ocr_languages or ["en"]
        self.memory_limit_mb = memory_limit_mb
        self.ocr_page_limit = ocr_page_limit
        self.text_confidence_threshold = text_confidence_threshold
        
        # Processors for different page types
        self.text_processor = DocumentProcessor(
            enable_ocr=False,  # Text pages don't need OCR
            page_batch_size=10  # Can process more text pages at once
        )
        
        self.ocr_processor = DocumentProcessor(
            enable_ocr=True,
            ocr_engine="easyocr",
            page_batch_size=2,  # Smaller batches for OCR
            max_memory_mb=memory_limit_mb
        ) if enable_ocr else None
    
    def analyze_document(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Analyze document to determine processing strategy.
        
        Returns:
            Dict with analysis results
        """
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        
        text_pages = []
        image_pages = []
        
        # Sample pages to determine document type
        sample_size = min(10, total_pages)
        sample_step = max(1, total_pages // sample_size)
        
        for i in range(0, total_pages, sample_step):
            if i >= total_pages:
                break
                
            page = doc[i]
            needs_ocr, reason = check_page_needs_ocr(
                page,
                text_threshold=self.text_confidence_threshold
            )
            
            if needs_ocr:
                image_pages.append(i)
            else:
                text_pages.append(i)
        
        doc.close()
        
        # Determine document type
        image_ratio = len(image_pages) / len(text_pages + image_pages)
        
        return {
            "total_pages": total_pages,
            "sampled_pages": len(text_pages) + len(image_pages),
            "text_pages_sampled": len(text_pages),
            "image_pages_sampled": len(image_pages),
            "image_ratio": image_ratio,
            "document_type": "image_based" if image_ratio > 0.7 else (
                "mixed" if image_ratio > 0.3 else "text_based"
            ),
            "recommended_ocr": image_ratio > 0.3
        }
    
    def process_document(
        self,
        pdf_path: Path,
        progress_callback: Optional[callable] = None
    ) -> Document:
        """
        Process document with smart OCR handling.
        
        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Document object with processed pages
        """
        # Analyze document first
        analysis = self.analyze_document(pdf_path)
        logger.info(f"Document analysis: {analysis}")
        
        if progress_callback:
            progress_callback(0, f"Analyzing document: {analysis['document_type']}")
        
        # Open document for page-by-page processing
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        pages = []
        ocr_pages_processed = 0
        
        for page_num in range(total_pages):
            if progress_callback:
                progress = (page_num / total_pages) * 100
                progress_callback(progress, f"Processing page {page_num + 1}/{total_pages}")
            
            page = doc[page_num]
            
            # Check if this specific page needs OCR
            needs_ocr, reason = check_page_needs_ocr(
                page,
                text_threshold=self.text_confidence_threshold
            )
            
            # Extract basic page info
            page_info = PageInfo(
                page_number=page_num + 1,
                width=page.rect.width,
                height=page.rect.height,
                text_content=None,
                word_count=0
            )
            
            if not needs_ocr:
                # Extract text directly from PDF
                text = page.get_text()
                page_info.text_content = text
                page_info.word_count = len(text.split()) if text else 0
                logger.debug(f"Page {page_num + 1}: Extracted {page_info.word_count} words directly")
                
            elif self.enable_ocr and ocr_pages_processed < self.ocr_page_limit:
                # Use OCR for this page
                logger.info(f"Page {page_num + 1}: Running OCR - {reason}")
                
                # Save page as temporary image for OCR
                pix = page.get_pixmap(dpi=150)  # Lower DPI for faster OCR
                img_data = pix.tobytes("png")
                
                # TODO: Pass image to OCR processor
                # For now, just mark as needing OCR
                page_info.text_content = f"[Page {page_num + 1} requires OCR - {reason}]"
                ocr_pages_processed += 1
                
                pix = None  # Free memory
            else:
                # OCR disabled or limit reached
                page_info.text_content = f"[Page {page_num + 1} is image-based but OCR skipped]"
                logger.warning(f"Page {page_num + 1}: Skipping OCR - limit reached or disabled")
            
            pages.append(page_info)
        
        doc.close()
        
        if progress_callback:
            progress_callback(100, "Processing complete")
        
        # Create document object
        import os
        file_size = os.path.getsize(pdf_path)
        
        # Create metadata
        from .models import DocumentMetadata
        metadata = DocumentMetadata(
            page_count=total_pages,
            file_size=file_size,
            custom_fields={
                "analysis": analysis,
                "ocr_pages_processed": ocr_pages_processed,
                "processing_strategy": "smart"
            }
        )
        
        document = Document(
            id=pdf_path.stem,
            filename=pdf_path.name,
            total_pages=total_pages,
            file_size=file_size,
            page_info=pages,
            metadata=metadata
        )
        
        return document
    
    def process_with_fallback(
        self,
        pdf_path: Path,
        use_smart_mode: bool = True
    ) -> Iterator[PageInfo]:
        """
        Process document with fallback to standard processing.
        
        Args:
            pdf_path: Path to PDF
            use_smart_mode: Whether to use smart processing
            
        Yields:
            PageInfo objects
        """
        if use_smart_mode:
            try:
                document = self.process_document(pdf_path)
                yield from document.pages
            except Exception as e:
                logger.error(f"Smart processing failed: {e}, falling back to standard")
                yield from self.text_processor.process_document(pdf_path)
        else:
            # Use standard processing
            processor = self.ocr_processor if self.enable_ocr else self.text_processor
            yield from processor.process_document(pdf_path)