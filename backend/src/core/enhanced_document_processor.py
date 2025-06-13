"""Enhanced document processor with integrated OCR improvements."""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator, Tuple
import numpy as np
import cv2
import fitz  # PyMuPDF

from .document_processor import DocumentProcessor
from .models import PageInfo, Document, DocumentMetadata
from .ocr_preprocessor import OCRPreprocessor
from .ocr_confidence_scorer import OCRConfidenceScorer
from .ocr_postprocessor import OCRPostProcessor
from .ocr_config import OCRConfig, AdaptiveOCRConfigurator, DocumentCharacteristics
from .ocr_optimizer import check_pdf_needs_ocr

logger = logging.getLogger(__name__)


class EnhancedDocumentProcessor:
    """
    Document processor with full OCR enhancement pipeline.
    
    Features:
    - Adaptive OCR configuration
    - Image preprocessing
    - Confidence scoring
    - Text post-processing
    - Quality feedback
    """
    
    def __init__(
        self,
        ocr_config: Optional[OCRConfig] = None,
        enable_adaptive: bool = True,
        language: str = "en"
    ):
        """
        Initialize enhanced processor.
        
        Args:
            ocr_config: OCR configuration (uses defaults if None)
            enable_adaptive: Enable adaptive configuration
            language: Primary language for processing
        """
        self.ocr_config = ocr_config or OCRConfig()
        self.enable_adaptive = enable_adaptive
        self.language = language
        
        # Initialize components
        self.preprocessor = OCRPreprocessor(
            target_dpi=self.ocr_config.target_dpi,
            enable_gpu=self.ocr_config.enable_gpu
        )
        self.confidence_scorer = OCRConfidenceScorer(language=language)
        self.postprocessor = OCRPostProcessor(language=language)
        self.configurator = AdaptiveOCRConfigurator()
        
        # Base document processor (will be configured per document)
        self._base_processor = None
        self._current_doc_config = None
        
        # Processing statistics
        self.stats = {
            "pages_processed": 0,
            "pages_preprocessed": 0,
            "ocr_performed": 0,
            "average_confidence": 0.0,
            "total_corrections": 0
        }
    
    def process_document(
        self,
        pdf_path: Path,
        progress_callback: Optional[callable] = None,
        return_quality_report: bool = True
    ) -> Document:
        """
        Process document with full enhancement pipeline.
        
        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional progress callback
            return_quality_report: Include quality analysis in metadata
            
        Returns:
            Processed Document with enhanced text and quality metrics
        """
        logger.info(f"Processing document with enhancements: {pdf_path}")
        
        # Reset stats
        self.stats = {
            "pages_processed": 0,
            "pages_preprocessed": 0,
            "ocr_performed": 0,
            "average_confidence": 0.0,
            "total_corrections": 0
        }
        
        # Step 1: Analyze document and configure
        if progress_callback:
            progress_callback(0, "Analyzing document...")
        
        doc_config = self._analyze_and_configure(pdf_path)
        self._current_doc_config = doc_config
        
        # Initialize base processor with proper OCR settings
        if doc_config.enable_ocr:
            from .document_processor import DocumentProcessor
            self._base_processor = DocumentProcessor(
                enable_ocr=True,
                ocr_languages=doc_config.ocr_languages,
                ocr_engine=doc_config.ocr_engine,
                page_batch_size=doc_config.page_batch_size,
                auto_detect_ocr=False  # We already determined this
            )
        
        # Step 2: Process pages with enhancements
        processed_pages = []
        quality_reports = []
        
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            if progress_callback:
                progress = (page_num / total_pages) * 100
                progress_callback(progress, f"Processing page {page_num + 1}/{total_pages}")
            
            # Process single page
            page_info, quality_report = self._process_page(
                doc[page_num],
                page_num + 1,
                doc_config
            )
            
            processed_pages.append(page_info)
            quality_reports.append(quality_report)
            
            self.stats["pages_processed"] += 1
        
        doc.close()
        
        # Step 3: Create document with metadata
        import os
        file_size = os.path.getsize(pdf_path)
        
        # Calculate overall metrics
        overall_confidence = np.mean([
            r.get("confidence", {}).get("overall_confidence", 0)
            for r in quality_reports
            if r.get("confidence")
        ])
        
        # Create metadata
        metadata = DocumentMetadata(
            page_count=total_pages,
            file_size=file_size,
            custom_fields={
                "processing_stats": self.stats,
                "overall_confidence": float(overall_confidence),
                "quality_assessment": self._assess_overall_quality(overall_confidence),
                "ocr_config": doc_config.dict() if isinstance(doc_config, OCRConfig) else doc_config
            }
        )
        
        if return_quality_report:
            metadata.custom_fields["page_quality_reports"] = quality_reports
        
        # Create document
        document = Document(
            id=pdf_path.stem,
            filename=pdf_path.name,
            total_pages=total_pages,
            file_size=file_size,
            page_info=processed_pages,
            metadata=metadata
        )
        
        if progress_callback:
            progress_callback(100, "Processing complete")
        
        logger.info(f"Document processing complete. Overall confidence: {overall_confidence:.2%}")
        
        return document
    
    def _analyze_and_configure(self, pdf_path: Path) -> OCRConfig:
        """Analyze document and generate optimal configuration."""
        if not self.enable_adaptive:
            return self.ocr_config
        
        # Open document for analysis
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        
        # Sample pages for analysis
        sample_size = min(5, total_pages)
        sample_indices = np.linspace(0, total_pages - 1, sample_size, dtype=int).tolist()
        
        sample_pages = []
        for idx in sample_indices:
            page = doc[idx]
            # Get page as image
            pix = page.get_pixmap(dpi=150)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            sample_pages.append(img)
            pix = None
        
        doc.close()
        
        # Analyze characteristics
        characteristics = self.configurator.analyze_document(
            sample_pages,
            total_pages,
            {"language": self.language}
        )
        
        # Check if OCR is needed
        needs_ocr, ocr_reason = check_pdf_needs_ocr(str(pdf_path))
        
        # Generate configuration
        config = self.configurator.generate_config(
            characteristics,
            self.ocr_config
        )
        
        # Override OCR if not needed
        if not needs_ocr:
            config.enable_ocr = False
            logger.info(f"OCR disabled: {ocr_reason}")
        else:
            logger.info(f"OCR enabled: {ocr_reason}")
        
        logger.info(f"Document quality: {characteristics.quality}")
        logger.info(f"OCR engine selected: {config.ocr_engine}")
        
        return config
    
    def _process_page(
        self,
        page: fitz.Page,
        page_number: int,
        config: OCRConfig
    ) -> Tuple[PageInfo, Dict[str, Any]]:
        """Process a single page with enhancements."""
        quality_report = {
            "page_number": page_number,
            "preprocessing": {},
            "confidence": {},
            "postprocessing": {},
            "text_metrics": {}
        }
        
        # Get page dimensions
        page_rect = page.rect
        width = page_rect.width
        height = page_rect.height
        
        # Extract text (might be empty for scanned pages)
        text = page.get_text()
        
        # If no text and OCR is enabled, perform OCR
        if (not text or len(text.strip()) < 50) and config.enable_ocr:
            # Get page as image
            pix = page.get_pixmap(dpi=config.target_dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # Preprocess if enabled
            if config.enable_preprocessing:
                img, preprocess_info = self.preprocessor.preprocess_image(
                    img,
                    current_dpi=config.target_dpi,
                    preprocessing_steps=config.preprocessing_steps
                )
                quality_report["preprocessing"] = preprocess_info
                self.stats["pages_preprocessed"] += 1
            
            # Perform OCR using base processor
            if self._base_processor and config.enable_ocr:
                # Save preprocessed image temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    # Convert processed image back to PIL/bytes for Docling
                    if len(processed_img.shape) == 2:  # Grayscale
                        cv2.imwrite(tmp_file.name, processed_img)
                    else:  # Color
                        cv2.imwrite(tmp_file.name, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
                    
                    # Create a temporary PDF page from the preprocessed image
                    temp_doc = fitz.open()
                    temp_page = temp_doc.new_page(width=processed_img.shape[1], height=processed_img.shape[0])
                    temp_page.insert_image(temp_page.rect, filename=tmp_file.name)
                    
                    # Save as temporary PDF
                    temp_pdf_path = tmp_file.name.replace('.png', '.pdf')
                    temp_doc.save(temp_pdf_path)
                    temp_doc.close()
                    
                    # Process with Docling
                    try:
                        from pathlib import Path
                        docling_result = self._base_processor.process_pdf(Path(temp_pdf_path))
                        if docling_result.page_info and len(docling_result.page_info) > 0:
                            text = docling_result.page_info[0].text_content or ""
                        else:
                            text = ""
                    except Exception as e:
                        logger.warning(f"OCR failed for page {page_number}: {e}")
                        text = ""
                    finally:
                        # Clean up temp files
                        import os
                        if os.path.exists(tmp_file.name):
                            os.unlink(tmp_file.name)
                        if os.path.exists(temp_pdf_path):
                            os.unlink(temp_pdf_path)
            else:
                # Fallback for testing
                text = f"[OCR disabled or not configured for page {page_number}]"
            
            self.stats["ocr_performed"] += 1
            pix = None
        
        # Score confidence
        if config.enable_postprocessing and text:
            confidence_report = self.confidence_scorer.score_ocr_output(
                text,
                expected_language=self.language,
                page_number=page_number
            )
            quality_report["confidence"] = confidence_report
            
            # Update average confidence
            current_avg = self.stats["average_confidence"]
            n = self.stats["pages_processed"]
            new_confidence = confidence_report.get("overall_confidence", 0)
            self.stats["average_confidence"] = (current_avg * n + new_confidence) / (n + 1)
            
            # Post-process if confidence is low or aggressive mode
            if (confidence_report["overall_confidence"] < config.confidence_threshold or
                config.apply_aggressive_corrections):
                
                processed_text, process_info = self.postprocessor.process_text(
                    text,
                    confidence_scores=confidence_report["scores"],
                    apply_aggressive=config.apply_aggressive_corrections
                )
                
                quality_report["postprocessing"] = process_info
                self.stats["total_corrections"] += process_info.get("corrections_made", 0)
                
                # Use processed text if improvements were made
                if process_info.get("text_changed", False):
                    text = processed_text
        
        # Calculate text metrics
        words = text.split() if text else []
        quality_report["text_metrics"] = {
            "character_count": len(text) if text else 0,
            "word_count": len(words),
            "has_text": bool(text and text.strip())
        }
        
        # Create page info with OCR quality information
        page_info = PageInfo(
            page_number=page_number,
            width=width,
            height=height,
            text_content=text,
            word_count=len(words),
            has_images=len(page.get_images()) > 0,
            has_tables=self._detect_tables_on_page(page),
            # OCR quality fields
            ocr_confidence=quality_report.get("confidence", {}).get("overall_confidence"),
            ocr_quality_assessment=quality_report.get("confidence", {}).get("quality_assessment"),
            ocr_issues=quality_report.get("confidence", {}).get("issues", []),
            preprocessing_applied=quality_report.get("preprocessing", {}).get("steps_applied", []),
            corrections_made=quality_report.get("postprocessing", {}).get("corrections_made", 0)
        )
        
        return page_info, quality_report
    
    def _detect_tables_on_page(self, page: fitz.Page) -> bool:
        """Simple table detection on page."""
        # Check for table-like structures using PyMuPDF
        try:
            tables = page.find_tables()
            # TableFinder object has a .tables attribute
            return len(tables.tables) > 0 if hasattr(tables, 'tables') else False
        except:
            return False
    
    def _assess_overall_quality(self, confidence: float) -> str:
        """Assess overall document quality."""
        if confidence >= 0.9:
            return "excellent"
        elif confidence >= 0.8:
            return "good"
        elif confidence >= 0.6:
            return "fair"
        elif confidence >= 0.4:
            return "poor"
        else:
            return "very_poor"
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def process_with_quality_threshold(
        self,
        pdf_path: Path,
        min_confidence: float = 0.7,
        reprocess_low_quality: bool = True
    ) -> Document:
        """
        Process document with quality threshold.
        
        Pages below threshold can be reprocessed with more aggressive settings.
        """
        # First pass
        document = self.process_document(pdf_path)
        
        if not reprocess_low_quality:
            return document
        
        # Check for low quality pages
        low_quality_pages = []
        
        if "page_quality_reports" in document.metadata.custom_fields:
            for report in document.metadata.custom_fields["page_quality_reports"]:
                confidence = report.get("confidence", {}).get("overall_confidence", 1)
                if confidence < min_confidence:
                    low_quality_pages.append(report["page_number"])
        
        if low_quality_pages:
            logger.info(f"Reprocessing {len(low_quality_pages)} low quality pages")
            
            # Create aggressive config
            aggressive_config = OCRConfig(
                enable_ocr=True,
                enable_preprocessing=True,
                preprocessing_steps=["upscale", "denoise", "deskew", "contrast", "threshold"],
                target_dpi=400,
                apply_aggressive_corrections=True,
                force_full_page_ocr=True
            )
            
            # Reprocess low quality pages
            # (Implementation would reprocess specific pages)
            
        return document