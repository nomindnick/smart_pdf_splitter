"""Enhanced document processor v2 with proper OCR integration."""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2
import fitz  # PyMuPDF

from .document_processor import DocumentProcessor
from .models import PageInfo, Document, DocumentMetadata
from .ocr_preprocessor import OCRPreprocessor
from .ocr_confidence_scorer import OCRConfidenceScorer
from .ocr_postprocessor import OCRPostProcessor
from .ocr_config import OCRConfig, AdaptiveOCRConfigurator
from .ocr_optimizer import check_pdf_needs_ocr

logger = logging.getLogger(__name__)


class EnhancedDocumentProcessorV2:
    """
    Enhanced document processor that properly integrates with Docling OCR.
    
    This version:
    1. Uses Docling for actual OCR execution
    2. Applies preprocessing before Docling processes the document
    3. Applies confidence scoring and post-processing after OCR
    """
    
    def __init__(
        self,
        ocr_config: Optional[OCRConfig] = None,
        enable_adaptive: bool = True,
        language: str = "en"
    ):
        """Initialize enhanced processor."""
        self.ocr_config = ocr_config or OCRConfig()
        self.enable_adaptive = enable_adaptive
        self.language = language
        
        # Initialize enhancement components
        self.preprocessor = OCRPreprocessor(
            target_dpi=self.ocr_config.target_dpi,
            enable_gpu=self.ocr_config.enable_gpu
        )
        self.confidence_scorer = OCRConfidenceScorer(language=language)
        self.postprocessor = OCRPostProcessor(language=language)
        self.configurator = AdaptiveOCRConfigurator()
        
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
        """Process document with full enhancement pipeline."""
        logger.info(f"Processing document with enhancements: {pdf_path}")
        
        # Reset stats
        self.stats = {
            "pages_processed": 0,
            "pages_preprocessed": 0,
            "ocr_performed": 0,
            "average_confidence": 0.0,
            "total_corrections": 0
        }
        
        # Step 1: Analyze and configure
        if progress_callback:
            progress_callback(0, "Analyzing document...")
        
        # Check if OCR is needed
        needs_ocr, ocr_reason = check_pdf_needs_ocr(str(pdf_path))
        logger.info(f"OCR needed: {needs_ocr} - {ocr_reason}")
        
        # Get optimal configuration
        config = self._analyze_and_configure(pdf_path)
        
        # Override OCR setting based on detection
        if not needs_ocr:
            config.enable_ocr = False
            logger.info("OCR disabled - document has embedded text")
        
        # Step 2: Preprocess if needed
        preprocessed_pdf_path = pdf_path
        if config.enable_ocr and config.enable_preprocessing:
            if progress_callback:
                progress_callback(10, "Preprocessing document...")
            preprocessed_pdf_path = self._preprocess_document(pdf_path, config)
        
        # Step 3: Process with Docling
        if progress_callback:
            progress_callback(30, "Performing OCR with Docling...")
        
        # Initialize document processor with our config
        doc_processor = DocumentProcessor(
            enable_ocr=config.enable_ocr,
            ocr_languages=config.ocr_languages,
            ocr_engine=config.ocr_engine,
            page_batch_size=config.page_batch_size,
            auto_detect_ocr=False  # We already checked
        )
        
        # Process with Docling
        docling_result = doc_processor.process_pdf(preprocessed_pdf_path)
        
        # Step 4: Apply confidence scoring and post-processing
        if progress_callback:
            progress_callback(70, "Analyzing OCR quality...")
        
        enhanced_pages = []
        quality_reports = []
        
        for i, page_info in enumerate(docling_result.page_info):
            self.stats["pages_processed"] += 1
            
            # Score confidence
            quality_report = {"page_number": i + 1}
            
            if config.enable_ocr and page_info.text_content:
                # Score the OCR output
                confidence_report = self.confidence_scorer.score_ocr_output(
                    page_info.text_content,
                    expected_language=self.language,
                    page_number=i + 1
                )
                quality_report["confidence"] = confidence_report
                
                # Update average confidence
                current_avg = self.stats["average_confidence"]
                n = self.stats["pages_processed"] - 1
                new_confidence = confidence_report.get("overall_confidence", 0)
                self.stats["average_confidence"] = (current_avg * n + new_confidence) / (n + 1)
                
                # Post-process if needed
                if (confidence_report["overall_confidence"] < config.confidence_threshold or
                    config.apply_aggressive_corrections):
                    
                    processed_text, process_info = self.postprocessor.process_text(
                        page_info.text_content,
                        confidence_scores=confidence_report["scores"],
                        apply_aggressive=config.apply_aggressive_corrections
                    )
                    
                    quality_report["postprocessing"] = process_info
                    self.stats["total_corrections"] += process_info.get("corrections_made", 0)
                    
                    # Update text if improved
                    if process_info.get("text_changed", False):
                        page_info.text_content = processed_text
                
                # Update page info with OCR quality data
                page_info.ocr_confidence = confidence_report.get("overall_confidence")
                page_info.ocr_quality_assessment = confidence_report.get("quality_assessment")
                page_info.ocr_issues = confidence_report.get("issues", [])
                page_info.corrections_made = quality_report.get("postprocessing", {}).get("corrections_made", 0)
            
            enhanced_pages.append(page_info)
            quality_reports.append(quality_report)
        
        # Update document with enhanced data
        docling_result.page_info = enhanced_pages
        
        # Add processing metadata
        if docling_result.metadata:
            docling_result.metadata.custom_fields.update({
                "processing_stats": self.stats,
                "overall_confidence": float(self.stats["average_confidence"]),
                "quality_assessment": self._assess_overall_quality(self.stats["average_confidence"]),
                "ocr_config": config.model_dump() if hasattr(config, 'model_dump') else config.dict()
            })
            
            if return_quality_report:
                docling_result.metadata.custom_fields["page_quality_reports"] = quality_reports
        
        # Clean up preprocessed file if different from original
        if preprocessed_pdf_path != pdf_path and preprocessed_pdf_path.exists():
            preprocessed_pdf_path.unlink()
        
        if progress_callback:
            progress_callback(100, "Processing complete")
        
        logger.info(f"Document processing complete. Overall confidence: {self.stats['average_confidence']:.2%}")
        
        return docling_result
    
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
        
        # Generate configuration
        config = self.configurator.generate_config(
            characteristics,
            self.ocr_config
        )
        
        logger.info(f"Document quality: {characteristics.quality}")
        logger.info(f"OCR engine selected: {config.ocr_engine}")
        
        return config
    
    def _preprocess_document(self, pdf_path: Path, config: OCRConfig) -> Path:
        """Preprocess entire document and save as new PDF."""
        import tempfile
        
        # Create temporary file for preprocessed PDF
        temp_dir = Path(tempfile.gettempdir())
        preprocessed_path = temp_dir / f"preprocessed_{pdf_path.stem}.pdf"
        
        # Open documents
        src_doc = fitz.open(str(pdf_path))
        dst_doc = fitz.open()
        
        logger.info(f"Preprocessing {len(src_doc)} pages...")
        
        for page_num in range(len(src_doc)):
            page = src_doc[page_num]
            
            # Get page as image
            pix = page.get_pixmap(dpi=config.target_dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            
            # Convert color space if needed
            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif pix.n == 1:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Preprocess
            processed_img, info = self.preprocessor.preprocess_image(
                img,
                current_dpi=config.target_dpi,
                preprocessing_steps=config.preprocessing_steps
            )
            
            if len(info["steps_applied"]) > 1:  # More than just grayscale conversion
                self.stats["pages_preprocessed"] += 1
            
            # Create new page with preprocessed image
            new_page = dst_doc.new_page(width=page.rect.width, height=page.rect.height)
            
            # Save processed image temporarily
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                if len(processed_img.shape) == 2:  # Grayscale
                    cv2.imwrite(tmp_file.name, processed_img)
                else:  # Color
                    cv2.imwrite(tmp_file.name, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
                
                # Insert into new page
                new_page.insert_image(new_page.rect, filename=tmp_file.name)
                
                # Clean up temp file
                Path(tmp_file.name).unlink()
            
            pix = None
        
        # Save preprocessed document
        dst_doc.save(str(preprocessed_path))
        dst_doc.close()
        src_doc.close()
        
        logger.info(f"Preprocessing complete. {self.stats['pages_preprocessed']} pages enhanced.")
        
        return preprocessed_path
    
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