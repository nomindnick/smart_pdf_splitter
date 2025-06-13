"""Unified document processor combining all features from previous implementations."""

import logging
import time
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
import tempfile

import cv2
import fitz  # PyMuPDF
import numpy as np
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    TesseractOcrOptions,
    TesseractCliOcrOptions
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption

from .models import Document, DocumentMetadata, PageInfo
from .ocr_optimizer import check_pdf_needs_ocr, check_page_needs_ocr
from .ocr_preprocessor import OCRPreprocessor
from .ocr_confidence_scorer import OCRConfidenceScorer
from .ocr_postprocessor import OCRPostProcessor
from .ocr_config import OCRConfig, AdaptiveOCRConfigurator

logger = logging.getLogger(__name__)

# Suppress Docling's verbose output - but don't override settings.debug


class ProcessingMode(str, Enum):
    """Document processing modes."""
    BASIC = "basic"        # Direct Docling processing
    ENHANCED = "enhanced"  # With preprocessing and quality improvements
    SMART = "smart"       # Per-page OCR decisions with resource limits


class UnifiedDocumentProcessor:
    """
    Unified document processor combining all features.
    
    Features:
    - Multiple processing modes (basic, enhanced, smart)
    - Docling integration for OCR and parsing
    - Image preprocessing for better OCR
    - Confidence scoring and quality assessment
    - Text post-processing and corrections
    - Smart per-page OCR decisions
    - Resource-aware processing
    """
    
    def __init__(
        self,
        mode: ProcessingMode = ProcessingMode.SMART,
        ocr_config: Optional[OCRConfig] = None,
        enable_adaptive: bool = True,
        language: str = "en",
        max_ocr_pages: Optional[int] = None,
        max_memory_mb: int = 4096
    ):
        """
        Initialize unified processor.
        
        Args:
            mode: Processing mode (basic, enhanced, smart)
            ocr_config: OCR configuration
            enable_adaptive: Enable adaptive configuration based on document analysis
            language: Primary language for processing
            max_ocr_pages: Maximum pages to OCR (None = no limit)
            max_memory_mb: Maximum memory usage in MB
        """
        self.mode = ProcessingMode(mode)
        self.ocr_config = ocr_config or OCRConfig()
        self.enable_adaptive = enable_adaptive
        self.language = language
        self.max_ocr_pages = max_ocr_pages
        self.max_memory_mb = max_memory_mb
        
        # Initialize enhancement components for enhanced/smart modes
        if self.mode != ProcessingMode.BASIC:
            self.preprocessor = OCRPreprocessor(
                target_dpi=self.ocr_config.target_dpi,
                enable_gpu=self.ocr_config.enable_gpu
            )
            self.confidence_scorer = OCRConfidenceScorer(language=language)
            self.postprocessor = OCRPostProcessor(language=language)
            self.configurator = AdaptiveOCRConfigurator()
        
        # Initialize Docling converter
        self._init_converter()
        
        # Processing statistics
        self.stats = self._reset_stats()
    
    def _reset_stats(self) -> Dict[str, Any]:
        """Reset processing statistics."""
        return {
            "pages_processed": 0,
            "pages_preprocessed": 0,
            "ocr_performed": 0,
            "pages_skipped": 0,
            "average_confidence": 0.0,
            "total_corrections": 0,
            "processing_time": 0.0
        }
    
    def _init_converter(self, config: Optional[OCRConfig] = None):
        """Initialize Docling converter with current configuration."""
        config = config or self.ocr_config
        
        # Create pipeline options
        pipeline_options = PdfPipelineOptions()
        
        # OCR configuration
        if config.enable_ocr:
            pipeline_options.do_ocr = True
            # Create appropriate OCR options based on engine
            if config.ocr_engine == "easyocr":
                pipeline_options.ocr_options = EasyOcrOptions(
                    force_full_page_ocr=config.force_full_page_ocr,
                    bitmap_area_threshold=config.bitmap_area_threshold,
                    lang=config.ocr_languages,
                    use_gpu=config.use_gpu_if_available
                )
            elif config.ocr_engine == "tesseract":
                pipeline_options.ocr_options = TesseractOcrOptions(
                    force_full_page_ocr=config.force_full_page_ocr,
                    bitmap_area_threshold=config.bitmap_area_threshold,
                    lang=config.ocr_languages
                )
            elif config.ocr_engine == "tesseract-cli":
                pipeline_options.ocr_options = TesseractCliOcrOptions(
                    force_full_page_ocr=config.force_full_page_ocr,
                    bitmap_area_threshold=config.bitmap_area_threshold,
                    lang=config.ocr_languages
                )
        else:
            pipeline_options.do_ocr = False
        
        # Performance settings
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 1.0
        
        # Initialize converter
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
    
    def process_document(
        self,
        pdf_source: Union[Path, str, bytes, BytesIO],
        progress_callback: Optional[callable] = None,
        return_quality_report: bool = True
    ) -> Document:
        """
        Process document with selected mode.
        
        Args:
            pdf_source: PDF file path, bytes, or BytesIO stream
            progress_callback: Optional callback for progress updates
            return_quality_report: Include quality analysis in metadata
            
        Returns:
            Processed Document with text, boundaries, and metadata
        """
        start_time = time.time()
        self.stats = self._reset_stats()
        
        # Convert source to Path if needed
        pdf_path = self._prepare_source(pdf_source)
        
        logger.info(f"Processing document in {self.mode} mode: {pdf_path}")
        
        # Route to appropriate processor
        if self.mode == ProcessingMode.BASIC:
            result = self._process_basic(pdf_path, progress_callback)
        elif self.mode == ProcessingMode.ENHANCED:
            result = self._process_enhanced(pdf_path, progress_callback, return_quality_report)
        else:  # SMART mode
            result = self._process_smart(pdf_path, progress_callback, return_quality_report)
        
        # Update stats
        self.stats["processing_time"] = time.time() - start_time
        
        # Add stats to metadata
        if result.metadata:
            result.metadata.custom_fields["processing_stats"] = self.stats
            result.metadata.custom_fields["processing_mode"] = self.mode
        
        logger.info(
            f"Processing complete in {self.stats['processing_time']:.2f}s. "
            f"Pages: {result.total_pages}, OCR: {self.stats['ocr_performed']}"
        )
        
        return result
    
    def _prepare_source(self, pdf_source: Union[Path, str, bytes, BytesIO]) -> Path:
        """Prepare PDF source for processing."""
        if isinstance(pdf_source, (str, Path)):
            return Path(pdf_source)
        
        # Handle bytes/BytesIO
        if isinstance(pdf_source, bytes):
            pdf_bytes = pdf_source
        else:
            pdf_bytes = pdf_source.read()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            return Path(tmp_file.name)
    
    def _process_basic(self, pdf_path: Path, progress_callback: Optional[callable]) -> Document:
        """Basic processing using Docling directly."""
        if progress_callback:
            progress_callback(0, "Processing with Docling...")
        
        # Process with Docling
        docling_output = self.converter.convert(pdf_path)
        
        # Convert to our Document model
        document = self._convert_docling_output(docling_output, pdf_path)
        
        if progress_callback:
            progress_callback(100, "Processing complete")
        
        return document
    
    def _process_enhanced(
        self,
        pdf_path: Path,
        progress_callback: Optional[callable],
        return_quality_report: bool
    ) -> Document:
        """Enhanced processing with preprocessing and quality improvements."""
        # Step 1: Analyze and configure
        if progress_callback:
            progress_callback(0, "Analyzing document...")
        
        config = self._analyze_and_configure(pdf_path)
        
        # Check if OCR is needed
        needs_ocr, ocr_reason = check_pdf_needs_ocr(str(pdf_path))
        if not needs_ocr:
            config.enable_ocr = False
            logger.info(f"OCR disabled: {ocr_reason}")
        
        # Re-initialize converter with new config
        self._init_converter(config)
        
        # Step 2: Preprocess if needed
        preprocessed_path = pdf_path
        if config.enable_ocr and config.enable_preprocessing:
            if progress_callback:
                progress_callback(10, "Preprocessing document...")
            preprocessed_path = self._preprocess_document(pdf_path, config)
        
        # Step 3: Process with Docling
        if progress_callback:
            progress_callback(30, "Performing OCR...")
        
        docling_output = self.converter.convert(preprocessed_path)
        document = self._convert_docling_output(docling_output, pdf_path)
        
        # Step 4: Apply enhancements
        if progress_callback:
            progress_callback(70, "Analyzing OCR quality...")
        
        quality_reports = []
        for page_info in document.page_info:
            quality_report = self._enhance_page(page_info, config)
            quality_reports.append(quality_report)
        
        # Add quality data to metadata
        if return_quality_report and document.metadata:
            document.metadata.custom_fields["page_quality_reports"] = quality_reports
            document.metadata.custom_fields["overall_confidence"] = self.stats["average_confidence"]
            document.metadata.custom_fields["quality_assessment"] = self._assess_overall_quality(
                self.stats["average_confidence"]
            )
        
        # Clean up preprocessed file
        if preprocessed_path != pdf_path and preprocessed_path.exists():
            preprocessed_path.unlink()
        
        if progress_callback:
            progress_callback(100, "Processing complete")
        
        return document
    
    def _process_smart(
        self,
        pdf_path: Path,
        progress_callback: Optional[callable],
        return_quality_report: bool
    ) -> Document:
        """Smart processing with per-page OCR decisions."""
        # Step 1: Analyze document
        if progress_callback:
            progress_callback(0, "Analyzing document structure...")
        
        doc_analysis = self.analyze_document(pdf_path)
        config = self._analyze_and_configure(pdf_path)
        
        # Step 2: Process pages based on analysis
        doc = fitz.open(str(pdf_path))
        processed_pages = []
        quality_reports = []
        ocr_pages_count = 0
        
        for i, page in enumerate(doc):
            if progress_callback:
                progress = (i / len(doc)) * 90
                progress_callback(progress, f"Processing page {i+1}/{len(doc)}")
            
            # Check if we should OCR this page
            needs_ocr, reason = check_page_needs_ocr(page)
            
            # Respect OCR page limit
            if needs_ocr and self.max_ocr_pages and ocr_pages_count >= self.max_ocr_pages:
                logger.info(f"Skipping OCR for page {i+1}: OCR limit reached")
                needs_ocr = False
                self.stats["pages_skipped"] += 1
            
            if needs_ocr:
                # Process with OCR
                page_info, quality_report = self._process_page_with_ocr(
                    page, i + 1, config, pdf_path
                )
                ocr_pages_count += 1
                self.stats["ocr_performed"] += 1
            else:
                # Extract text without OCR
                page_info = self._extract_page_text(page, i + 1)
                quality_report = {"page_number": i + 1, "ocr_skipped": True, "reason": reason}
            
            processed_pages.append(page_info)
            quality_reports.append(quality_report)
            self.stats["pages_processed"] += 1
        
        doc.close()
        
        # Create document
        document = self._create_document(pdf_path, processed_pages, doc_analysis)
        
        # Add quality data
        if return_quality_report and document.metadata:
            document.metadata.custom_fields["page_quality_reports"] = quality_reports
            document.metadata.custom_fields["document_analysis"] = doc_analysis
        
        if progress_callback:
            progress_callback(100, "Processing complete")
        
        return document
    
    def _analyze_and_configure(self, pdf_path: Path) -> OCRConfig:
        """Analyze document and generate optimal configuration."""
        if not self.enable_adaptive:
            return self.ocr_config
        
        # Get sample pages
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        
        sample_size = min(5, total_pages)
        sample_indices = np.linspace(0, total_pages - 1, sample_size, dtype=int).tolist()
        
        sample_pages = []
        for idx in sample_indices:
            page = doc[idx]
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
            sample_pages, total_pages, {"language": self.language}
        )
        
        # Generate configuration
        config = self.configurator.generate_config(characteristics, self.ocr_config)
        
        logger.info(f"Document quality: {characteristics.quality}, OCR engine: {config.ocr_engine}")
        
        return config
    
    def _preprocess_document(self, pdf_path: Path, config: OCRConfig) -> Path:
        """Preprocess entire document for better OCR."""
        temp_path = Path(tempfile.gettempdir()) / f"preprocessed_{pdf_path.stem}.pdf"
        
        src_doc = fitz.open(str(pdf_path))
        dst_doc = fitz.open()
        
        for page_num in range(len(src_doc)):
            page = src_doc[page_num]
            
            # Get page as image
            pix = page.get_pixmap(dpi=config.target_dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            
            # Preprocess
            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            processed_img, info = self.preprocessor.preprocess_image(
                img,
                current_dpi=config.target_dpi,
                preprocessing_steps=config.preprocessing_steps
            )
            
            if len(info["steps_applied"]) > 1:
                self.stats["pages_preprocessed"] += 1
            
            # Create new page
            new_page = dst_doc.new_page(width=page.rect.width, height=page.rect.height)
            
            # Save and insert image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                if len(processed_img.shape) == 2:
                    cv2.imwrite(tmp_file.name, processed_img)
                else:
                    cv2.imwrite(tmp_file.name, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
                
                new_page.insert_image(new_page.rect, filename=tmp_file.name)
                Path(tmp_file.name).unlink()
            
            pix = None
        
        dst_doc.save(str(temp_path))
        dst_doc.close()
        src_doc.close()
        
        return temp_path
    
    def _enhance_page(self, page_info: PageInfo, config: OCRConfig) -> Dict[str, Any]:
        """Apply enhancements to a page."""
        quality_report = {"page_number": page_info.page_number}
        
        if not page_info.text_content:
            return quality_report
        
        # Score confidence
        confidence_report = self.confidence_scorer.score_ocr_output(
            page_info.text_content,
            expected_language=self.language,
            page_number=page_info.page_number
        )
        quality_report["confidence"] = confidence_report
        
        # Update stats
        current_avg = self.stats["average_confidence"]
        n = self.stats["pages_processed"]
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
            
            if process_info.get("text_changed", False):
                page_info.text_content = processed_text
        
        # Update page info with quality data
        page_info.ocr_confidence = confidence_report.get("overall_confidence")
        page_info.ocr_quality_assessment = confidence_report.get("quality_assessment")
        page_info.ocr_issues = confidence_report.get("issues", [])
        page_info.corrections_made = quality_report.get("postprocessing", {}).get("corrections_made", 0)
        
        return quality_report
    
    def _process_page_with_ocr(
        self,
        page: fitz.Page,
        page_number: int,
        config: OCRConfig,
        pdf_path: Path
    ) -> Tuple[PageInfo, Dict[str, Any]]:
        """Process a single page with OCR."""
        # Extract page as temporary PDF
        temp_doc = fitz.open()
        temp_doc.insert_pdf(fitz.open(str(pdf_path)), from_page=page_number-1, to_page=page_number-1)
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            temp_doc.save(tmp_file.name)
            temp_doc.close()
            
            # Process with Docling
            try:
                docling_output = self.converter.convert(Path(tmp_file.name))
                
                # Extract page info
                if docling_output.pages:
                    text = docling_output.pages[0].text if docling_output.pages[0].text else ""
                else:
                    text = ""
                
                page_info = PageInfo(
                    page_number=page_number,
                    width=page.rect.width,
                    height=page.rect.height,
                    text_content=text,
                    word_count=len(text.split()) if text else 0,
                    has_images=len(page.get_images()) > 0,
                    has_tables=self._detect_tables(page)
                )
                
                # Apply enhancements
                quality_report = self._enhance_page(page_info, config)
                
            finally:
                Path(tmp_file.name).unlink()
        
        return page_info, quality_report
    
    def _extract_page_text(self, page: fitz.Page, page_number: int) -> PageInfo:
        """Extract text from page without OCR."""
        text = page.get_text()
        
        return PageInfo(
            page_number=page_number,
            width=page.rect.width,
            height=page.rect.height,
            text_content=text,
            word_count=len(text.split()) if text else 0,
            has_images=len(page.get_images()) > 0,
            has_tables=self._detect_tables(page)
        )
    
    def _detect_tables(self, page: fitz.Page) -> bool:
        """Detect if page has tables."""
        try:
            tables = page.find_tables()
            return len(tables.tables) > 0 if hasattr(tables, 'tables') else False
        except:
            return False
    
    def _convert_docling_output(self, docling_output: Any, pdf_path: Path) -> Document:
        """Convert Docling output to our Document model."""
        # Extract pages
        pages = []
        for i, page in enumerate(docling_output.pages):
            page_info = PageInfo(
                page_number=i + 1,
                width=page.size.width if hasattr(page, 'size') else 0,
                height=page.size.height if hasattr(page, 'size') else 0,
                text_content=page.text if hasattr(page, 'text') else "",
                word_count=len(page.text.split()) if hasattr(page, 'text') and page.text else 0,
                has_images=False,  # Would need additional detection
                has_tables=False   # Would need additional detection
            )
            pages.append(page_info)
            self.stats["pages_processed"] += 1
        
        # Create metadata
        import os
        file_size = os.path.getsize(pdf_path)
        
        metadata = DocumentMetadata(
            page_count=len(pages),
            file_size=file_size,
            custom_fields={}
        )
        
        # Create document
        return Document(
            id=pdf_path.stem,
            filename=pdf_path.name,
            total_pages=len(pages),
            file_size=file_size,
            page_info=pages,
            metadata=metadata
        )
    
    def _create_document(
        self,
        pdf_path: Path,
        pages: List[PageInfo],
        doc_analysis: Dict[str, Any]
    ) -> Document:
        """Create document from processed pages."""
        import os
        file_size = os.path.getsize(pdf_path)
        
        metadata = DocumentMetadata(
            page_count=len(pages),
            file_size=file_size,
            custom_fields={
                "document_type": doc_analysis.get("document_type", "unknown"),
                "has_ocr_content": doc_analysis.get("has_ocr_content", False)
            }
        )
        
        return Document(
            id=pdf_path.stem,
            filename=pdf_path.name,
            total_pages=len(pages),
            file_size=file_size,
            page_info=pages,
            metadata=metadata
        )
    
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
    
    def analyze_document(self, pdf_path: Path) -> Dict[str, Any]:
        """Analyze document structure and content."""
        doc = fitz.open(str(pdf_path))
        
        total_pages = len(doc)
        text_pages = 0
        image_pages = 0
        mixed_pages = 0
        empty_pages = 0
        
        for page in doc:
            text = page.get_text()
            images = page.get_images()
            
            has_text = len(text.strip()) > 50
            has_images = len(images) > 0
            
            if not has_text and not has_images:
                empty_pages += 1
            elif has_text and not has_images:
                text_pages += 1
            elif not has_text and has_images:
                image_pages += 1
            else:
                mixed_pages += 1
        
        doc.close()
        
        # Determine document type
        if image_pages > total_pages * 0.8:
            doc_type = "image-based"
        elif text_pages > total_pages * 0.8:
            doc_type = "text-based"
        else:
            doc_type = "mixed"
        
        return {
            "total_pages": total_pages,
            "text_pages": text_pages,
            "image_pages": image_pages,
            "mixed_pages": mixed_pages,
            "empty_pages": empty_pages,
            "document_type": doc_type,
            "has_ocr_content": image_pages > 0
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()