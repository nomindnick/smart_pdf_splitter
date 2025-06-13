"""
Hybrid boundary detector combining text-based and visual boundary detection.

This module provides the main interface for boundary detection, combining
signals from both text analysis and visual features for improved accuracy.
"""

import logging
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path

from .boundary_detector import BoundaryDetector, BoundaryCandidate
from .visual_boundary_detector import VisualBoundaryDetector, VisualBoundaryCandidate
from .unified_document_processor import UnifiedDocumentProcessor, ProcessingMode
from .intelligent_ocr_strategy import IntelligentOCRStrategy
from .pipeline_config import PipelineProfiles
from .ocr_config import OCRConfig
from .models import (
    PageInfo, 
    PageVisualInfo, 
    Boundary, 
    Signal, 
    SignalType,
    VisualSignalType,
    VisualSignal,
    DocumentType
)

logger = logging.getLogger(__name__)


class VisualProcessingConfig:
    """Configuration for visual processing features."""
    
    def __init__(
        self,
        enable_visual_features: bool = True,
        enable_picture_classification: bool = True,
        enable_vlm: bool = False,
        enable_llm: bool = False,
        visual_memory_limit_mb: int = 2048,
        max_image_dimension: int = 1024,
        page_image_resolution: int = 150,
        visual_batch_size: int = 2,
        visual_confidence_threshold: float = 0.5,
        vlm_model: str = "docling-vlm-v1",
        vlm_prompt_template: str = "Analyze if this page transition represents a document boundary.",
        enable_intelligent_ocr: bool = True,
        llm_confidence_threshold: float = 0.7
    ):
        self.enable_visual_features = enable_visual_features
        self.enable_picture_classification = enable_picture_classification
        self.enable_vlm = enable_vlm
        self.enable_llm = enable_llm
        self.visual_memory_limit_mb = visual_memory_limit_mb
        self.max_image_dimension = max_image_dimension
        self.page_image_resolution = page_image_resolution
        self.visual_batch_size = visual_batch_size
        self.visual_confidence_threshold = visual_confidence_threshold
        self.vlm_model = vlm_model
        self.vlm_prompt_template = vlm_prompt_template
        self.enable_intelligent_ocr = enable_intelligent_ocr
        self.llm_confidence_threshold = llm_confidence_threshold


class HybridBoundaryDetector:
    """
    Combines text-based and visual boundary detection for improved accuracy.
    """
    
    def __init__(
        self,
        config: Optional[VisualProcessingConfig] = None,
        text_weight: float = 0.6,
        visual_weight: float = 0.4,
        min_combined_confidence: float = 0.6
    ):
        """
        Initialize hybrid boundary detector.
        
        Args:
            config: Visual processing configuration
            text_weight: Weight for text-based signals
            visual_weight: Weight for visual signals
            min_combined_confidence: Minimum confidence for final boundaries
        """
        self.config = config or VisualProcessingConfig()
        self.text_weight = text_weight
        self.visual_weight = visual_weight
        self.min_combined_confidence = min_combined_confidence
        
        # Initialize detectors
        self.text_detector = BoundaryDetector()
        self.visual_detector = VisualBoundaryDetector(
            min_visual_confidence=self.config.visual_confidence_threshold,
            enable_vlm_analysis=self.config.enable_vlm
        ) if self.config.enable_visual_features else None
        
        # Initialize intelligent OCR strategy if enabled
        detection_methods = []
        if True:  # Text detection is always enabled
            detection_methods.append("text")
        if self.config.enable_visual_features:
            detection_methods.append("visual")
        if self.config.enable_llm:
            detection_methods.append("llm")
            
        self.ocr_strategy = IntelligentOCRStrategy(detection_methods) if self.config.enable_intelligent_ocr else None
        
        # Initialize processor with base config
        # Will be reconfigured based on intelligent strategy if enabled
        self.processor = UnifiedDocumentProcessor(
            mode=ProcessingMode.SMART if self.config.enable_visual_features else ProcessingMode.BASIC,
            enable_adaptive=True
        )
    
    def detect_boundaries(
        self,
        file_path: Path,
        use_visual: Optional[bool] = None,
        page_range: Optional[Tuple[int, int]] = None
    ) -> List[Boundary]:
        """
        Detect boundaries using hybrid approach with intelligent OCR.
        
        Args:
            file_path: Path to PDF file
            use_visual: Override config to enable/disable visual processing
            page_range: Optional page range to process
            
        Returns:
            List of detected boundaries with combined confidence
        """
        logger.info(f"Starting hybrid boundary detection for: {file_path}")
        
        # Process document
        if use_visual is None:
            use_visual = self.config.enable_visual_features
        
        # Use intelligent OCR strategy if enabled
        if self.ocr_strategy and self.config.enable_intelligent_ocr:
            return self._detect_with_intelligent_ocr(file_path, use_visual, page_range)
        
        # Fall back to standard processing
        if use_visual and self.visual_detector:
            # Process with visual features
            pages = list(self.processor.process_document_with_visual(file_path, page_range))
            return self._detect_hybrid_boundaries(pages)
        else:
            # Process text only
            pages = list(self.processor.process_document(file_path, page_range))
            page_infos = [PageInfo(**p.dict()) for p in pages]
            return self.text_detector.detect_boundaries(page_infos)
    
    def detect_boundaries_from_stream(
        self,
        pdf_bytes: bytes,
        filename: str = "document.pdf",
        use_visual: Optional[bool] = None
    ) -> List[Boundary]:
        """
        Detect boundaries from PDF bytes stream.
        
        Args:
            pdf_bytes: PDF content as bytes
            filename: Optional filename
            use_visual: Override config to enable/disable visual processing
            
        Returns:
            List of detected boundaries
        """
        logger.info(f"Starting hybrid boundary detection for stream: {filename}")
        
        if use_visual is None:
            use_visual = self.config.enable_visual_features
        
        if use_visual and self.visual_detector:
            # Process with visual features
            pages = list(self.processor.process_document_stream_with_visual(pdf_bytes, filename))
            return self._detect_hybrid_boundaries(pages)
        else:
            # Process text only
            pages = list(self.processor.process_document_stream(pdf_bytes, filename))
            return self.text_detector.detect_boundaries(pages)
    
    def _detect_hybrid_boundaries(
        self,
        pages: List[PageVisualInfo]
    ) -> List[Boundary]:
        """Combine text and visual boundary detection."""
        # Get text-based boundaries
        page_infos = [PageInfo(**p.dict()) for p in pages]
        text_boundaries = self.text_detector.detect_boundaries(page_infos)
        
        # Get visual boundaries
        visual_candidates = self.visual_detector.detect_visual_boundaries(pages)
        
        # Get visual summary for additional context
        visual_summary = self.processor.get_visual_summary(pages)
        
        # Combine results
        combined_boundaries = self._combine_boundaries(
            text_boundaries,
            visual_candidates,
            pages,
            visual_summary
        )
        
        return combined_boundaries
    
    def _combine_boundaries(
        self,
        text_boundaries: List[Boundary],
        visual_candidates: List[VisualBoundaryCandidate],
        pages: List[PageVisualInfo],
        visual_summary: Dict[str, Any]
    ) -> List[Boundary]:
        """Combine text and visual boundaries with weighted confidence."""
        # Create page number to boundary mapping
        boundary_map = {}
        
        # Add text boundaries
        for boundary in text_boundaries:
            if boundary.start_page not in boundary_map:
                boundary_map[boundary.start_page] = {
                    'text_boundary': boundary,
                    'visual_candidate': None,
                    'page': pages[boundary.start_page - 1] if boundary.start_page <= len(pages) else None
                }
            else:
                boundary_map[boundary.start_page]['text_boundary'] = boundary
        
        # Add visual candidates
        for candidate in visual_candidates:
            if candidate.page_number not in boundary_map:
                boundary_map[candidate.page_number] = {
                    'text_boundary': None,
                    'visual_candidate': candidate,
                    'page': pages[candidate.page_number - 1] if candidate.page_number <= len(pages) else None
                }
            else:
                boundary_map[candidate.page_number]['visual_candidate'] = candidate
        
        # Add significant layout changes from visual summary
        for change in visual_summary.get('significant_layout_changes', []):
            page_num = change['page']
            if page_num not in boundary_map and change['type'] in ['orientation_change', 'column_change']:
                # Create a visual signal for significant layout change
                vsignal = VisualSignal(
                    type=VisualSignalType.LAYOUT_STRUCTURE_CHANGE,
                    confidence=0.7,
                    page_number=page_num,
                    description=f"{change['type']}: {change['from']} to {change['to']}"
                )
                
                candidate = VisualBoundaryCandidate(
                    page_number=page_num,
                    visual_signals=[vsignal],
                    visual_confidence=0.7,
                    layout_change_score=0.8,
                    visual_separator_score=0.0
                )
                
                boundary_map[page_num] = {
                    'text_boundary': None,
                    'visual_candidate': candidate,
                    'page': pages[page_num - 1] if page_num <= len(pages) else None
                }
        
        # Combine and create final boundaries
        combined_boundaries = []
        
        for page_num in sorted(boundary_map.keys()):
            data = boundary_map[page_num]
            text_boundary = data['text_boundary']
            visual_candidate = data['visual_candidate']
            page = data['page']
            
            # Calculate combined confidence
            if text_boundary and visual_candidate:
                # Both detectors agree - boost confidence
                combined_confidence = (
                    self.text_weight * text_boundary.confidence +
                    self.visual_weight * visual_candidate.visual_confidence
                )
                # Boost for agreement
                combined_confidence = min(1.0, combined_confidence * 1.1)
                
                # Combine signals (convert visual signals to standard signals)
                visual_signals_standard = self._convert_visual_signals_to_standard(visual_candidate.visual_signals)
                all_signals = text_boundary.signals + visual_signals_standard
                
                boundary = Boundary(
                    start_page=page_num,
                    end_page=page_num,
                    confidence=combined_confidence,
                    signals=all_signals,
                    document_type=text_boundary.document_type,
                    metadata={
                        'text_confidence': text_boundary.confidence,
                        'visual_confidence': visual_candidate.visual_confidence,
                        'detection_method': 'hybrid',
                        'layout_change_score': visual_candidate.layout_change_score,
                        'visual_separator_score': visual_candidate.visual_separator_score
                    }
                )
            
            elif text_boundary:
                # Only text detection
                boundary = text_boundary
                boundary.metadata['detection_method'] = 'text_only'
            
            elif visual_candidate and visual_candidate.visual_confidence > 0.7:
                # Strong visual signal without text confirmation
                visual_signals_standard = self._convert_visual_signals_to_standard(visual_candidate.visual_signals)
                boundary = Boundary(
                    start_page=page_num,
                    end_page=page_num,
                    confidence=visual_candidate.visual_confidence * 0.8,  # Slight penalty
                    signals=visual_signals_standard,
                    document_type=self._infer_document_type(page) if page else None,
                    metadata={
                        'visual_confidence': visual_candidate.visual_confidence,
                        'detection_method': 'visual_only',
                        'layout_change_score': visual_candidate.layout_change_score,
                        'visual_separator_score': visual_candidate.visual_separator_score
                    }
                )
            else:
                continue
            
            # Skip if below minimum confidence
            if boundary.confidence < self.min_combined_confidence:
                continue
            
            combined_boundaries.append(boundary)
        
        # Fix end pages
        for i in range(len(combined_boundaries) - 1):
            combined_boundaries[i].end_page = combined_boundaries[i + 1].start_page - 1
        
        if combined_boundaries and pages:
            combined_boundaries[-1].end_page = pages[-1].page_number
        
        logger.info(f"Detected {len(combined_boundaries)} boundaries using hybrid approach")
        return combined_boundaries
    
    def _convert_visual_signals_to_standard(self, visual_signals: List[VisualSignal]) -> List[Signal]:
        """Convert visual signals to standard signals for boundary objects."""
        standard_signals = []
        
        for vsignal in visual_signals:
            # Map visual signal types to standard signal types
            if vsignal.type in [
                VisualSignalType.LAYOUT_STRUCTURE_CHANGE,
                VisualSignalType.COLUMN_LAYOUT_CHANGE,
                VisualSignalType.PAGE_ORIENTATION_CHANGE
            ]:
                signal_type = SignalType.LAYOUT_CHANGE
            elif vsignal.type in [
                VisualSignalType.VISUAL_SEPARATOR_LINE,
                VisualSignalType.WHITESPACE_PATTERN
            ]:
                signal_type = SignalType.VISUAL_SEPARATOR
            elif vsignal.type in [
                VisualSignalType.FONT_STYLE_CHANGE,
                VisualSignalType.COLOR_SCHEME_CHANGE,
                VisualSignalType.HEADER_FOOTER_CHANGE
            ]:
                signal_type = SignalType.LAYOUT_CHANGE
            elif vsignal.type in [
                VisualSignalType.LOGO_DETECTION,
                VisualSignalType.SIGNATURE_DETECTION
            ]:
                signal_type = SignalType.DOCUMENT_HEADER
            else:
                signal_type = SignalType.VISUAL_SEPARATOR
            
            standard_signal = Signal(
                type=signal_type,
                confidence=vsignal.confidence,
                page_number=vsignal.page_number,
                description=f"[Visual] {vsignal.description}",
                metadata=vsignal.metadata
            )
            standard_signals.append(standard_signal)
        
        return standard_signals
    
    def _infer_document_type(self, page: PageVisualInfo) -> Optional[DocumentType]:
        """Infer document type from visual features when text detection is not available."""
        if not page or not page.visual_features:
            return DocumentType.OTHER
        
        features = page.visual_features
        
        # Check for form-like characteristics
        if features.num_tables > 2 or (features.has_signature and features.num_tables > 0):
            return DocumentType.FORM
        
        # Check for presentation characteristics
        if features.num_charts > 0 and features.num_images > 2:
            return DocumentType.PRESENTATION
        
        # Check for spreadsheet characteristics
        if features.num_tables > 3 and features.num_images == 0:
            return DocumentType.SPREADSHEET
        
        # Check for letter characteristics
        if features.has_signature and features.num_columns == 1:
            return DocumentType.LETTER
        
        # Check for report characteristics
        if features.num_columns > 1 or (features.has_header and features.has_footer):
            return DocumentType.REPORT
        
        return DocumentType.OTHER
    
    def get_detection_summary(
        self,
        boundaries: List[Boundary]
    ) -> Dict[str, Any]:
        """
        Get a summary of the detection results.
        
        Args:
            boundaries: List of detected boundaries
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_documents': len(boundaries),
            'detection_methods': {
                'hybrid': 0,
                'text_only': 0,
                'visual_only': 0
            },
            'document_types': {},
            'average_confidence': 0.0,
            'high_confidence_boundaries': 0,
            'low_confidence_boundaries': 0,
            'visual_signals_used': {},
            'text_signals_used': {}
        }
        
        if not boundaries:
            return summary
        
        total_confidence = 0.0
        
        for boundary in boundaries:
            # Count detection methods
            method = boundary.metadata.get('detection_method', 'unknown')
            if method in summary['detection_methods']:
                summary['detection_methods'][method] += 1
            
            # Count document types
            doc_type = boundary.document_type.value if boundary.document_type else 'unknown'
            summary['document_types'][doc_type] = summary['document_types'].get(doc_type, 0) + 1
            
            # Track confidence
            total_confidence += boundary.confidence
            if boundary.confidence >= 0.8:
                summary['high_confidence_boundaries'] += 1
            elif boundary.confidence < 0.6:
                summary['low_confidence_boundaries'] += 1
            
            # Count signal types
            for signal in boundary.signals:
                if isinstance(signal.type, VisualSignalType):
                    signal_name = signal.type.value
                    summary['visual_signals_used'][signal_name] = \
                        summary['visual_signals_used'].get(signal_name, 0) + 1
                elif isinstance(signal.type, SignalType):
                    signal_name = signal.type.value
                    summary['text_signals_used'][signal_name] = \
                        summary['text_signals_used'].get(signal_name, 0) + 1
        
        summary['average_confidence'] = total_confidence / len(boundaries)
        
        return summary
    
    def _detect_with_intelligent_ocr(
        self,
        file_path: Path,
        use_visual: bool,
        page_range: Optional[Tuple[int, int]] = None
    ) -> List[Boundary]:
        """
        Detect boundaries using intelligent OCR strategy.
        
        This method:
        1. Analyzes the document to plan OCR strategy
        2. Processes pages with appropriate quality based on their importance
        3. Runs boundary detection with optimized text extraction
        """
        logger.info("Using intelligent OCR strategy for boundary detection")
        
        # Step 1: Plan OCR strategy
        ocr_plan = self.ocr_strategy.plan_ocr_strategy(file_path)
        logger.info(
            f"OCR Plan: {ocr_plan['quality_summary']['high_quality_pages']} high quality, "
            f"{ocr_plan['quality_summary']['medium_quality_pages']} medium, "
            f"{ocr_plan['quality_summary']['fast_pages']} fast, "
            f"{ocr_plan['quality_summary']['skipped_pages']} skipped"
        )
        
        # Step 2: Process pages according to strategy
        all_pages = []
        processed_pages = {}
        
        # Process in optimal order
        for page_idx, page_num in enumerate(ocr_plan['processing_order']):
            if page_range and (page_num < page_range[0] - 1 or page_num > page_range[1] - 1):
                continue
                
            strategy_info = ocr_plan['page_strategies'][page_num]
            
            if strategy_info['strategy'] == 'skip':
                # Use existing text if available
                logger.debug(f"Skipping OCR for page {page_num + 1}: {strategy_info['reason']}")
                continue
            
            # Create processor with page-specific config
            page_config = strategy_info['config']
            if page_config:
                # Process single page with specific config
                page_result = self._process_single_page(
                    file_path, 
                    page_num, 
                    page_config,
                    use_visual
                )
                processed_pages[page_num] = page_result
                
                # Early boundary detection on high-quality pages
                if strategy_info['strategy'] == 'high_quality' and len(processed_pages) > 3:
                    # Check if we've found enough boundaries to optimize remaining pages
                    temp_pages = [processed_pages[i] for i in sorted(processed_pages.keys())]
                    early_boundaries = self._quick_boundary_check(temp_pages)
                    
                    if self._can_optimize_remaining(early_boundaries, page_idx, len(ocr_plan['processing_order'])):
                        logger.info(f"Found sufficient boundaries, optimizing remaining {len(ocr_plan['processing_order']) - page_idx - 1} pages")
                        # Switch remaining pages to fast mode
                        for remaining_page in ocr_plan['processing_order'][page_idx + 1:]:
                            if ocr_plan['page_strategies'][remaining_page]['strategy'] == 'high_quality':
                                ocr_plan['page_strategies'][remaining_page]['strategy'] = 'fast'
                                ocr_plan['page_strategies'][remaining_page]['config'] = PipelineProfiles.get_splitter_detection_config()
        
        # Step 3: Fill in any gaps with fast processing
        import fitz
        doc = fitz.open(str(file_path))
        total_pages = len(doc)
        doc.close()
        
        for page_num in range(total_pages):
            if page_num not in processed_pages:
                # Process with fast config
                fast_config = PipelineProfiles.get_splitter_detection_config()
                page_result = self._process_single_page(
                    file_path,
                    page_num,
                    fast_config,
                    use_visual
                )
                processed_pages[page_num] = page_result
        
        # Step 4: Assemble pages in order
        all_pages = [processed_pages[i] for i in sorted(processed_pages.keys())]
        
        # Step 5: Run boundary detection
        if use_visual and self.visual_detector:
            boundaries = self._detect_hybrid_boundaries(all_pages)
        else:
            page_infos = [PageInfo(**p.dict()) if hasattr(p, 'dict') else p for p in all_pages]
            boundaries = self.text_detector.detect_boundaries(page_infos)
        
        # Step 6: If LLM is enabled and we have low-confidence boundaries, enhance with LLM
        if self.config.enable_llm:
            boundaries = self._enhance_with_llm(boundaries, all_pages, file_path)
        
        return boundaries
    
    def _process_single_page(
        self,
        file_path: Path,
        page_num: int,
        config: OCRConfig,
        use_visual: bool
    ) -> Union[PageInfo, PageVisualInfo]:
        """Process a single page with specific OCR configuration."""
        # Create temporary processor with specific config
        temp_processor = UnifiedDocumentProcessor(
            mode=ProcessingMode.SMART,
            ocr_config=config,
            enable_adaptive=False  # Use our specific config
        )
        
        # Process just this page
        import fitz
        doc = fitz.open(str(file_path))
        
        # Create single-page PDF
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            single_page_doc = fitz.open()
            single_page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            single_page_doc.save(tmp_file.name)
            single_page_doc.close()
            
            # Process the single page
            if use_visual:
                pages = list(temp_processor.process_document_with_visual(Path(tmp_file.name)))
                result = pages[0] if pages else None
            else:
                processed_doc = temp_processor.process_document(Path(tmp_file.name))
                result = processed_doc.page_info[0] if processed_doc.page_info else None
            
            # Clean up
            Path(tmp_file.name).unlink()
        
        doc.close()
        
        # Adjust page number
        if result:
            result.page_number = page_num + 1
        
        return result
    
    def _quick_boundary_check(self, pages: List[PageInfo]) -> List[Boundary]:
        """Quick boundary check on processed pages."""
        try:
            return self.text_detector.detect_boundaries(pages)
        except:
            return []
    
    def _can_optimize_remaining(
        self,
        early_boundaries: List[Boundary],
        current_idx: int,
        total_pages: int
    ) -> bool:
        """
        Determine if we can optimize remaining pages based on early results.
        
        Returns True if:
        - We've found a reasonable number of boundaries
        - The boundaries have high confidence
        - We've processed enough pages to establish a pattern
        """
        if not early_boundaries:
            return False
        
        # Need at least 30% of expected boundaries
        pages_processed_ratio = current_idx / total_pages
        if pages_processed_ratio < 0.3:
            return False
        
        # Check average confidence
        avg_confidence = sum(b.confidence for b in early_boundaries) / len(early_boundaries)
        if avg_confidence < 0.75:
            return False
        
        # Check if we have a good distribution of boundaries
        boundaries_per_page = len(early_boundaries) / (current_idx + 1)
        expected_total_boundaries = boundaries_per_page * total_pages
        
        # If we expect a reasonable number of documents (2-20), we can optimize
        return 2 <= expected_total_boundaries <= 20
    
    def _enhance_with_llm(
        self,
        boundaries: List[Boundary],
        pages: List[PageInfo],
        file_path: Path
    ) -> List[Boundary]:
        """
        Enhance boundaries with LLM analysis for low-confidence cases.
        
        This is a placeholder for LLM integration.
        """
        # Find low-confidence boundaries
        low_confidence = [b for b in boundaries if b.confidence < self.config.llm_confidence_threshold]
        
        if low_confidence:
            logger.info(f"Found {len(low_confidence)} low-confidence boundaries for LLM analysis")
            # TODO: Integrate with Phi-4 Mini or other LLM
            # For now, just log
            for boundary in low_confidence:
                logger.debug(f"Would analyze boundary at page {boundary.start_page} with LLM (confidence: {boundary.confidence:.2f})")
        
        return boundaries