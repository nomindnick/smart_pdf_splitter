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
from .enhanced_document_processor import EnhancedDocumentProcessor
from .models import (
    PageInfo, 
    PageVisualInfo, 
    Boundary, 
    Signal, 
    SignalType,
    VisualSignalType,
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
        visual_memory_limit_mb: int = 2048,
        max_image_dimension: int = 1024,
        page_image_resolution: int = 150,
        visual_batch_size: int = 2,
        visual_confidence_threshold: float = 0.5,
        vlm_model: str = "docling-vlm-v1",
        vlm_prompt_template: str = "Analyze if this page transition represents a document boundary."
    ):
        self.enable_visual_features = enable_visual_features
        self.enable_picture_classification = enable_picture_classification
        self.enable_vlm = enable_vlm
        self.visual_memory_limit_mb = visual_memory_limit_mb
        self.max_image_dimension = max_image_dimension
        self.page_image_resolution = page_image_resolution
        self.visual_batch_size = visual_batch_size
        self.visual_confidence_threshold = visual_confidence_threshold
        self.vlm_model = vlm_model
        self.vlm_prompt_template = vlm_prompt_template


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
        
        # Initialize processor
        self.processor = EnhancedDocumentProcessor(
            enable_visual_features=self.config.enable_visual_features,
            enable_vlm=self.config.enable_vlm,
            visual_memory_limit_mb=self.config.visual_memory_limit_mb,
            page_batch_size=self.config.visual_batch_size
        )
    
    def detect_boundaries(
        self,
        file_path: Path,
        use_visual: Optional[bool] = None,
        page_range: Optional[Tuple[int, int]] = None
    ) -> List[Boundary]:
        """
        Detect boundaries using hybrid approach.
        
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
                signal = Signal(
                    type=VisualSignalType.LAYOUT_STRUCTURE_CHANGE,
                    confidence=0.7,
                    page_number=page_num,
                    description=f"{change['type']}: {change['from']} to {change['to']}"
                )
                
                candidate = VisualBoundaryCandidate(
                    page_number=page_num,
                    visual_signals=[signal],
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
                
                # Combine signals
                all_signals = text_boundary.signals + visual_candidate.visual_signals
                
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
                boundary = Boundary(
                    start_page=page_num,
                    end_page=page_num,
                    confidence=visual_candidate.visual_confidence * 0.8,  # Slight penalty
                    signals=visual_candidate.visual_signals,
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