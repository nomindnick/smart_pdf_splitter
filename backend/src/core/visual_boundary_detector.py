"""
Visual boundary detection module for identifying document boundaries using visual features.

This module uses layout analysis, visual separators, and structural changes to detect
where documents begin and end in multi-document PDFs.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass

from .models import (
    PageVisualInfo, 
    Boundary, 
    Signal, 
    SignalType,
    VisualSignalType,
    VisualFeatures
)
from .visual_processor import VisualFeatureProcessor

logger = logging.getLogger(__name__)


@dataclass
class VisualBoundaryCandidate:
    """Candidate boundary based on visual analysis."""
    page_number: int
    visual_signals: List[Signal]
    visual_confidence: float
    layout_change_score: float
    visual_separator_score: float
    

class VisualBoundaryDetector:
    """
    Detects document boundaries using visual features and layout analysis.
    """
    
    # Thresholds for visual changes
    LAYOUT_CHANGE_THRESHOLD = 0.3
    FONT_CHANGE_THRESHOLD = 2.0  # Font size difference in points
    MARGIN_CHANGE_THRESHOLD = 0.1  # 10% of page dimension
    COLOR_CHANGE_THRESHOLD = 0.2
    
    # Visual signal weights
    VISUAL_SIGNAL_WEIGHTS = {
        VisualSignalType.LAYOUT_STRUCTURE_CHANGE: 0.8,
        VisualSignalType.FONT_STYLE_CHANGE: 0.6,
        VisualSignalType.COLOR_SCHEME_CHANGE: 0.5,
        VisualSignalType.VISUAL_SEPARATOR_LINE: 0.9,
        VisualSignalType.HEADER_FOOTER_CHANGE: 0.7,
        VisualSignalType.LOGO_DETECTION: 0.8,
        VisualSignalType.SIGNATURE_DETECTION: 0.6,
        VisualSignalType.PAGE_ORIENTATION_CHANGE: 0.9,
        VisualSignalType.COLUMN_LAYOUT_CHANGE: 0.7,
        VisualSignalType.WHITESPACE_PATTERN: 0.5
    }
    
    def __init__(
        self,
        min_visual_confidence: float = 0.5,
        enable_vlm_analysis: bool = True,
        visual_processor: Optional[VisualFeatureProcessor] = None
    ):
        """
        Initialize visual boundary detector.
        
        Args:
            min_visual_confidence: Minimum confidence for visual boundaries
            enable_vlm_analysis: Whether to use VLM for analysis
            visual_processor: Optional visual feature processor instance
        """
        self.min_visual_confidence = min_visual_confidence
        self.enable_vlm_analysis = enable_vlm_analysis
        self.visual_processor = visual_processor or VisualFeatureProcessor()
    
    def detect_visual_boundaries(
        self,
        pages: List[PageVisualInfo],
        context_window: int = 2
    ) -> List[VisualBoundaryCandidate]:
        """
        Detect boundaries based on visual features.
        
        Args:
            pages: List of pages with visual information
            context_window: Number of pages to consider for context
            
        Returns:
            List of visual boundary candidates
        """
        if not pages or len(pages) < 2:
            return []
        
        candidates = []
        
        # Analyze each page transition
        for i in range(1, len(pages)):
            prev_page = pages[i - 1]
            curr_page = pages[i]
            
            visual_signals = []
            
            # Check layout structure changes
            layout_signal = self._detect_layout_change(prev_page, curr_page)
            if layout_signal:
                visual_signals.append(layout_signal)
            
            # Check font style changes
            font_signal = self._detect_font_change(prev_page, curr_page)
            if font_signal:
                visual_signals.append(font_signal)
            
            # Check color scheme changes
            color_signal = self._detect_color_change(prev_page, curr_page)
            if color_signal:
                visual_signals.append(color_signal)
            
            # Check for visual separators
            separator_signal = self._detect_visual_separator(curr_page)
            if separator_signal:
                visual_signals.append(separator_signal)
            
            # Check header/footer changes
            header_footer_signal = self._detect_header_footer_change(prev_page, curr_page)
            if header_footer_signal:
                visual_signals.append(header_footer_signal)
            
            # Check for logos (often indicate new document)
            logo_signal = self._detect_logo_change(prev_page, curr_page)
            if logo_signal:
                visual_signals.append(logo_signal)
            
            # Check page orientation
            orientation_signal = self._detect_orientation_change(prev_page, curr_page)
            if orientation_signal:
                visual_signals.append(orientation_signal)
            
            # Check whitespace patterns
            whitespace_signal = self._detect_whitespace_pattern(prev_page, curr_page)
            if whitespace_signal:
                visual_signals.append(whitespace_signal)
            
            # Analyze with VLM if enabled
            if self.enable_vlm_analysis and visual_signals:
                vlm_signal = self._analyze_with_vlm(prev_page, curr_page)
                if vlm_signal:
                    visual_signals.append(vlm_signal)
            
            # Create candidate if significant visual changes detected
            if visual_signals:
                confidence = self._calculate_visual_confidence(visual_signals)
                
                if confidence >= self.min_visual_confidence:
                    candidate = VisualBoundaryCandidate(
                        page_number=curr_page.page_number,
                        visual_signals=visual_signals,
                        visual_confidence=confidence,
                        layout_change_score=self._calculate_layout_change_score(
                            prev_page, curr_page
                        ),
                        visual_separator_score=self._calculate_separator_score(curr_page)
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def _detect_layout_change(
        self,
        prev_page: PageVisualInfo,
        curr_page: PageVisualInfo
    ) -> Optional[Signal]:
        """Detect significant layout structure changes."""
        if not (prev_page.visual_features and curr_page.visual_features):
            return None
        
        prev_features = prev_page.visual_features
        curr_features = curr_page.visual_features
        
        # Check column layout change
        if prev_features.num_columns != curr_features.num_columns:
            return Signal(
                type=VisualSignalType.COLUMN_LAYOUT_CHANGE,
                confidence=0.8,
                page_number=curr_page.page_number,
                description=f"Column layout changed from {prev_features.num_columns} to {curr_features.num_columns}"
            )
        
        # Check alignment change
        if prev_features.text_alignment != curr_features.text_alignment:
            return Signal(
                type=VisualSignalType.LAYOUT_STRUCTURE_CHANGE,
                confidence=0.6,
                page_number=curr_page.page_number,
                description=f"Text alignment changed from {prev_features.text_alignment} to {curr_features.text_alignment}"
            )
        
        # Check margin changes
        margin_changes = self._calculate_margin_changes(prev_features, curr_features)
        if margin_changes > self.MARGIN_CHANGE_THRESHOLD:
            return Signal(
                type=VisualSignalType.LAYOUT_STRUCTURE_CHANGE,
                confidence=0.7,
                page_number=curr_page.page_number,
                description=f"Significant margin changes detected ({margin_changes:.0%})"
            )
        
        return None
    
    def _detect_font_change(
        self,
        prev_page: PageVisualInfo,
        curr_page: PageVisualInfo
    ) -> Optional[Signal]:
        """Detect significant font changes."""
        if not (prev_page.visual_features and curr_page.visual_features):
            return None
        
        prev_features = prev_page.visual_features
        curr_features = curr_page.visual_features
        
        # Check font size change
        if (prev_features.primary_font_size and curr_features.primary_font_size):
            size_diff = abs(prev_features.primary_font_size - curr_features.primary_font_size)
            if size_diff > self.FONT_CHANGE_THRESHOLD:
                return Signal(
                    type=VisualSignalType.FONT_STYLE_CHANGE,
                    confidence=0.7,
                    page_number=curr_page.page_number,
                    description=f"Font size changed by {size_diff:.1f} points"
                )
        
        # Check font family change
        if (prev_features.primary_font_family and curr_features.primary_font_family and
            prev_features.primary_font_family != curr_features.primary_font_family):
            return Signal(
                type=VisualSignalType.FONT_STYLE_CHANGE,
                confidence=0.8,
                page_number=curr_page.page_number,
                description=f"Font family changed from {prev_features.primary_font_family} to {curr_features.primary_font_family}"
            )
        
        return None
    
    def _detect_color_change(
        self,
        prev_page: PageVisualInfo,
        curr_page: PageVisualInfo
    ) -> Optional[Signal]:
        """Detect significant color scheme changes."""
        if not (prev_page.visual_features and curr_page.visual_features):
            return None
        
        prev_features = prev_page.visual_features
        curr_features = curr_page.visual_features
        
        # Check background color change
        if (prev_features.background_color and curr_features.background_color and
            prev_features.background_color != curr_features.background_color):
            return Signal(
                type=VisualSignalType.COLOR_SCHEME_CHANGE,
                confidence=0.6,
                page_number=curr_page.page_number,
                description="Background color changed"
            )
        
        # Check text color change
        if (prev_features.primary_text_color and curr_features.primary_text_color and
            prev_features.primary_text_color != curr_features.primary_text_color):
            return Signal(
                type=VisualSignalType.COLOR_SCHEME_CHANGE,
                confidence=0.5,
                page_number=curr_page.page_number,
                description="Primary text color changed"
            )
        
        return None
    
    def _detect_visual_separator(self, page: PageVisualInfo) -> Optional[Signal]:
        """Detect visual separators like lines or boxes."""
        if not page.visual_features:
            return None
        
        # Check for separator patterns in layout elements
        for element in page.layout_elements:
            element_type = element.get('type', '').lower()
            
            # Look for horizontal lines that might be separators
            if element_type in ['line', 'separator', 'rule']:
                bbox = element.get('bbox')
                if bbox and hasattr(bbox, 'width') and hasattr(bbox, 'height'):
                    # Horizontal line (width much greater than height)
                    if bbox.width > bbox.height * 10:
                        return Signal(
                            type=VisualSignalType.VISUAL_SEPARATOR_LINE,
                            confidence=0.9,
                            page_number=page.page_number,
                            description="Horizontal separator line detected"
                        )
            
            # Look for separator boxes
            elif element_type == 'box' and element.get('is_separator'):
                return Signal(
                    type=VisualSignalType.VISUAL_SEPARATOR_LINE,
                    confidence=0.8,
                    page_number=page.page_number,
                    description="Visual separator box detected"
                )
        
        return None
    
    def _detect_header_footer_change(
        self,
        prev_page: PageVisualInfo,
        curr_page: PageVisualInfo
    ) -> Optional[Signal]:
        """Detect changes in headers or footers."""
        if not (prev_page.visual_features and curr_page.visual_features):
            return None
        
        prev_features = prev_page.visual_features
        curr_features = curr_page.visual_features
        
        # Check header change
        if (prev_features.has_header and curr_features.has_header and
            prev_features.header_text != curr_features.header_text):
            return Signal(
                type=VisualSignalType.HEADER_FOOTER_CHANGE,
                confidence=0.8,
                page_number=curr_page.page_number,
                description="Header content changed"
            )
        
        # Check footer change
        if (prev_features.has_footer and curr_features.has_footer and
            prev_features.footer_text != curr_features.footer_text):
            return Signal(
                type=VisualSignalType.HEADER_FOOTER_CHANGE,
                confidence=0.7,
                page_number=curr_page.page_number,
                description="Footer content changed"
            )
        
        # Check header/footer appearance or disappearance
        if prev_features.has_header != curr_features.has_header:
            return Signal(
                type=VisualSignalType.HEADER_FOOTER_CHANGE,
                confidence=0.6,
                page_number=curr_page.page_number,
                description=f"Header {'appeared' if curr_features.has_header else 'disappeared'}"
            )
        
        return None
    
    def _detect_logo_change(
        self,
        prev_page: PageVisualInfo,
        curr_page: PageVisualInfo
    ) -> Optional[Signal]:
        """Detect appearance of logos."""
        if not (prev_page.visual_features and curr_page.visual_features):
            return None
        
        # Logo on current page but not previous
        if (curr_page.visual_features.has_logo and 
            not prev_page.visual_features.has_logo):
            return Signal(
                type=VisualSignalType.LOGO_DETECTION,
                confidence=0.8,
                page_number=curr_page.page_number,
                description="New logo detected"
            )
        
        # Check picture classifications for logo
        if curr_page.picture_classifications:
            logo_confidence = curr_page.picture_classifications.get('logo', 0)
            if logo_confidence > 0.7:
                return Signal(
                    type=VisualSignalType.LOGO_DETECTION,
                    confidence=logo_confidence,
                    page_number=curr_page.page_number,
                    description=f"Logo detected with {logo_confidence:.0%} confidence"
                )
        
        return None
    
    def _detect_orientation_change(
        self,
        prev_page: PageVisualInfo,
        curr_page: PageVisualInfo
    ) -> Optional[Signal]:
        """Detect page orientation changes."""
        if not (prev_page.visual_features and curr_page.visual_features):
            return None
        
        if (prev_page.visual_features.orientation != 
            curr_page.visual_features.orientation):
            return Signal(
                type=VisualSignalType.PAGE_ORIENTATION_CHANGE,
                confidence=0.9,
                page_number=curr_page.page_number,
                description=f"Page orientation changed from {prev_page.visual_features.orientation} to {curr_page.visual_features.orientation}"
            )
        
        return None
    
    def _detect_whitespace_pattern(
        self,
        prev_page: PageVisualInfo,
        curr_page: PageVisualInfo
    ) -> Optional[Signal]:
        """Detect significant whitespace patterns indicating boundaries."""
        if not curr_page.visual_features:
            return None
        
        # Check if current page has significantly more whitespace
        curr_features = curr_page.visual_features
        
        # Calculate whitespace score based on margins and content density
        whitespace_score = 0.0
        
        # Large top margin might indicate new document
        if curr_features.margin_top and curr_features.margin_top > 0.2:
            whitespace_score += 0.3
        
        # Check overall content density
        if curr_page.word_count < 100 and curr_page.visual_features.num_images == 0:
            whitespace_score += 0.4
        
        # Compare with previous page
        if prev_page.visual_features:
            prev_density = prev_page.word_count / (prev_page.width * prev_page.height)
            curr_density = curr_page.word_count / (curr_page.width * curr_page.height)
            
            if curr_density < prev_density * 0.3:  # 70% less dense
                whitespace_score += 0.3
        
        if whitespace_score > 0.5:
            return Signal(
                type=VisualSignalType.WHITESPACE_PATTERN,
                confidence=min(whitespace_score, 0.8),
                page_number=curr_page.page_number,
                description="Significant whitespace pattern detected"
            )
        
        return None
    
    def _analyze_with_vlm(
        self,
        prev_page: PageVisualInfo,
        curr_page: PageVisualInfo
    ) -> Optional[Signal]:
        """Use VLM to analyze visual boundary."""
        if not self.enable_vlm_analysis:
            return None
        
        try:
            # This would integrate with Docling's VLM capabilities
            # For demonstration, we'll create a placeholder
            
            # Check if VLM analysis is available in page metadata
            if curr_page.vlm_analysis:
                boundary_score = curr_page.vlm_analysis.get('boundary_score', 0)
                if boundary_score > 0.7:
                    return Signal(
                        type=VisualSignalType.VISUAL_SEPARATOR_LINE,
                        confidence=boundary_score,
                        page_number=curr_page.page_number,
                        description="VLM detected visual boundary",
                        metadata={'vlm_analysis': curr_page.vlm_analysis}
                    )
        except Exception as e:
            logger.error(f"Error in VLM analysis: {e}")
        
        return None
    
    def _calculate_margin_changes(
        self,
        prev_features: VisualFeatures,
        curr_features: VisualFeatures
    ) -> float:
        """Calculate the magnitude of margin changes."""
        changes = []
        
        if prev_features.margin_top and curr_features.margin_top:
            changes.append(abs(prev_features.margin_top - curr_features.margin_top))
        if prev_features.margin_bottom and curr_features.margin_bottom:
            changes.append(abs(prev_features.margin_bottom - curr_features.margin_bottom))
        if prev_features.margin_left and curr_features.margin_left:
            changes.append(abs(prev_features.margin_left - curr_features.margin_left))
        if prev_features.margin_right and curr_features.margin_right:
            changes.append(abs(prev_features.margin_right - curr_features.margin_right))
        
        return max(changes) if changes else 0.0
    
    def _calculate_visual_confidence(self, signals: List[Signal]) -> float:
        """Calculate overall confidence from visual signals."""
        if not signals:
            return 0.0
        
        # Weighted average of signal confidences
        total_weight = 0.0
        weighted_sum = 0.0
        
        for signal in signals:
            weight = self.VISUAL_SIGNAL_WEIGHTS.get(signal.type, 0.5)
            weighted_sum += signal.confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        base_confidence = weighted_sum / total_weight
        
        # Boost for multiple strong signals
        strong_signals = sum(1 for s in signals if s.confidence >= 0.7)
        if strong_signals >= 2:
            base_confidence = min(1.0, base_confidence * 1.1)
        
        # Boost for very strong individual signals
        max_signal_confidence = max(s.confidence for s in signals)
        if max_signal_confidence >= 0.9:
            base_confidence = max(base_confidence, max_signal_confidence * 0.95)
        
        return base_confidence
    
    def _calculate_layout_change_score(
        self,
        prev_page: PageVisualInfo,
        curr_page: PageVisualInfo
    ) -> float:
        """Calculate a comprehensive layout change score."""
        if not (prev_page.visual_features and curr_page.visual_features):
            return 0.0
        
        scores = []
        
        # Column change
        if prev_page.visual_features.num_columns != curr_page.visual_features.num_columns:
            scores.append(0.8)
        
        # Visual element count changes
        prev_elements = (prev_page.visual_features.num_images + 
                        prev_page.visual_features.num_tables +
                        prev_page.visual_features.num_charts)
        curr_elements = (curr_page.visual_features.num_images + 
                        curr_page.visual_features.num_tables +
                        curr_page.visual_features.num_charts)
        
        if abs(prev_elements - curr_elements) > 2:
            scores.append(0.6)
        
        # Margin changes
        margin_change = self._calculate_margin_changes(
            prev_page.visual_features,
            curr_page.visual_features
        )
        if margin_change > 0.1:
            scores.append(margin_change)
        
        # Orientation change
        if prev_page.visual_features.orientation != curr_page.visual_features.orientation:
            scores.append(0.9)
        
        return max(scores) if scores else 0.0
    
    def _calculate_separator_score(self, page: PageVisualInfo) -> float:
        """Calculate score for visual separators on the page."""
        score = 0.0
        
        # Check for separator elements
        for element in page.layout_elements:
            element_type = element.get('type', '').lower()
            if element_type in ['line', 'separator', 'rule']:
                score = max(score, 0.8)
            elif element_type == 'box' and element.get('is_separator'):
                score = max(score, 0.7)
        
        # Check for significant whitespace patterns
        if page.visual_features:
            # Large margins might indicate separator page
            if (page.visual_features.margin_top and page.visual_features.margin_top > 0.3):
                score = max(score, 0.5)
            
            # Very low word count might indicate separator page
            if page.word_count < 50:
                score = max(score, 0.4)
        
        return score