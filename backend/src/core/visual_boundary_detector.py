"""
Visual boundary detection module for analyzing visual features to detect document boundaries.

This module implements visual signal detection methods to identify document boundaries
using layout changes, font variations, color schemes, margins, and other visual cues.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from .models import (
    PageVisualInfo,
    VisualFeatures,
    VisualSignal,
    VisualSignalType,
    DocumentType
)

logger = logging.getLogger(__name__)


@dataclass
class VisualBoundaryCandidate:
    """Represents a potential document boundary based on visual signals."""
    page_number: int
    visual_signals: List[VisualSignal]
    confidence: float
    layout_similarity_score: float = 0.0
    visual_continuity_score: float = 0.0
    
    @property
    def total_visual_confidence(self) -> float:
        """Calculate total confidence from visual signals."""
        if not self.visual_signals:
            return 0.0
        return sum(s.confidence for s in self.visual_signals) / len(self.visual_signals)


class VisualBoundaryDetector:
    """
    Detects document boundaries using visual features and layout analysis.
    
    Analyzes visual characteristics like layout structure, fonts, colors,
    margins, and visual elements to identify document transitions.
    """
    
    # Visual signal weights for confidence calculation
    VISUAL_SIGNAL_WEIGHTS = {
        VisualSignalType.LAYOUT_STRUCTURE_CHANGE: 0.8,
        VisualSignalType.FONT_STYLE_CHANGE: 0.7,
        VisualSignalType.COLOR_SCHEME_CHANGE: 0.6,
        VisualSignalType.VISUAL_SEPARATOR_LINE: 0.9,
        VisualSignalType.HEADER_FOOTER_CHANGE: 0.75,
        VisualSignalType.LOGO_DETECTION: 0.85,
        VisualSignalType.SIGNATURE_DETECTION: 0.8,
        VisualSignalType.PAGE_ORIENTATION_CHANGE: 0.95,
        VisualSignalType.COLUMN_LAYOUT_CHANGE: 0.7,
        VisualSignalType.WHITESPACE_PATTERN: 0.5,
    }
    
    # Thresholds for detection
    LAYOUT_CHANGE_THRESHOLD = 0.3  # 30% difference in layout metrics
    FONT_CHANGE_THRESHOLD = 2.0    # 2pt font size difference
    COLOR_CHANGE_THRESHOLD = 0.2   # 20% color difference
    MARGIN_CHANGE_THRESHOLD = 20   # 20px margin difference
    WHITESPACE_THRESHOLD = 0.7     # 70% of page is whitespace
    
    def __init__(
        self,
        min_visual_confidence: float = 0.5,
        min_visual_signals: int = 1,
        enable_vlm_analysis: bool = False
    ):
        """
        Initialize visual boundary detector.
        
        Args:
            min_visual_confidence: Minimum confidence threshold for visual boundaries
            min_visual_signals: Minimum number of visual signals required
            enable_vlm_analysis: Whether to use vision-language model analysis
        """
        self.min_visual_confidence = min_visual_confidence
        self.min_visual_signals = min_visual_signals
        self.enable_vlm_analysis = enable_vlm_analysis
    
    def detect_visual_boundaries(
        self,
        pages: List[PageVisualInfo],
        context_window: int = 2
    ) -> List[VisualBoundaryCandidate]:
        """
        Detect document boundaries based on visual features.
        
        Args:
            pages: List of pages with visual features
            context_window: Number of pages to look ahead/behind for context
            
        Returns:
            List of visual boundary candidates
        """
        if not pages:
            return []
        
        candidates = []
        
        # Analyze each page transition for visual changes
        for i in range(1, len(pages)):
            prev_page = pages[i-1]
            curr_page = pages[i]
            
            if not (prev_page.visual_features and curr_page.visual_features):
                continue
            
            visual_signals = []
            
            # Check for layout structure changes
            layout_signal = self._detect_layout_change(
                prev_page.visual_features,
                curr_page.visual_features,
                curr_page.page_number
            )
            if layout_signal:
                visual_signals.append(layout_signal)
            
            # Check for font style changes
            font_signal = self._detect_font_change(
                prev_page.visual_features,
                curr_page.visual_features,
                curr_page.page_number
            )
            if font_signal:
                visual_signals.append(font_signal)
            
            # Check for color scheme changes
            color_signal = self._detect_color_change(
                prev_page.visual_features,
                curr_page.visual_features,
                curr_page.page_number
            )
            if color_signal:
                visual_signals.append(color_signal)
            
            # Check for margin changes
            margin_signal = self._detect_margin_change(
                prev_page.visual_features,
                curr_page.visual_features,
                curr_page.page_number
            )
            if margin_signal:
                visual_signals.append(margin_signal)
            
            # Check for header/footer changes
            header_footer_signal = self._detect_header_footer_change(
                prev_page.visual_features,
                curr_page.visual_features,
                curr_page.page_number
            )
            if header_footer_signal:
                visual_signals.append(header_footer_signal)
            
            # Check for visual separator lines
            separator_signal = self._detect_visual_separator(curr_page)
            if separator_signal:
                visual_signals.append(separator_signal)
            
            # Check for logo presence (often indicates new document)
            logo_signal = self._detect_logo_presence(curr_page.visual_features, curr_page.page_number)
            if logo_signal:
                visual_signals.append(logo_signal)
            
            # Check for signature (often indicates document end)
            if i > 0:
                signature_signal = self._detect_signature_presence(
                    prev_page.visual_features,
                    prev_page.page_number
                )
                if signature_signal:
                    visual_signals.append(signature_signal)
            
            # Check for page orientation changes
            orientation_signal = self._detect_orientation_change(
                prev_page.visual_features,
                curr_page.visual_features,
                curr_page.page_number
            )
            if orientation_signal:
                visual_signals.append(orientation_signal)
            
            # Check for column layout changes
            column_signal = self._detect_column_change(
                prev_page.visual_features,
                curr_page.visual_features,
                curr_page.page_number
            )
            if column_signal:
                visual_signals.append(column_signal)
            
            # Check for significant whitespace patterns
            whitespace_signal = self._detect_whitespace_pattern(curr_page)
            if whitespace_signal:
                visual_signals.append(whitespace_signal)
            
            # Use VLM analysis if enabled
            if self.enable_vlm_analysis and curr_page.vlm_analysis:
                vlm_signal = self._analyze_vlm_results(curr_page.vlm_analysis, curr_page.page_number)
                if vlm_signal:
                    visual_signals.append(vlm_signal)
            
            # Create candidate if signals found
            if visual_signals:
                # Calculate layout similarity
                layout_similarity = self._calculate_layout_similarity(
                    prev_page.visual_features,
                    curr_page.visual_features
                )
                
                # Calculate visual continuity
                visual_continuity = self._calculate_visual_continuity(
                    pages,
                    i,
                    context_window
                )
                
                candidate = VisualBoundaryCandidate(
                    page_number=curr_page.page_number,
                    visual_signals=visual_signals,
                    confidence=self._calculate_visual_confidence(visual_signals),
                    layout_similarity_score=layout_similarity,
                    visual_continuity_score=visual_continuity
                )
                
                if (candidate.confidence >= self.min_visual_confidence and
                    len(candidate.visual_signals) >= self.min_visual_signals):
                    candidates.append(candidate)
        
        logger.info(f"Detected {len(candidates)} visual boundary candidates")
        return candidates
    
    def _detect_layout_change(
        self,
        prev_features: VisualFeatures,
        curr_features: VisualFeatures,
        page_number: int
    ) -> Optional[VisualSignal]:
        """Detect significant layout structure changes."""
        changes = []
        
        # Check column count change
        if prev_features.num_columns != curr_features.num_columns:
            changes.append(f"columns: {prev_features.num_columns}→{curr_features.num_columns}")
        
        # Check text alignment change
        if (prev_features.text_alignment and curr_features.text_alignment and
            prev_features.text_alignment != curr_features.text_alignment):
            changes.append(f"alignment: {prev_features.text_alignment}→{curr_features.text_alignment}")
        
        # Check table/chart presence changes
        prev_has_tables = prev_features.num_tables > 0
        curr_has_tables = curr_features.num_tables > 0
        if prev_has_tables != curr_has_tables:
            changes.append(f"tables: {prev_has_tables}→{curr_has_tables}")
        
        if changes:
            confidence = min(0.8, 0.3 * len(changes))
            return VisualSignal(
                type=VisualSignalType.LAYOUT_STRUCTURE_CHANGE,
                confidence=confidence,
                page_number=page_number,
                description=f"Layout changes: {', '.join(changes)}"
            )
        
        return None
    
    def _detect_font_change(
        self,
        prev_features: VisualFeatures,
        curr_features: VisualFeatures,
        page_number: int
    ) -> Optional[VisualSignal]:
        """Detect significant font style changes."""
        if not (prev_features.primary_font_size and curr_features.primary_font_size):
            return None
        
        # Check font size change
        size_diff = abs(prev_features.primary_font_size - curr_features.primary_font_size)
        
        # Check font family change
        family_changed = (prev_features.primary_font_family and 
                         curr_features.primary_font_family and
                         prev_features.primary_font_family != curr_features.primary_font_family)
        
        if size_diff > self.FONT_CHANGE_THRESHOLD or family_changed:
            changes = []
            if size_diff > self.FONT_CHANGE_THRESHOLD:
                changes.append(f"size: {prev_features.primary_font_size:.1f}→{curr_features.primary_font_size:.1f}pt")
            if family_changed:
                changes.append(f"family: {prev_features.primary_font_family}→{curr_features.primary_font_family}")
            
            confidence = 0.7 if family_changed else 0.6
            return VisualSignal(
                type=VisualSignalType.FONT_STYLE_CHANGE,
                confidence=confidence,
                page_number=page_number,
                description=f"Font changes: {', '.join(changes)}"
            )
        
        return None
    
    def _detect_color_change(
        self,
        prev_features: VisualFeatures,
        curr_features: VisualFeatures,
        page_number: int
    ) -> Optional[VisualSignal]:
        """Detect significant color scheme changes."""
        changes = []
        
        # Check background color change
        if (prev_features.background_color and curr_features.background_color and
            prev_features.background_color != curr_features.background_color):
            changes.append("background color")
        
        # Check text color change
        if (prev_features.primary_text_color and curr_features.primary_text_color and
            prev_features.primary_text_color != curr_features.primary_text_color):
            changes.append("text color")
        
        # Check color image presence change
        if prev_features.has_color_images != curr_features.has_color_images:
            changes.append(f"color images: {prev_features.has_color_images}→{curr_features.has_color_images}")
        
        if len(changes) >= 2:  # Need at least 2 color changes to be significant
            return VisualSignal(
                type=VisualSignalType.COLOR_SCHEME_CHANGE,
                confidence=0.6,
                page_number=page_number,
                description=f"Color changes: {', '.join(changes)}"
            )
        
        return None
    
    def _detect_margin_change(
        self,
        prev_features: VisualFeatures,
        curr_features: VisualFeatures,
        page_number: int
    ) -> Optional[VisualSignal]:
        """Detect significant margin changes."""
        if not all([
            prev_features.margin_top, prev_features.margin_bottom,
            prev_features.margin_left, prev_features.margin_right,
            curr_features.margin_top, curr_features.margin_bottom,
            curr_features.margin_left, curr_features.margin_right
        ]):
            return None
        
        # Calculate margin differences
        margin_diffs = {
            'top': abs(prev_features.margin_top - curr_features.margin_top),
            'bottom': abs(prev_features.margin_bottom - curr_features.margin_bottom),
            'left': abs(prev_features.margin_left - curr_features.margin_left),
            'right': abs(prev_features.margin_right - curr_features.margin_right)
        }
        
        # Count significant changes
        significant_changes = [
            margin for margin, diff in margin_diffs.items()
            if diff > self.MARGIN_CHANGE_THRESHOLD
        ]
        
        if len(significant_changes) >= 2:  # At least 2 margins changed significantly
            return VisualSignal(
                type=VisualSignalType.LAYOUT_STRUCTURE_CHANGE,
                confidence=0.65,
                page_number=page_number,
                description=f"Margin changes: {', '.join(significant_changes)}"
            )
        
        return None
    
    def _detect_header_footer_change(
        self,
        prev_features: VisualFeatures,
        curr_features: VisualFeatures,
        page_number: int
    ) -> Optional[VisualSignal]:
        """Detect header/footer changes."""
        changes = []
        
        # Check header presence change
        if prev_features.has_header != curr_features.has_header:
            changes.append(f"header: {prev_features.has_header}→{curr_features.has_header}")
        
        # Check footer presence change
        if prev_features.has_footer != curr_features.has_footer:
            changes.append(f"footer: {prev_features.has_footer}→{curr_features.has_footer}")
        
        # Check header text change (if both have headers)
        if (prev_features.has_header and curr_features.has_header and
            prev_features.header_text and curr_features.header_text and
            prev_features.header_text != curr_features.header_text):
            changes.append("header text changed")
        
        # Check footer text change (if both have footers)
        if (prev_features.has_footer and curr_features.has_footer and
            prev_features.footer_text and curr_features.footer_text and
            prev_features.footer_text != curr_features.footer_text):
            changes.append("footer text changed")
        
        if changes:
            confidence = min(0.75, 0.25 * len(changes))
            return VisualSignal(
                type=VisualSignalType.HEADER_FOOTER_CHANGE,
                confidence=confidence,
                page_number=page_number,
                description=f"Header/footer changes: {', '.join(changes)}"
            )
        
        return None
    
    def _detect_visual_separator(self, page: PageVisualInfo) -> Optional[VisualSignal]:
        """Detect visual separator lines or elements."""
        # Check layout elements for horizontal lines or separators
        separator_count = 0
        
        for element in page.layout_elements:
            if isinstance(element, dict):
                # Look for thin horizontal rectangles (lines)
                if 'bbox' in element:
                    bbox = element['bbox']
                    # Check if it's a horizontal line (very thin height, wide width)
                    if hasattr(bbox, 'height') and hasattr(bbox, 'width'):
                        aspect_ratio = bbox.width / max(bbox.height, 1)
                        if aspect_ratio > 50 and bbox.height < 5:
                            separator_count += 1
                
                # Check for separator-like text elements
                if 'text' in element and isinstance(element['text'], str):
                    text = element['text'].strip()
                    if text in ['---', '___', '***', '• • •', '···']:
                        separator_count += 1
        
        if separator_count > 0:
            return VisualSignal(
                type=VisualSignalType.VISUAL_SEPARATOR_LINE,
                confidence=0.9,
                page_number=page.page_number,
                description=f"Found {separator_count} visual separator(s)"
            )
        
        return None
    
    def _detect_logo_presence(self, features: VisualFeatures, page_number: int) -> Optional[VisualSignal]:
        """Detect logo presence which often indicates document start."""
        if features.has_logo:
            return VisualSignal(
                type=VisualSignalType.LOGO_DETECTION,
                confidence=0.85,
                page_number=page_number,
                description="Logo detected (likely document start)"
            )
        return None
    
    def _detect_signature_presence(self, features: VisualFeatures, page_number: int) -> Optional[VisualSignal]:
        """Detect signature presence which often indicates document end."""
        if features.has_signature:
            return VisualSignal(
                type=VisualSignalType.SIGNATURE_DETECTION,
                confidence=0.8,
                page_number=page_number,
                description="Signature detected (likely document end)"
            )
        return None
    
    def _detect_orientation_change(
        self,
        prev_features: VisualFeatures,
        curr_features: VisualFeatures,
        page_number: int
    ) -> Optional[VisualSignal]:
        """Detect page orientation changes."""
        if prev_features.orientation != curr_features.orientation:
            return VisualSignal(
                type=VisualSignalType.PAGE_ORIENTATION_CHANGE,
                confidence=0.95,
                page_number=page_number,
                description=f"Orientation: {prev_features.orientation}→{curr_features.orientation}"
            )
        return None
    
    def _detect_column_change(
        self,
        prev_features: VisualFeatures,
        curr_features: VisualFeatures,
        page_number: int
    ) -> Optional[VisualSignal]:
        """Detect column layout changes."""
        if prev_features.num_columns != curr_features.num_columns:
            # Significant change if going from/to single column
            is_significant = (
                prev_features.num_columns == 1 or 
                curr_features.num_columns == 1 or
                abs(prev_features.num_columns - curr_features.num_columns) > 1
            )
            
            if is_significant:
                return VisualSignal(
                    type=VisualSignalType.COLUMN_LAYOUT_CHANGE,
                    confidence=0.7,
                    page_number=page_number,
                    description=f"Columns: {prev_features.num_columns}→{curr_features.num_columns}"
                )
        return None
    
    def _detect_whitespace_pattern(self, page: PageVisualInfo) -> Optional[VisualSignal]:
        """Detect significant whitespace patterns."""
        if not page.visual_features:
            return None
        
        # Check if page is mostly empty (based on word count)
        if page.is_mostly_empty:
            # Calculate whitespace based on margins and content density
            features = page.visual_features
            if all([features.margin_top, features.margin_bottom, 
                   features.margin_left, features.margin_right]):
                # Calculate approximate whitespace percentage
                total_margin = (features.margin_top + features.margin_bottom + 
                              features.margin_left + features.margin_right)
                page_area = page.width * page.height if page.width > 0 and page.height > 0 else 1
                margin_area = total_margin * min(page.width, page.height)
                whitespace_ratio = min(1.0, margin_area / page_area)
                
                if whitespace_ratio > self.WHITESPACE_THRESHOLD or page.word_count < 20:
                    return VisualSignal(
                        type=VisualSignalType.WHITESPACE_PATTERN,
                        confidence=0.5 + (0.3 * whitespace_ratio),
                        page_number=page.page_number,
                        description=f"High whitespace ratio ({whitespace_ratio:.0%}), {page.word_count} words"
                    )
        
        return None
    
    def _analyze_vlm_results(
        self,
        vlm_analysis: Dict[str, Any],
        page_number: int
    ) -> Optional[VisualSignal]:
        """Analyze vision-language model results for boundary detection."""
        if 'boundary_score' in vlm_analysis:
            score = vlm_analysis['boundary_score']
            if score > 0.7:
                return VisualSignal(
                    type=VisualSignalType.VISUAL_SEPARATOR_LINE,  # Using as generic visual signal
                    confidence=score,
                    page_number=page_number,
                    description=f"VLM boundary detection (score: {score:.2f})"
                )
        
        return None
    
    def _calculate_layout_similarity(
        self,
        prev_features: VisualFeatures,
        curr_features: VisualFeatures
    ) -> float:
        """Calculate layout similarity score between two pages."""
        similarity_factors = []
        
        # Column similarity
        if prev_features.num_columns == curr_features.num_columns:
            similarity_factors.append(1.0)
        else:
            col_diff = abs(prev_features.num_columns - curr_features.num_columns)
            similarity_factors.append(max(0, 1.0 - col_diff * 0.3))
        
        # Font size similarity
        if prev_features.primary_font_size and curr_features.primary_font_size:
            size_diff = abs(prev_features.primary_font_size - curr_features.primary_font_size)
            similarity_factors.append(max(0, 1.0 - size_diff / 10.0))
        
        # Margin similarity
        if all([prev_features.margin_top, curr_features.margin_top]):
            margin_diffs = [
                abs(prev_features.margin_top - curr_features.margin_top),
                abs(prev_features.margin_bottom - curr_features.margin_bottom) if prev_features.margin_bottom and curr_features.margin_bottom else 0,
                abs(prev_features.margin_left - curr_features.margin_left) if prev_features.margin_left and curr_features.margin_left else 0,
                abs(prev_features.margin_right - curr_features.margin_right) if prev_features.margin_right and curr_features.margin_right else 0
            ]
            avg_margin_diff = sum(margin_diffs) / 4
            similarity_factors.append(max(0, 1.0 - avg_margin_diff / 50.0))
        
        # Orientation similarity
        if prev_features.orientation == curr_features.orientation:
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.0)
        
        # Calculate average similarity
        return sum(similarity_factors) / len(similarity_factors) if similarity_factors else 0.5
    
    def _calculate_visual_continuity(
        self,
        pages: List[PageVisualInfo],
        current_index: int,
        window_size: int
    ) -> float:
        """Calculate visual continuity score based on surrounding pages."""
        if not pages or current_index >= len(pages):
            return 0.5
        
        curr_page = pages[current_index]
        if not curr_page.visual_features:
            return 0.5
        
        continuity_scores = []
        
        # Look at previous pages
        for i in range(max(0, current_index - window_size), current_index):
            if pages[i].visual_features:
                similarity = self._calculate_layout_similarity(
                    pages[i].visual_features,
                    curr_page.visual_features
                )
                # Weight by distance (closer pages have more weight)
                weight = 1.0 / (current_index - i)
                continuity_scores.append(similarity * weight)
        
        # Look at following pages
        for i in range(current_index + 1, min(len(pages), current_index + window_size + 1)):
            if pages[i].visual_features:
                similarity = self._calculate_layout_similarity(
                    curr_page.visual_features,
                    pages[i].visual_features
                )
                # Weight by distance
                weight = 1.0 / (i - current_index)
                continuity_scores.append(similarity * weight)
        
        if continuity_scores:
            # Lower continuity score suggests a boundary
            avg_continuity = sum(continuity_scores) / sum(1.0 / (i + 1) for i in range(len(continuity_scores)))
            return 1.0 - avg_continuity  # Invert so high score means likely boundary
        
        return 0.5
    
    def _calculate_visual_confidence(self, signals: List[VisualSignal]) -> float:
        """Calculate overall confidence from visual signals."""
        if not signals:
            return 0.0
        
        # Weighted average based on signal types
        total_weight = 0.0
        weighted_sum = 0.0
        
        for signal in signals:
            weight = self.VISUAL_SIGNAL_WEIGHTS.get(signal.type, 0.5)
            weighted_sum += signal.confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        base_confidence = weighted_sum / total_weight
        
        # Boost confidence if multiple strong visual signals
        strong_signals = sum(1 for s in signals if s.confidence >= 0.7)
        if strong_signals >= 2:
            base_confidence = min(1.0, base_confidence * 1.15)
        
        return base_confidence
    
    def infer_document_type_from_visual(
        self,
        pages: List[PageVisualInfo],
        start_idx: int,
        end_idx: int
    ) -> Optional[DocumentType]:
        """
        Infer document type based on visual characteristics.
        
        Args:
            pages: List of pages with visual features
            start_idx: Start index of document
            end_idx: End index of document
            
        Returns:
            Inferred document type or None
        """
        if not pages or start_idx >= len(pages):
            return None
        
        # Collect visual characteristics
        doc_pages = pages[start_idx:min(end_idx + 1, len(pages))]
        
        # Count various visual features
        has_logo = any(p.visual_features.has_logo for p in doc_pages if p.visual_features)
        has_signature = any(p.visual_features.has_signature for p in doc_pages if p.visual_features)
        avg_tables = sum(p.visual_features.num_tables for p in doc_pages if p.visual_features) / len(doc_pages)
        has_multi_column = any(p.visual_features.num_columns > 1 for p in doc_pages if p.visual_features)
        
        # Simple heuristics for document type inference
        if has_logo and avg_tables > 0.5:
            return DocumentType.INVOICE
        elif has_signature and len(doc_pages) <= 3:
            return DocumentType.LETTER
        elif avg_tables > 1.0:
            return DocumentType.REPORT
        elif has_multi_column and len(doc_pages) > 5:
            return DocumentType.REPORT
        elif has_logo and len(doc_pages) == 1:
            return DocumentType.FORM
        
        return DocumentType.OTHER