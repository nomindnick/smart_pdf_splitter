"""
Boundary detection module for identifying document boundaries in multi-document PDFs.

This module implements multiple signal detection methods to identify where one document
ends and another begins, using content patterns, visual signals, and confidence scoring.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .models import (
    PageInfo, 
    Boundary, 
    Signal, 
    SignalType, 
    DocumentType,
    BoundingBox
)
from .document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


@dataclass
class BoundaryCandidate:
    """Represents a potential document boundary with confidence scoring."""
    page_number: int
    signals: List[Signal]
    confidence: float
    suggested_type: Optional[DocumentType] = None
    
    @property
    def total_confidence(self) -> float:
        """Calculate total confidence from all signals."""
        if not self.signals:
            return 0.0
        return sum(s.confidence for s in self.signals) / len(self.signals)


class BoundaryDetector:
    """
    Detects document boundaries using multiple signals and patterns.
    
    Uses content analysis, visual signals, and pattern matching to identify
    where documents begin and end within a multi-document PDF.
    """
    
    # Email header patterns
    EMAIL_PATTERNS = {
        'from': re.compile(r'^From:\s*(.+)$', re.MULTILINE | re.IGNORECASE),
        'to': re.compile(r'^To:\s*(.+)$', re.MULTILINE | re.IGNORECASE),
        'subject': re.compile(r'^Subject:\s*(.+)$', re.MULTILINE | re.IGNORECASE),
        'date': re.compile(r'^Date:\s*(.+)$', re.MULTILINE | re.IGNORECASE),
        'sent': re.compile(r'^Sent:\s*(.+)$', re.MULTILINE | re.IGNORECASE),
    }
    
    # Document header patterns
    DOCUMENT_PATTERNS = {
        'invoice': re.compile(r'(?:invoice|inv)\s*(?:#|no\.?|number)?\s*:?\s*(\d+)', re.IGNORECASE),
        'purchase_order': re.compile(r'(?:purchase\s*order|p\.?o\.?)\s*(?:#|no\.?|number)?\s*:?\s*(\d+)', re.IGNORECASE),
        'quote': re.compile(r'(?:quote|quotation)\s*(?:#|no\.?|number)?\s*:?\s*(\d+)', re.IGNORECASE),
        'contract': re.compile(r'(?:contract)\s*(?:#|no\.?|number)?\s*:?\s*(\d+)', re.IGNORECASE),
        'submittal': re.compile(r'(?:submittal)\s*(?:#|no\.?|number)?\s*:?\s*(\d+)', re.IGNORECASE),
        'rfi': re.compile(r'(?:rfi|request\s*for\s*information)\s*(?:#|no\.?|number)?\s*:?\s*(\d+)', re.IGNORECASE),
        'proposal': re.compile(r'(?:proposal|cost\s*proposal)\s*(?:#|no\.?|number)?\s*:?\s*(\d+)', re.IGNORECASE),
        'application': re.compile(r'(?:application\s*for\s*payment)', re.IGNORECASE),
        'schedule': re.compile(r'(?:schedule\s*of\s*values)', re.IGNORECASE),
    }
    
    # Page number reset patterns
    PAGE_NUMBER_PATTERNS = [
        re.compile(r'^\s*(?:page\s*)?1\s*(?:of\s*\d+)?\s*$', re.IGNORECASE),
        re.compile(r'^\s*-?\s*1\s*-?\s*$'),  # Simple "1" or "-1-"
        re.compile(r'(?:page|p\.?)\s*1\b', re.IGNORECASE),
    ]
    
    # Signal weights for confidence calculation
    SIGNAL_WEIGHTS = {
        SignalType.EMAIL_HEADER: 0.9,
        SignalType.DOCUMENT_HEADER: 0.85,
        SignalType.PAGE_NUMBER_RESET: 0.7,
        SignalType.LAYOUT_CHANGE: 0.6,
        SignalType.WHITE_SPACE: 0.5,
        SignalType.DOCUMENT_TYPE_CHANGE: 0.8,
        SignalType.VISUAL_SEPARATOR: 0.6,
        SignalType.TEXT_PATTERN: 0.5,
    }
    
    def __init__(
        self,
        min_confidence: float = 0.6,
        min_signals: int = 1,
        enable_visual_analysis: bool = True
    ):
        """
        Initialize boundary detector.
        
        Args:
            min_confidence: Minimum confidence threshold for boundaries
            min_signals: Minimum number of signals required
            enable_visual_analysis: Whether to use visual signal detection
        """
        self.min_confidence = min_confidence
        self.min_signals = min_signals
        self.enable_visual_analysis = enable_visual_analysis
        self.document_processor = None
        
    def detect_boundaries(
        self,
        pages: List[PageInfo],
        context_window: int = 3
    ) -> List[Boundary]:
        """
        Detect document boundaries in a list of pages.
        
        Args:
            pages: List of PageInfo objects from document processor
            context_window: Number of pages to look ahead/behind for context
            
        Returns:
            List of detected boundaries with confidence scores
        """
        if not pages:
            return []
            
        boundaries = []
        candidates = []
        
        # Always mark first page as boundary
        first_boundary = Boundary(
            start_page=1,
            end_page=1,  # Will be updated later
            confidence=1.0,
            signals=[Signal(
                type=SignalType.DOCUMENT_HEADER,
                confidence=1.0,
                page_number=1,
                description="First page of document"
            )],
            document_type=self._detect_document_type(pages[0])
        )
        boundaries.append(first_boundary)
        
        # Analyze each page for boundary signals
        for i, page in enumerate(pages[1:], 1):  # Start from second page
            signals = []
            
            # Check for email headers
            email_signal = self._detect_email_header(page)
            if email_signal:
                signals.append(email_signal)
            
            # Check for document headers
            doc_signal = self._detect_document_header(page)
            if doc_signal:
                signals.append(doc_signal)
            
            # Check for page number reset
            page_num_signal = self._detect_page_number_reset(page)
            if page_num_signal:
                signals.append(page_num_signal)
            
            # Check for layout changes
            if i > 0:
                layout_signal = self._detect_layout_change(pages[i-1], page)
                if layout_signal:
                    signals.append(layout_signal)
            
            # Check for significant white space
            whitespace_signal = self._detect_white_space(page)
            if whitespace_signal:
                signals.append(whitespace_signal)
            
            # Check for document type change
            if i > 0:
                type_signal = self._detect_document_type_change(pages[i-1], page)
                if type_signal:
                    signals.append(type_signal)
            
            # Create boundary candidate if signals found
            if signals:
                candidate = BoundaryCandidate(
                    page_number=page.page_number,
                    signals=signals,
                    confidence=self._calculate_confidence(signals),
                    suggested_type=self._detect_document_type(page)
                )
                candidates.append(candidate)
        
        # Convert high-confidence candidates to boundaries
        for candidate in candidates:
            if (candidate.confidence >= self.min_confidence and 
                len(candidate.signals) >= self.min_signals):
                
                # Update the end page of the previous boundary
                if boundaries:
                    boundaries[-1].end_page = candidate.page_number - 1
                
                # Create new boundary
                boundary = Boundary(
                    start_page=candidate.page_number,
                    end_page=candidate.page_number,  # Will be updated by next boundary
                    confidence=candidate.confidence,
                    signals=candidate.signals,
                    document_type=candidate.suggested_type
                )
                boundaries.append(boundary)
        
        # Set the last boundary's end page
        if boundaries and pages:
            boundaries[-1].end_page = pages[-1].page_number
        
        logger.info(f"Detected {len(boundaries)} document boundaries")
        return boundaries
    
    def _detect_email_header(self, page: PageInfo) -> Optional[Signal]:
        """Detect email header patterns in page text."""
        if not page.text_content:
            return None
            
        # Check first 1000 characters for email headers
        text_sample = page.text_content[:1000]
        
        # Count matching email header patterns
        matches = 0
        details = []
        
        for pattern_name, pattern in self.EMAIL_PATTERNS.items():
            if pattern.search(text_sample):
                matches += 1
                details.append(pattern_name)
        
        # Need at least 2 email headers to be confident
        if matches >= 2:
            confidence = min(0.9, 0.3 * matches)  # Max 0.9 confidence
            return Signal(
                type=SignalType.EMAIL_HEADER,
                confidence=confidence,
                page_number=page.page_number,
                description=f"Email headers found: {', '.join(details)}"
            )
        
        return None
    
    def _detect_document_header(self, page: PageInfo) -> Optional[Signal]:
        """Detect document header patterns (invoice, PO, etc.)."""
        if not page.text_content:
            return None
            
        # Check first 500 characters
        text_sample = page.text_content[:500].lower()
        
        for doc_type, pattern in self.DOCUMENT_PATTERNS.items():
            if pattern.search(text_sample):
                return Signal(
                    type=SignalType.DOCUMENT_HEADER,
                    confidence=0.85,
                    page_number=page.page_number,
                    description=f"Document type detected: {doc_type}"
                )
        
        return None
    
    def _detect_page_number_reset(self, page: PageInfo) -> Optional[Signal]:
        """Detect if page shows 'Page 1' or similar reset."""
        if not page.text_content:
            return None
            
        # Check various locations for page numbers
        lines = page.text_content.split('\n')
        
        # Check first and last few lines
        check_lines = lines[:5] + lines[-5:] if len(lines) > 10 else lines
        
        for line in check_lines:
            for pattern in self.PAGE_NUMBER_PATTERNS:
                if pattern.search(line.strip()):
                    return Signal(
                        type=SignalType.PAGE_NUMBER_RESET,
                        confidence=0.7,
                        page_number=page.page_number,
                        description="Page number reset to 1"
                    )
        
        return None
    
    def _detect_layout_change(self, prev_page: PageInfo, curr_page: PageInfo) -> Optional[Signal]:
        """Detect significant layout changes between pages."""
        # Simple heuristic based on text density and structure
        prev_density = prev_page.word_count / (prev_page.width * prev_page.height) if prev_page.width > 0 else 0
        curr_density = curr_page.word_count / (curr_page.width * curr_page.height) if curr_page.width > 0 else 0
        
        # Check for significant density change (>50% difference)
        density_change = abs(curr_density - prev_density) / max(prev_density, curr_density, 0.001)
        
        if density_change > 0.5:
            return Signal(
                type=SignalType.LAYOUT_CHANGE,
                confidence=0.6,
                page_number=curr_page.page_number,
                description=f"Text density change: {density_change:.0%}"
            )
        
        return None
    
    def _detect_white_space(self, page: PageInfo) -> Optional[Signal]:
        """Detect if page has significant white space (mostly empty)."""
        if page.is_mostly_empty:
            # Calculate confidence based on how empty the page is
            confidence = 0.5 + (0.3 * (1 - page.word_count / 50))
            return Signal(
                type=SignalType.WHITE_SPACE,
                confidence=min(0.8, confidence),
                page_number=page.page_number,
                description=f"Page has only {page.word_count} words"
            )
        
        return None
    
    def _detect_document_type_change(self, prev_page: PageInfo, curr_page: PageInfo) -> Optional[Signal]:
        """Detect if document type changed between pages."""
        prev_type = self._detect_document_type(prev_page)
        curr_type = self._detect_document_type(curr_page)
        
        if prev_type and curr_type and prev_type != curr_type:
            return Signal(
                type=SignalType.DOCUMENT_TYPE_CHANGE,
                confidence=0.8,
                page_number=curr_page.page_number,
                description=f"Document type changed from {prev_type.value} to {curr_type.value}"
            )
        
        return None
    
    def _detect_document_type(self, page: PageInfo) -> Optional[DocumentType]:
        """Detect the type of document based on content."""
        if not page.text_content:
            return None
            
        text_lower = page.text_content[:1000].lower()
        
        # Check for email patterns
        email_matches = sum(1 for p in self.EMAIL_PATTERNS.values() if p.search(page.text_content[:500]))
        if email_matches >= 2:
            return DocumentType.EMAIL
        
        # Check for specific document types
        if any(p in text_lower for p in ['invoice', 'bill to', 'ship to', 'total amount']):
            return DocumentType.INVOICE
        
        if any(p in text_lower for p in ['purchase order', 'p.o.', 'po number']):
            return DocumentType.PURCHASE_ORDER
        
        if any(p in text_lower for p in ['quotation', 'quote', 'estimate']):
            return DocumentType.QUOTE
        
        if 'contract' in text_lower and 'agreement' in text_lower:
            return DocumentType.CONTRACT
        
        if any(p in text_lower for p in ['request for information', 'rfi']):
            return DocumentType.REPORT
        
        if 'submittal' in text_lower:
            return DocumentType.REPORT
        
        if any(p in text_lower for p in ['proposal', 'cost proposal']):
            return DocumentType.REPORT
        
        if 'schedule of values' in text_lower:
            return DocumentType.REPORT
        
        if 'application for payment' in text_lower:
            return DocumentType.FORM
        
        # Check for report-like content
        if any(p in text_lower for p in ['executive summary', 'table of contents', 'introduction', 'conclusion']):
            return DocumentType.REPORT
        
        # Check for letter patterns
        if any(p in text_lower for p in ['dear', 'sincerely', 'regards', 'yours truly']):
            return DocumentType.LETTER
        
        return DocumentType.OTHER
    
    def _calculate_confidence(self, signals: List[Signal]) -> float:
        """Calculate overall confidence from multiple signals."""
        if not signals:
            return 0.0
        
        # Weighted average based on signal types
        total_weight = 0.0
        weighted_sum = 0.0
        
        for signal in signals:
            weight = self.SIGNAL_WEIGHTS.get(signal.type, 0.5)
            weighted_sum += signal.confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
            
        # Base confidence from weighted average
        base_confidence = weighted_sum / total_weight
        
        # Boost confidence if multiple strong signals
        strong_signals = sum(1 for s in signals if s.confidence >= 0.7)
        if strong_signals >= 2:
            base_confidence = min(1.0, base_confidence * 1.2)
        
        return base_confidence
    
    def refine_boundaries(
        self,
        boundaries: List[Boundary],
        pages: List[PageInfo],
        user_feedback: Optional[Dict[int, bool]] = None
    ) -> List[Boundary]:
        """
        Refine boundaries based on additional analysis or user feedback.
        
        Args:
            boundaries: Initial boundary detections
            pages: All pages in the document
            user_feedback: Optional dict of {page_number: is_boundary}
            
        Returns:
            Refined list of boundaries
        """
        refined = boundaries.copy()
        
        # Apply user feedback if provided
        if user_feedback:
            # Remove boundaries marked as false by user
            refined = [b for b in refined if user_feedback.get(b.start_page, True)]
            
            # Add boundaries marked as true by user
            for page_num, is_boundary in user_feedback.items():
                if is_boundary and not any(b.start_page == page_num for b in refined):
                    # Find appropriate position to insert
                    insert_pos = 0
                    for i, b in enumerate(refined):
                        if b.start_page > page_num:
                            insert_pos = i
                            break
                    else:
                        insert_pos = len(refined)
                    
                    # Create new boundary
                    new_boundary = Boundary(
                        start_page=page_num,
                        end_page=page_num,
                        confidence=1.0,  # User-specified
                        signals=[Signal(
                            type=SignalType.TEXT_PATTERN,
                            confidence=1.0,
                            page_number=page_num,
                            description="User-specified boundary"
                        )],
                        document_type=self._detect_document_type(pages[page_num - 1])
                    )
                    refined.insert(insert_pos, new_boundary)
            
            # Recalculate end pages
            for i in range(len(refined) - 1):
                refined[i].end_page = refined[i + 1].start_page - 1
            if refined and pages:
                refined[-1].end_page = pages[-1].page_number
        
        return refined