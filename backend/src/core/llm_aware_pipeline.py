"""LLM-aware processing pipeline that balances speed and quality."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import fitz  # PyMuPDF

from .ocr_config import OCRConfig
from .models import PageInfo, Boundary
from .ocr_optimizer import check_page_needs_ocr

logger = logging.getLogger(__name__)


class BoundaryDetectionMethod(str, Enum):
    """Methods for boundary detection."""
    TEXT_PATTERN = "text_pattern"      # Fast, pattern matching
    VISUAL = "visual"                  # Medium, layout analysis  
    LLM = "llm"                       # Slow, high quality
    HYBRID = "hybrid"                 # Intelligent combination


@dataclass
class PageContext:
    """Context for a page including surrounding pages."""
    page_number: int
    text: str
    confidence: float
    previous_text: Optional[str] = None
    next_text: Optional[str] = None
    visual_features: Optional[Dict[str, Any]] = None


class LLMAwarePipeline:
    """
    Pipeline that intelligently manages OCR quality based on detection method.
    
    Key insight: Not all pages need high-quality OCR, but pages near
    potential boundaries do, especially for LLM analysis.
    """
    
    def __init__(self, primary_method: BoundaryDetectionMethod = BoundaryDetectionMethod.HYBRID):
        self.primary_method = primary_method
        self.logger = logging.getLogger(__name__)
        
    def create_adaptive_ocr_strategy(self, pdf_path: Path) -> Dict[int, OCRConfig]:
        """
        Create page-specific OCR strategies based on detection needs.
        
        Returns:
            Dict mapping page numbers to OCR configs
        """
        import fitz
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        
        # First pass: Identify potential boundary pages
        potential_boundaries = self._identify_potential_boundaries(doc)
        
        # Create page-specific strategies
        page_configs = {}
        
        for page_num in range(total_pages):
            # Determine if this page needs high-quality OCR
            needs_high_quality = self._page_needs_high_quality_ocr(
                page_num,
                potential_boundaries,
                total_pages
            )
            
            if needs_high_quality:
                # High quality for LLM analysis
                page_configs[page_num] = self._create_llm_quality_config()
            else:
                # Fast config for basic detection
                page_configs[page_num] = self._create_fast_detection_config()
        
        doc.close()
        
        logger.info(
            f"Created adaptive OCR strategy: "
            f"{sum(1 for c in page_configs.values() if c.target_dpi >= 300)} high-quality pages, "
            f"{sum(1 for c in page_configs.values() if c.target_dpi < 300)} fast pages"
        )
        
        return page_configs
    
    def _identify_potential_boundaries(self, doc: fitz.Document) -> List[int]:
        """
        Quick scan to identify potential boundary pages.
        
        This uses fast heuristics to find pages that might be boundaries,
        so we can focus high-quality OCR on those areas.
        """
        potential_boundaries = []
        
        for i, page in enumerate(doc):
            # Quick text extraction (no OCR)
            text = page.get_text()
            
            # Check for boundary indicators
            if self._has_boundary_indicators(text, i, len(doc)):
                potential_boundaries.append(i)
            
            # Check visual layout changes
            if i > 0 and self._has_layout_change(doc[i-1], page):
                potential_boundaries.append(i)
        
        return potential_boundaries
    
    def _has_boundary_indicators(self, text: str, page_num: int, total_pages: int) -> bool:
        """Check if page has indicators of being a document boundary."""
        text_lower = text.lower()
        
        # First/last pages are always potential boundaries
        if page_num == 0 or page_num == total_pages - 1:
            return True
        
        # Look for document start indicators
        indicators = [
            "from:", "to:", "subject:",  # Email
            "invoice", "bill", "statement",  # Financial
            "page 1", "page 1 of",  # Page numbering
            "dear", "sincerely",  # Letters
            "contract", "agreement",  # Legal
        ]
        
        return any(indicator in text_lower for indicator in indicators)
    
    def _has_layout_change(self, prev_page: fitz.Page, curr_page: fitz.Page) -> bool:
        """Detect significant layout changes between pages."""
        # Compare text density
        prev_text_len = len(prev_page.get_text())
        curr_text_len = len(curr_page.get_text())
        
        # Significant change in text amount
        if prev_text_len > 100 and curr_text_len < 50:
            return True
        if prev_text_len < 50 and curr_text_len > 100:
            return True
        
        # Could add more sophisticated layout analysis here
        return False
    
    def _page_needs_high_quality_ocr(
        self,
        page_num: int,
        potential_boundaries: List[int],
        total_pages: int
    ) -> bool:
        """
        Determine if a page needs high-quality OCR for LLM analysis.
        
        Rules:
        1. Potential boundary pages need high quality
        2. Context pages (before/after boundaries) need high quality
        3. For LLM method, sample more pages for context
        """
        # Always high quality for potential boundaries
        if page_num in potential_boundaries:
            return True
        
        # Context window around boundaries
        context_window = 2 if self.primary_method == BoundaryDetectionMethod.LLM else 1
        
        for boundary in potential_boundaries:
            if abs(page_num - boundary) <= context_window:
                return True
        
        # For LLM method, sample additional pages for context
        if self.primary_method == BoundaryDetectionMethod.LLM:
            # Every 5th page for context continuity
            if page_num % 5 == 0:
                return True
        
        # For very small documents, process everything with high quality
        if total_pages <= 10 and self.primary_method in [BoundaryDetectionMethod.LLM, BoundaryDetectionMethod.HYBRID]:
            return True
        
        return False
    
    def _create_llm_quality_config(self) -> OCRConfig:
        """Create high-quality OCR config for LLM analysis."""
        return OCRConfig(
            enable_ocr=True,
            ocr_engine="easyocr",  # Better for complex layouts
            ocr_languages=["en"],
            
            # High quality settings
            target_dpi=300,  # High resolution
            force_full_page_ocr=True,  # Get all text
            bitmap_area_threshold=0.05,  # Process small text too
            
            # Comprehensive preprocessing
            enable_preprocessing=True,
            preprocessing_steps=["deskew", "denoise", "contrast"],
            
            # Enable corrections for better LLM understanding
            enable_postprocessing=True,
            apply_aggressive_corrections=True,
            confidence_threshold=0.7
        )
    
    def _create_fast_detection_config(self) -> OCRConfig:
        """Create fast OCR config for basic detection."""
        return OCRConfig(
            enable_ocr=True,
            ocr_engine="tesseract-cli",  # Faster
            ocr_languages=["eng"],
            
            # Speed optimizations
            target_dpi=200,  # Moderate resolution
            force_full_page_ocr=False,
            bitmap_area_threshold=0.1,
            
            # Minimal preprocessing
            enable_preprocessing=True,
            preprocessing_steps=["deskew"],
            
            # Skip corrections for speed
            enable_postprocessing=False,
            confidence_threshold=0.5
        )
    
    def process_for_boundary_detection(
        self,
        pdf_path: Path,
        pages: List[PageInfo],
        method: Optional[BoundaryDetectionMethod] = None
    ) -> List[PageContext]:
        """
        Process pages with appropriate quality for boundary detection method.
        
        Returns:
            List of PageContext objects with text and context
        """
        method = method or self.primary_method
        contexts = []
        
        for i, page in enumerate(pages):
            # Build context including surrounding pages
            context = PageContext(
                page_number=page.page_number,
                text=page.text_content or "",
                confidence=page.ocr_confidence or 0.0
            )
            
            # Add surrounding context for LLM
            if method in [BoundaryDetectionMethod.LLM, BoundaryDetectionMethod.HYBRID]:
                if i > 0:
                    context.previous_text = pages[i-1].text_content
                if i < len(pages) - 1:
                    context.next_text = pages[i+1].text_content
            
            contexts.append(context)
        
        return contexts
    
    def optimize_for_method(self, method: BoundaryDetectionMethod) -> Dict[str, Any]:
        """
        Get optimization recommendations for specific detection method.
        """
        if method == BoundaryDetectionMethod.TEXT_PATTERN:
            return {
                "ocr_quality": "low",
                "preprocessing": "minimal",
                "context_needed": False,
                "parallel_safe": True,
                "estimated_speed": "fast"
            }
        
        elif method == BoundaryDetectionMethod.VISUAL:
            return {
                "ocr_quality": "medium",
                "preprocessing": "basic",
                "context_needed": False,
                "parallel_safe": True,
                "estimated_speed": "medium"
            }
        
        elif method == BoundaryDetectionMethod.LLM:
            return {
                "ocr_quality": "high",
                "preprocessing": "comprehensive",
                "context_needed": True,
                "parallel_safe": False,  # Need sequential for context
                "estimated_speed": "slow"
            }
        
        else:  # HYBRID
            return {
                "ocr_quality": "adaptive",
                "preprocessing": "adaptive",
                "context_needed": True,
                "parallel_safe": "partial",  # Parallel for OCR, sequential for LLM
                "estimated_speed": "medium"
            }


class SmartBoundaryPipeline:
    """
    Pipeline that uses multiple methods intelligently.
    
    Strategy:
    1. Fast text pattern detection first
    2. Visual detection for ambiguous cases  
    3. LLM for complex boundaries
    4. Combine results with confidence weights
    """
    
    def __init__(self):
        self.llm_aware_pipeline = LLMAwarePipeline(BoundaryDetectionMethod.HYBRID)
        
    def detect_boundaries_smart(
        self,
        pdf_path: Path,
        enable_llm: bool = True,
        llm_threshold: float = 0.7
    ) -> List[Boundary]:
        """
        Detect boundaries using intelligent method selection.
        
        Args:
            pdf_path: Path to PDF
            enable_llm: Whether to use LLM for difficult cases
            llm_threshold: Confidence threshold to trigger LLM
        """
        # Step 1: Create adaptive OCR strategy
        page_configs = self.llm_aware_pipeline.create_adaptive_ocr_strategy(pdf_path)
        
        # Step 2: Process pages with appropriate quality
        # (This would integrate with your existing processors)
        
        # Step 3: Run detection methods in order of speed
        results = []
        
        # Fast text patterns
        text_boundaries = self._detect_with_text_patterns(pdf_path)
        results.extend(text_boundaries)
        
        # Medium-speed visual detection
        visual_boundaries = self._detect_with_visual_analysis(pdf_path)
        results.extend(visual_boundaries)
        
        # Identify low-confidence boundaries that need LLM
        if enable_llm:
            low_confidence = [b for b in results if b.confidence < llm_threshold]
            if low_confidence:
                # Process only these pages with high-quality OCR for LLM
                llm_boundaries = self._detect_with_llm(pdf_path, low_confidence)
                # Update results with LLM insights
                results = self._merge_results(results, llm_boundaries)
        
        return results
    
    def _detect_with_text_patterns(self, pdf_path: Path) -> List[Boundary]:
        """Fast text pattern detection."""
        # Placeholder - would use your existing text detector
        return []
    
    def _detect_with_visual_analysis(self, pdf_path: Path) -> List[Boundary]:
        """Visual layout analysis."""
        # Placeholder - would use your existing visual detector
        return []
    
    def _detect_with_llm(self, pdf_path: Path, candidates: List[Boundary]) -> List[Boundary]:
        """LLM analysis for difficult cases."""
        # Placeholder - would use your existing LLM detector
        return []
    
    def _merge_results(self, results: List[Boundary], llm_results: List[Boundary]) -> List[Boundary]:
        """Merge results from different methods."""
        # Combine results, preferring LLM decisions for conflicts
        merged = {}
        for b in results:
            merged[b.start_page] = b
        for b in llm_results:
            merged[b.start_page] = b  # LLM overrides
        return list(merged.values())