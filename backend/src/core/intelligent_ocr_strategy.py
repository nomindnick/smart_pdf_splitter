"""Intelligent OCR strategy that adapts based on detection method."""

import logging
from typing import Dict, List, Optional, Set, Any
from pathlib import Path

import fitz  # PyMuPDF

from .models import PageInfo
from .pipeline_config import PipelineProfiles
from .llm_aware_pipeline import LLMAwarePipeline, BoundaryDetectionMethod

logger = logging.getLogger(__name__)


class IntelligentOCRStrategy:
    """
    Strategy that adapts OCR quality based on what boundary detection
    methods will be used and what pages are likely boundaries.
    
    Key principle: Invest OCR time where it matters most.
    """
    
    def __init__(self, detection_methods: List[str]):
        """
        Initialize with the detection methods that will be used.
        
        Args:
            detection_methods: List of methods like ["text", "visual", "llm"]
        """
        self.detection_methods = detection_methods
        self.uses_llm = "llm" in detection_methods
        self.llm_pipeline = LLMAwarePipeline()
        
    def plan_ocr_strategy(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Create an intelligent OCR plan for the document.
        
        Returns a strategy that includes:
        - Which pages to OCR with high quality
        - Which pages can use fast OCR
        - Which pages to skip
        - Optimal processing order
        """
        import fitz
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        
        # Phase 1: Quick document analysis (no OCR)
        doc_profile = self._profile_document(doc)
        
        # Phase 2: Identify critical pages
        critical_pages = self._identify_critical_pages(doc, doc_profile)
        
        # Phase 3: Create page-specific strategies
        page_strategies = {}
        
        for page_num in range(total_pages):
            if page_num in critical_pages["high_quality"]:
                # These pages need the best OCR
                strategy = "high_quality"
                config = PipelineProfiles.get_llm_detection_config()
            elif page_num in critical_pages["medium_quality"]:
                # These pages need decent OCR
                strategy = "medium_quality"
                config = self._get_medium_quality_config()
            elif page_num in critical_pages["skip"]:
                # These pages can be skipped
                strategy = "skip"
                config = None
            else:
                # Default: fast OCR
                strategy = "fast"
                config = PipelineProfiles.get_splitter_detection_config()
            
            page_strategies[page_num] = {
                "strategy": strategy,
                "config": config,
                "reason": critical_pages.get("reasons", {}).get(page_num, "default")
            }
        
        doc.close()
        
        # Phase 4: Optimize processing order
        processing_order = self._optimize_processing_order(
            page_strategies,
            doc_profile
        )
        
        # Create final strategy
        strategy = {
            "total_pages": total_pages,
            "page_strategies": page_strategies,
            "processing_order": processing_order,
            "estimated_time": self._estimate_processing_time(page_strategies),
            "quality_summary": {
                "high_quality_pages": len(critical_pages["high_quality"]),
                "medium_quality_pages": len(critical_pages["medium_quality"]),
                "fast_pages": sum(1 for s in page_strategies.values() 
                                if s["strategy"] == "fast"),
                "skipped_pages": len(critical_pages["skip"])
            }
        }
        
        logger.info(
            f"OCR Strategy: {strategy['quality_summary']['high_quality_pages']} high, "
            f"{strategy['quality_summary']['medium_quality_pages']} medium, "
            f"{strategy['quality_summary']['fast_pages']} fast, "
            f"{strategy['quality_summary']['skipped_pages']} skip"
        )
        
        return strategy
    
    def _profile_document(self, doc: fitz.Document) -> Dict[str, Any]:
        """Quick document profiling without OCR."""
        profile = {
            "total_pages": len(doc),
            "has_text_pages": 0,
            "has_image_pages": 0,
            "potential_boundaries": [],
            "page_characteristics": {}
        }
        
        for i, page in enumerate(doc):
            # Quick text check
            text = page.get_text()
            has_text = len(text.strip()) > 50
            
            # Image check
            images = page.get_images()
            has_images = len(images) > 0
            
            if has_text:
                profile["has_text_pages"] += 1
            if has_images and not has_text:
                profile["has_image_pages"] += 1
            
            # Store characteristics
            profile["page_characteristics"][i] = {
                "has_embedded_text": has_text,
                "has_images": has_images,
                "text_length": len(text),
                "image_count": len(images)
            }
            
            # Quick boundary check
            if self._is_potential_boundary(text, i, len(doc)):
                profile["potential_boundaries"].append(i)
        
        return profile
    
    def _identify_critical_pages(
        self,
        doc: fitz.Document,
        profile: Dict[str, Any]
    ) -> Dict[str, Set[int]]:
        """Identify which pages need which quality level."""
        critical_pages = {
            "high_quality": set(),
            "medium_quality": set(),
            "skip": set(),
            "reasons": {}
        }
        
        # For LLM detection, we need high quality around boundaries
        if self.uses_llm:
            # All potential boundaries need high quality
            for boundary_page in profile["potential_boundaries"]:
                critical_pages["high_quality"].add(boundary_page)
                critical_pages["reasons"][boundary_page] = "potential_boundary"
                
                # Context pages (before and after)
                for offset in [-2, -1, 1, 2]:
                    context_page = boundary_page + offset
                    if 0 <= context_page < profile["total_pages"]:
                        if context_page not in critical_pages["high_quality"]:
                            critical_pages["medium_quality"].add(context_page)
                            critical_pages["reasons"][context_page] = "boundary_context"
            
            # For small documents with LLM, process everything with decent quality
            if profile["total_pages"] <= 10:
                for i in range(profile["total_pages"]):
                    if i not in critical_pages["high_quality"]:
                        critical_pages["medium_quality"].add(i)
                        critical_pages["reasons"][i] = "small_document"
        
        # Pages that already have good embedded text can potentially be skipped
        for page_num, chars in profile["page_characteristics"].items():
            if (chars["has_embedded_text"] and 
                chars["text_length"] > 500 and 
                not chars["has_images"] and
                page_num not in critical_pages["high_quality"] and
                page_num not in critical_pages["medium_quality"]):
                
                # Only skip if we're not using LLM or it's not near a boundary
                if not self.uses_llm or page_num not in profile["potential_boundaries"]:
                    critical_pages["skip"].add(page_num)
                    critical_pages["reasons"][page_num] = "has_embedded_text"
        
        return critical_pages
    
    def _is_potential_boundary(self, text: str, page_num: int, total_pages: int) -> bool:
        """Quick check if page might be a boundary."""
        # First/last pages
        if page_num == 0 or page_num == total_pages - 1:
            return True
        
        # Common boundary indicators
        text_lower = text.lower()
        indicators = [
            # Email
            "from:", "to:", "subject:", "date:",
            # Documents
            "invoice", "purchase order", "contract",
            "page 1", "1 of",
            # Letters
            "dear", "sincerely", "regards",
            # Forms
            "application", "form", "questionnaire"
        ]
        
        return any(indicator in text_lower for indicator in indicators)
    
    def _get_medium_quality_config(self):
        """Get medium quality OCR config."""
        from .ocr_config import OCRConfig
        return OCRConfig(
            enable_ocr=True,
            ocr_engine="tesseract-cli",  # Faster than EasyOCR
            ocr_languages=["eng"],
            target_dpi=200,              # Moderate quality
            force_full_page_ocr=False,
            bitmap_area_threshold=0.1,
            enable_preprocessing=True,
            preprocessing_steps=["deskew", "denoise"],
            enable_postprocessing=True,  # Some corrections
            confidence_threshold=0.6,
            max_processing_time=20
        )
    
    def _optimize_processing_order(
        self,
        page_strategies: Dict[int, Dict],
        profile: Dict[str, Any]
    ) -> List[int]:
        """
        Optimize the order of page processing.
        
        Strategy:
        1. Process high-quality pages first (for early boundary detection)
        2. Then process pages near boundaries
        3. Finally process remaining pages
        """
        high_quality = []
        medium_quality = []
        fast = []
        
        for page_num, strategy in page_strategies.items():
            if strategy["strategy"] == "high_quality":
                high_quality.append(page_num)
            elif strategy["strategy"] == "medium_quality":
                medium_quality.append(page_num)
            elif strategy["strategy"] == "fast":
                fast.append(page_num)
        
        # Process potential boundaries first
        high_quality.sort(key=lambda p: (
            0 if p in profile["potential_boundaries"] else 1,
            p
        ))
        
        return high_quality + medium_quality + fast
    
    def _estimate_processing_time(self, page_strategies: Dict[int, Dict]) -> float:
        """Estimate total processing time based on strategies."""
        time_estimates = {
            "high_quality": 15.0,   # 15 seconds per page
            "medium_quality": 8.0,  # 8 seconds per page
            "fast": 3.0,           # 3 seconds per page
            "skip": 0.1            # 0.1 seconds to check
        }
        
        total_time = 0
        for strategy in page_strategies.values():
            total_time += time_estimates.get(strategy["strategy"], 5.0)
        
        return total_time