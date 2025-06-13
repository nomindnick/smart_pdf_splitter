"""Adaptive OCR configuration module."""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
import cv2
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentQuality(str, Enum):
    """Document quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good" 
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"


class DocumentCharacteristics(BaseModel):
    """Characteristics of a document for OCR configuration."""
    
    quality: DocumentQuality = DocumentQuality.FAIR
    is_scanned: bool = True
    is_handwritten: bool = False
    has_tables: bool = False
    has_multiple_columns: bool = False
    average_font_size: Optional[float] = None
    estimated_dpi: int = 150
    noise_level: float = 0.0
    skew_angle: float = 0.0
    contrast_ratio: float = 1.0
    page_count: int = 1
    dominant_language: str = "en"


class OCRConfig(BaseModel):
    """Enhanced OCR configuration with adaptive settings."""
    
    # Basic settings
    enable_ocr: bool = True
    ocr_engine: str = "auto"  # auto, easyocr, tesserocr, tesseract-cli
    ocr_languages: List[str] = Field(default_factory=lambda: ["en"])
    
    # Preprocessing settings
    enable_preprocessing: bool = True
    preprocessing_steps: List[str] = Field(
        default_factory=lambda: ["deskew", "denoise", "contrast"]
    )
    target_dpi: int = 300
    quality_threshold: float = 0.7
    
    # Performance settings
    enable_gpu: Optional[bool] = None  # Auto-detect if None
    max_processing_time: int = 30  # seconds per page
    page_batch_size: int = 4
    memory_limit_mb: int = 4096
    
    # Quality settings
    confidence_threshold: float = 0.7
    enable_postprocessing: bool = True
    apply_aggressive_corrections: bool = False
    
    # Advanced settings
    force_full_page_ocr: bool = False
    bitmap_area_threshold: float = 0.05
    enable_multi_engine_voting: bool = False
    cache_results: bool = True


class AdaptiveOCRConfigurator:
    """Dynamically adjust OCR settings based on document analysis."""
    
    def __init__(self):
        """Initialize configurator with default profiles."""
        self.quality_profiles = {
            DocumentQuality.EXCELLENT: {
                "target_dpi": 150,
                "preprocessing_steps": ["deskew"],
                "enable_gpu": False,
                "page_batch_size": 8
            },
            DocumentQuality.GOOD: {
                "target_dpi": 200,
                "preprocessing_steps": ["deskew", "contrast"],
                "enable_gpu": False,
                "page_batch_size": 4
            },
            DocumentQuality.FAIR: {
                "target_dpi": 300,
                "preprocessing_steps": ["deskew", "denoise", "contrast"],
                "enable_gpu": True,
                "page_batch_size": 2
            },
            DocumentQuality.POOR: {
                "target_dpi": 300,
                "preprocessing_steps": ["upscale", "denoise", "deskew", "contrast", "threshold"],
                "enable_gpu": True,
                "page_batch_size": 1,
                "apply_aggressive_corrections": True
            },
            DocumentQuality.VERY_POOR: {
                "target_dpi": 400,
                "preprocessing_steps": ["upscale", "denoise", "deskew", "contrast", "threshold", "borders"],
                "enable_gpu": True,
                "page_batch_size": 1,
                "apply_aggressive_corrections": True,
                "force_full_page_ocr": True
            }
        }
    
    def analyze_document(
        self,
        sample_pages: List[np.ndarray],
        page_count: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentCharacteristics:
        """
        Analyze document characteristics from sample pages.
        
        Args:
            sample_pages: Sample page images as numpy arrays
            page_count: Total number of pages
            metadata: Optional document metadata
            
        Returns:
            DocumentCharacteristics object
        """
        if not sample_pages:
            return DocumentCharacteristics(page_count=page_count)
        
        # Analyze quality metrics
        quality_scores = []
        noise_levels = []
        skew_angles = []
        contrast_ratios = []
        
        for page_img in sample_pages:
            # Assess quality
            quality = self._assess_page_quality(page_img)
            quality_scores.append(quality["overall_score"])
            noise_levels.append(quality["noise_level"])
            skew_angles.append(abs(quality["skew_angle"]))
            contrast_ratios.append(quality["contrast_ratio"])
        
        # Average metrics
        avg_quality = np.mean(quality_scores)
        avg_noise = np.mean(noise_levels)
        avg_skew = np.mean(skew_angles)
        avg_contrast = np.mean(contrast_ratios)
        
        # Determine quality level
        if avg_quality >= 0.85:
            quality = DocumentQuality.EXCELLENT
        elif avg_quality >= 0.7:
            quality = DocumentQuality.GOOD
        elif avg_quality >= 0.5:
            quality = DocumentQuality.FAIR
        elif avg_quality >= 0.3:
            quality = DocumentQuality.POOR
        else:
            quality = DocumentQuality.VERY_POOR
        
        # Check if scanned (all images)
        is_scanned = all(self._is_page_scanned(page) for page in sample_pages)
        
        # Estimate DPI
        estimated_dpi = self._estimate_dpi(sample_pages[0]) if sample_pages else 150
        
        # Detect other characteristics
        has_tables = any(self._detect_tables(page) for page in sample_pages)
        has_columns = any(self._detect_columns(page) for page in sample_pages)
        
        return DocumentCharacteristics(
            quality=quality,
            is_scanned=is_scanned,
            is_handwritten=False,  # TODO: Implement handwriting detection
            has_tables=has_tables,
            has_multiple_columns=has_columns,
            estimated_dpi=estimated_dpi,
            noise_level=avg_noise,
            skew_angle=avg_skew,
            contrast_ratio=avg_contrast,
            page_count=page_count,
            dominant_language=metadata.get("language", "en") if metadata else "en"
        )
    
    def generate_config(
        self,
        characteristics: DocumentCharacteristics,
        base_config: Optional[OCRConfig] = None,
        hardware_info: Optional[Dict[str, Any]] = None
    ) -> OCRConfig:
        """
        Generate optimal OCR configuration based on document characteristics.
        
        Args:
            characteristics: Document characteristics
            base_config: Base configuration to modify
            hardware_info: Hardware capabilities
            
        Returns:
            Optimized OCRConfig
        """
        # Start with base config or default
        config = base_config or OCRConfig()
        
        # Get quality profile
        profile = self.quality_profiles.get(
            characteristics.quality,
            self.quality_profiles[DocumentQuality.FAIR]
        )
        
        # Apply profile settings
        config.target_dpi = profile["target_dpi"]
        config.preprocessing_steps = profile["preprocessing_steps"].copy()
        config.page_batch_size = profile["page_batch_size"]
        
        # GPU settings
        if hardware_info and "gpu_available" in hardware_info:
            config.enable_gpu = hardware_info["gpu_available"] and profile.get("enable_gpu", False)
        else:
            config.enable_gpu = profile.get("enable_gpu", None)
        
        # Aggressive corrections for poor quality
        config.apply_aggressive_corrections = profile.get("apply_aggressive_corrections", False)
        config.force_full_page_ocr = profile.get("force_full_page_ocr", False)
        
        # Adjust for specific characteristics
        if characteristics.is_handwritten:
            config.preprocessing_steps.append("threshold")
            config.ocr_engine = "easyocr"  # Better for handwriting
            config.force_full_page_ocr = True
        
        if characteristics.has_tables:
            config.preprocessing_steps.append("borders")
            config.bitmap_area_threshold = 0.02  # Lower threshold for tables
        
        if characteristics.skew_angle > 1.0 and "deskew" not in config.preprocessing_steps:
            config.preprocessing_steps.insert(0, "deskew")
        
        if characteristics.noise_level > 20 and "denoise" not in config.preprocessing_steps:
            config.preprocessing_steps.insert(0, "denoise")
        
        # Optimize for large documents
        if characteristics.page_count > 100:
            config.page_batch_size = min(config.page_batch_size, 2)
            config.enable_multi_engine_voting = False  # Too slow for large docs
            
        # Language settings
        if characteristics.dominant_language != "en":
            config.ocr_languages = [characteristics.dominant_language, "en"]
        
        # Select OCR engine
        if config.ocr_engine == "auto":
            config.ocr_engine = self._select_optimal_engine(
                characteristics,
                config.enable_gpu
            )
        
        return config
    
    def _assess_page_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Assess page quality metrics."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        sharpness_score = min(sharpness / 500, 1.0)
        
        # Contrast
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cumsum = np.cumsum(hist)
        total = cumsum[-1]
        
        lower_percentile = np.searchsorted(cumsum, 0.05 * total)
        upper_percentile = np.searchsorted(cumsum, 0.95 * total)
        contrast_range = upper_percentile - lower_percentile
        contrast_ratio = contrast_range / 255
        contrast_score = min(contrast_ratio * 2, 1.0)
        
        # Noise level
        noise_level = np.std(laplacian)
        noise_score = max(0, 1.0 - (noise_level / 100))
        
        # Skew detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        skew_angle = 0.0
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = (theta * 180 / np.pi) - 90
                if -45 <= angle <= 45:
                    angles.append(angle)
            if angles:
                skew_angle = np.median(angles)
        
        # Overall score
        overall_score = (sharpness_score + contrast_score + noise_score) / 3
        
        return {
            "overall_score": overall_score,
            "sharpness": sharpness,
            "contrast_ratio": contrast_ratio,
            "noise_level": noise_level,
            "skew_angle": skew_angle
        }
    
    def _is_page_scanned(self, image: np.ndarray) -> bool:
        """Check if page appears to be scanned."""
        # Simple heuristic: check if image has scanner artifacts
        # Real implementation would be more sophisticated
        
        # Check for black borders (common in scans)
        h, w = image.shape[:2]
        border_size = 20
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Check borders
        top_border = gray[:border_size, :]
        bottom_border = gray[-border_size:, :]
        left_border = gray[:, :border_size]
        right_border = gray[:, -border_size:]
        
        # If borders are mostly black, likely scanned
        border_mean = np.mean([
            np.mean(top_border),
            np.mean(bottom_border),
            np.mean(left_border),
            np.mean(right_border)
        ])
        
        return border_mean < 50  # Dark borders indicate scan
    
    def _estimate_dpi(self, image: np.ndarray) -> int:
        """Estimate DPI of the image."""
        h, w = image.shape[:2]
        
        # Assume standard page sizes and estimate DPI
        # A4: 210mm x 297mm = 8.27" x 11.69"
        # Letter: 8.5" x 11"
        
        # Check aspect ratio
        aspect_ratio = h / w
        
        if 1.3 < aspect_ratio < 1.5:  # Likely portrait
            # Assume ~11" height
            estimated_dpi = int(h / 11)
        else:
            # Assume ~8.5" width
            estimated_dpi = int(w / 8.5)
        
        # Round to common DPI values
        common_dpis = [72, 96, 150, 200, 300, 400, 600]
        closest_dpi = min(common_dpis, key=lambda x: abs(x - estimated_dpi))
        
        return closest_dpi
    
    def _detect_tables(self, image: np.ndarray) -> bool:
        """Detect if image contains tables."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect horizontal and vertical lines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=100,
            minLineLength=100, maxLineGap=10
        )
        
        if lines is None:
            return False
        
        # Count horizontal and vertical lines
        horizontal_lines = 0
        vertical_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 10 or angle > 170:  # Horizontal
                horizontal_lines += 1
            elif 80 < angle < 100:  # Vertical
                vertical_lines += 1
        
        # If we have both horizontal and vertical lines, likely a table
        return horizontal_lines > 3 and vertical_lines > 3
    
    def _detect_columns(self, image: np.ndarray) -> bool:
        """Detect if image has multiple columns."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Simple vertical projection profile
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        vertical_projection = np.sum(binary, axis=0)
        
        # Look for gaps in projection (column separators)
        threshold = np.mean(vertical_projection) * 0.3
        gaps = vertical_projection < threshold
        
        # Count significant gaps
        gap_regions = []
        in_gap = False
        gap_start = 0
        
        for i, is_gap in enumerate(gaps):
            if is_gap and not in_gap:
                gap_start = i
                in_gap = True
            elif not is_gap and in_gap:
                gap_width = i - gap_start
                if gap_width > 20:  # Significant gap
                    gap_regions.append((gap_start, i))
                in_gap = False
        
        # Multiple significant gaps suggest columns
        return len(gap_regions) >= 1
    
    def _select_optimal_engine(
        self,
        characteristics: DocumentCharacteristics,
        gpu_available: Optional[bool]
    ) -> str:
        """Select optimal OCR engine based on characteristics."""
        # For handwritten text, EasyOCR is generally better
        if characteristics.is_handwritten:
            return "easyocr"
        
        # For very poor quality, EasyOCR with GPU can help
        if characteristics.quality == DocumentQuality.VERY_POOR and gpu_available:
            return "easyocr"
        
        # For large documents without GPU, use tesseract-cli for efficiency
        if characteristics.page_count > 50 and not gpu_available:
            return "tesseract-cli"
        
        # Default to tesserocr for good balance
        # Note: In production, check if tesserocr is available
        return "easyocr"  # Using easyocr as default since tesserocr requires compilation