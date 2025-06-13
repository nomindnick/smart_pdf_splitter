"""OCR preprocessing module for image enhancement."""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class OCRPreprocessor:
    """Preprocess images to improve OCR accuracy."""
    
    def __init__(
        self,
        target_dpi: int = 300,
        enable_gpu: Optional[bool] = None,
        quality_threshold: float = 0.7
    ):
        """
        Initialize preprocessor with configuration.
        
        Args:
            target_dpi: Target DPI for upscaling
            enable_gpu: Enable GPU acceleration (auto-detect if None)
            quality_threshold: Minimum quality score to skip preprocessing
        """
        self.target_dpi = target_dpi
        self.quality_threshold = quality_threshold
        
        # Auto-detect GPU if not specified
        if enable_gpu is None:
            try:
                # Check if CUDA is available via OpenCV
                test_mat = cv2.cuda_GpuMat()
                self.enable_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
                logger.info(f"GPU auto-detected: {self.enable_gpu}")
            except:
                self.enable_gpu = False
        else:
            self.enable_gpu = enable_gpu
    
    def preprocess_image(
        self,
        image: np.ndarray,
        current_dpi: int = 150,
        auto_enhance: bool = True,
        preprocessing_steps: Optional[list] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply preprocessing pipeline to improve OCR quality.
        
        Args:
            image: Input image as numpy array
            current_dpi: Current image DPI
            auto_enhance: Automatically apply all enhancements
            preprocessing_steps: Specific steps to apply (overrides auto_enhance)
            
        Returns:
            Tuple of (processed_image, processing_info)
        """
        info = {
            "steps_applied": [],
            "quality_score_before": 0.0,
            "quality_score_after": 0.0,
            "skew_angle": 0.0,
            "processing_time_ms": 0
        }
        
        import time
        start_time = time.time()
        
        # Assess initial quality
        info["quality_score_before"] = self._assess_quality(image)
        
        # Skip preprocessing if quality is already good
        if info["quality_score_before"] >= self.quality_threshold and not preprocessing_steps:
            info["steps_applied"].append("skipped_high_quality")
            info["quality_score_after"] = info["quality_score_before"]
            return image, info
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            info["steps_applied"].append("grayscale")
        
        # Determine preprocessing steps
        if preprocessing_steps:
            steps = preprocessing_steps
        elif auto_enhance:
            steps = ["upscale", "denoise", "deskew", "contrast", "threshold", "borders"]
        else:
            steps = ["deskew", "contrast"]  # Minimal preprocessing
        
        # Apply preprocessing steps
        if "upscale" in steps and current_dpi < self.target_dpi:
            scale = self.target_dpi / current_dpi
            image = self._upscale_image(image, scale)
            info["steps_applied"].append(f"upscale_{scale:.1f}x")
        
        if "denoise" in steps and self._needs_denoising(image):
            image = self._denoise_image(image)
            info["steps_applied"].append("denoise")
        
        if "deskew" in steps:
            angle = self._detect_skew(image)
            info["skew_angle"] = angle
            if abs(angle) > 0.5:  # More than 0.5 degrees
                image = self._deskew_image(image, angle)
                info["steps_applied"].append(f"deskew_{angle:.1f}°")
        
        if "contrast" in steps and self._needs_contrast_enhancement(image):
            image = self._enhance_contrast(image)
            info["steps_applied"].append("contrast_enhance")
        
        if "threshold" in steps:
            image = self._adaptive_threshold(image)
            info["steps_applied"].append("binarize")
        
        if "borders" in steps:
            image = self._remove_borders(image)
            info["steps_applied"].append("remove_borders")
        
        # Assess final quality
        info["quality_score_after"] = self._assess_quality(image)
        info["processing_time_ms"] = int((time.time() - start_time) * 1000)
        
        return image, info
    
    def preprocess_pdf_page(
        self,
        pdf_page_image: np.ndarray,
        page_number: int,
        document_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess a PDF page image with document context.
        
        Args:
            pdf_page_image: Page image as numpy array
            page_number: Page number in document
            document_info: Optional document metadata
            
        Returns:
            Tuple of (processed_image, processing_info)
        """
        # Determine DPI from document info or estimate
        current_dpi = 150  # Default
        if document_info and "dpi" in document_info:
            current_dpi = document_info["dpi"]
        
        # Apply preprocessing
        processed_image, info = self.preprocess_image(
            pdf_page_image,
            current_dpi=current_dpi,
            auto_enhance=True
        )
        
        info["page_number"] = page_number
        
        return processed_image, info
    
    def _upscale_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Upscale image using appropriate interpolation."""
        if scale <= 1.0:
            return image
            
        height, width = image.shape[:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        # Use INTER_CUBIC for moderate upscaling, INTER_LANCZOS4 for large upscaling
        interpolation = cv2.INTER_CUBIC if scale < 2.0 else cv2.INTER_LANCZOS4
        
        return cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise using non-local means denoising."""
        # Adjust parameters based on noise level
        noise_level = self._estimate_noise_level(image)
        
        if noise_level < 10:
            return image  # Low noise, skip denoising
        
        h = min(noise_level, 30)  # Denoising strength
        template_window_size = 7
        search_window_size = 21
        
        return cv2.fastNlMeansDenoising(
            image, None, h=h,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )
    
    def _detect_skew(self, image: np.ndarray) -> float:
        """Detect skew angle using Hough transform."""
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return 0.0
        
        # Calculate angles
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta * 180 / np.pi) - 90
            if -45 <= angle <= 45:  # Reasonable skew range
                angles.append(angle)
        
        if not angles:
            return 0.0
        
        # Use median to avoid outliers
        return float(np.median(angles))
    
    def _deskew_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image to correct skew."""
        if abs(angle) < 0.1:
            return image
            
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image bounds
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Adjust rotation matrix for translation
        M[0, 2] += (new_width / 2) - center[0]
        M[1, 2] += (new_height / 2) - center[1]
        
        # Rotate image with white background
        rotated = cv2.warpAffine(
            image, M, (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255  # White background
        )
        
        return rotated
    
    def _needs_denoising(self, image: np.ndarray) -> bool:
        """Check if image has significant noise."""
        noise_level = self._estimate_noise_level(image)
        return noise_level > 15
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level using Laplacian variance."""
        # Apply Laplacian operator
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        
        # Calculate standard deviation
        sigma = np.std(laplacian)
        
        return float(sigma)
    
    def _needs_contrast_enhancement(self, image: np.ndarray) -> bool:
        """Check if image has poor contrast."""
        # Calculate histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        
        # Find the range of intensities that contain 95% of pixels
        cumsum = np.cumsum(hist)
        total = cumsum[-1]
        
        lower_percentile = np.searchsorted(cumsum, 0.025 * total)
        upper_percentile = np.searchsorted(cumsum, 0.975 * total)
        
        # If the range is too narrow, contrast is poor
        intensity_range = upper_percentile - lower_percentile
        
        return intensity_range < 100
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE."""
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Apply CLAHE
        enhanced = clahe.apply(image)
        
        return enhanced
    
    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for better binarization."""
        # First apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
        
        return binary
    
    def _remove_borders(self, image: np.ndarray) -> np.ndarray:
        """Remove black borders from scanned images."""
        # Create binary image
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return image
        
        # Find the largest contour (likely the page)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add small margin
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        # Crop image
        cropped = image[y:y+h, x:x+w]
        
        return cropped
    
    def _assess_quality(self, image: np.ndarray) -> float:
        """
        Assess image quality for OCR (0-1 scale).
        
        Factors:
        - Sharpness (Laplacian variance)
        - Contrast (histogram spread)
        - Noise level
        - Resolution
        """
        scores = []
        
        # Sharpness score
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        sharpness = laplacian.var()
        sharpness_score = min(sharpness / 500, 1.0)
        scores.append(sharpness_score)
        
        # Contrast score
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        non_zero_bins = np.count_nonzero(hist)
        contrast_score = min(non_zero_bins / 200, 1.0)
        scores.append(contrast_score)
        
        # Noise score (inverse of noise level)
        noise_level = self._estimate_noise_level(image)
        noise_score = max(0, 1.0 - (noise_level / 100))
        scores.append(noise_score)
        
        # Resolution score
        height, width = image.shape[:2]
        pixel_count = height * width
        resolution_score = min(pixel_count / (2000 * 2000), 1.0)  # Normalize to 2MP
        scores.append(resolution_score)
        
        # Overall quality is the average of all scores
        quality = sum(scores) / len(scores)
        
        return round(quality, 3)