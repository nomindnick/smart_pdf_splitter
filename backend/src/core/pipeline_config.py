"""Configuration profiles for different processing scenarios."""

from typing import Dict, Any
from .ocr_config import OCRConfig


class PipelineProfiles:
    """
    Pre-configured profiles for different use cases.
    
    This makes it easy to switch between configurations for
    standalone splitter vs RAG pipeline.
    """
    
    @staticmethod
    def get_llm_detection_config() -> OCRConfig:
        """
        OCR config optimized for LLM-based boundary detection.
        
        LLMs need high-quality, complete text to understand context.
        """
        return OCRConfig(
            # Engine settings
            enable_ocr=True,
            ocr_engine="easyocr",        # Better for understanding context
            ocr_languages=["en"],
            
            # Quality settings for LLM comprehension
            target_dpi=250,              # Good balance of quality/speed
            force_full_page_ocr=True,    # LLMs need all text
            bitmap_area_threshold=0.05,  # Don't miss small text
            
            # Preprocessing for clarity
            enable_preprocessing=True,
            preprocessing_steps=[
                "deskew",
                "denoise",
                "contrast"
            ],
            
            # Post-processing to fix OCR errors that confuse LLMs
            enable_postprocessing=True,
            apply_aggressive_corrections=True,
            confidence_threshold=0.6,
            
            # Performance settings
            max_processing_time=30,      # 30 seconds per page
            page_batch_size=2,           # Small batches for context
            memory_limit_mb=4096
        )
    
    @staticmethod
    def get_splitter_detection_config() -> OCRConfig:
        """
        Fast OCR config for boundary detection in standalone splitter.
        
        Optimized for speed over quality.
        """
        return OCRConfig(
            # Engine settings
            enable_ocr=True,
            ocr_engine="tesseract-cli",  # Faster than EasyOCR
            ocr_languages=["eng"],       # Single language for speed
            
            # Quality/speed tradeoffs
            target_dpi=150,              # Lower DPI for speed
            force_full_page_ocr=False,   # Only OCR text regions
            bitmap_area_threshold=0.2,   # Skip small images
            
            # Minimal preprocessing
            enable_preprocessing=True,
            preprocessing_steps=["deskew"],  # Only essential
            
            # Skip quality enhancements
            enable_postprocessing=False,
            confidence_threshold=0.5,    # Lower threshold OK
            
            # Performance settings
            max_processing_time=10,      # 10 seconds per page max
            page_batch_size=4,
            memory_limit_mb=2048        # 2GB limit
        )
    
    @staticmethod
    def get_rag_extraction_config() -> OCRConfig:
        """
        High-quality OCR config for RAG text extraction.
        
        Optimized for quality and completeness.
        """
        return OCRConfig(
            # Engine settings
            enable_ocr=True,
            ocr_engine="easyocr",        # Better quality
            ocr_languages=["en"],        # Can add more languages
            
            # Quality settings
            target_dpi=300,              # Higher DPI for quality
            force_full_page_ocr=True,    # Process everything
            bitmap_area_threshold=0.05,  # Process smaller images too
            
            # Full preprocessing
            enable_preprocessing=True,
            preprocessing_steps=[
                "deskew",
                "denoise", 
                "contrast",
                "threshold"
            ],
            
            # Quality enhancements
            enable_postprocessing=True,
            apply_aggressive_corrections=True,
            confidence_threshold=0.7,
            
            # Performance settings (can be more generous)
            max_processing_time=60,      # 1 minute per page
            page_batch_size=8,
            memory_limit_mb=8192        # 8GB limit
        )
    
    @staticmethod
    def get_test_config() -> OCRConfig:
        """
        Test configuration for development.
        
        Balanced settings for testing.
        """
        return OCRConfig(
            enable_ocr=True,
            ocr_engine="tesseract-cli",
            ocr_languages=["eng"],
            target_dpi=200,
            preprocessing_steps=["deskew"],
            enable_postprocessing=True,
            max_processing_time=30
        )


class ProcessingStrategies:
    """
    Processing strategies for different scenarios.
    """
    
    @staticmethod
    def get_splitter_strategy() -> Dict[str, Any]:
        """Strategy for standalone PDF splitter."""
        return {
            "purpose": "boundary_detection",
            "max_ocr_pages": 10,         # Sample up to 10 pages
            "parallel_processing": True,  # Use parallel for speed
            "parallel_workers": 4,        # Moderate parallelism
            "cache_results": True,        # Cache for potential reuse
            "quality_target": "fast",     # Speed over quality
            "page_selection": {
                "strategy": "smart",      # Smart page selection
                "include_first_last": True,
                "sample_rate": 0.2,       # Sample 20% of pages
                "max_consecutive_skip": 5  # Don't skip more than 5 pages
            }
        }
    
    @staticmethod
    def get_rag_strategy() -> Dict[str, Any]:
        """Strategy for RAG pipeline processing."""
        return {
            "purpose": "full_extraction",
            "max_ocr_pages": None,        # Process all pages
            "parallel_processing": True,  # Maximum parallelism
            "parallel_workers": -1,       # Use all CPU cores
            "cache_results": True,        # Cache everything
            "quality_target": "best",     # Quality over speed
            "page_selection": {
                "strategy": "all",        # Process everything
                "include_first_last": True,
                "sample_rate": 1.0,       # 100% of pages
                "max_consecutive_skip": 0  # Never skip
            }
        }


# Environment-specific configurations
class EnvironmentConfig:
    """Configuration based on deployment environment."""
    
    DEVELOPMENT = {
        "max_workers": 2,
        "enable_profiling": True,
        "log_level": "DEBUG",
        "cache_enabled": True,
        "cache_ttl": 3600  # 1 hour
    }
    
    PRODUCTION_SPLITTER = {
        "max_workers": 4,
        "enable_profiling": False,
        "log_level": "INFO",
        "cache_enabled": True,
        "cache_ttl": 86400  # 24 hours
    }
    
    PRODUCTION_RAG = {
        "max_workers": -1,  # All cores
        "enable_profiling": False,
        "log_level": "INFO",
        "cache_enabled": True,
        "cache_ttl": 604800  # 7 days
    }