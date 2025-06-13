"""Tests for OCR improvement modules."""

import pytest
import numpy as np
import cv2
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.ocr_preprocessor import OCRPreprocessor
from src.core.ocr_confidence_scorer import OCRConfidenceScorer
from src.core.ocr_postprocessor import OCRPostProcessor
from src.core.ocr_config import (
    OCRConfig, AdaptiveOCRConfigurator, DocumentCharacteristics, DocumentQuality
)


class TestOCRPreprocessor:
    """Test OCR preprocessing functionality."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = OCRPreprocessor(target_dpi=300)
        assert preprocessor.target_dpi == 300
        assert preprocessor.quality_threshold == 0.7
    
    def test_grayscale_conversion(self):
        """Test grayscale conversion."""
        # Create color image
        color_img = np.zeros((100, 100, 3), dtype=np.uint8)
        color_img[:, :] = [100, 150, 200]  # Blue-ish color
        
        preprocessor = OCRPreprocessor()
        processed, info = preprocessor.preprocess_image(color_img)
        
        assert len(processed.shape) == 2  # Grayscale
        assert "grayscale" in info["steps_applied"]
    
    def test_deskew_detection(self):
        """Test skew detection and correction."""
        # Create skewed image with text
        img = np.ones((500, 500), dtype=np.uint8) * 255
        cv2.putText(img, "Test Text", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)
        
        # Rotate by 5 degrees
        center = (250, 250)
        M = cv2.getRotationMatrix2D(center, 5, 1.0)
        skewed = cv2.warpAffine(img, M, (500, 500), borderValue=255)
        
        preprocessor = OCRPreprocessor()
        processed, info = preprocessor.preprocess_image(
            skewed, preprocessing_steps=["deskew"]
        )
        
        # Should detect and correct skew
        assert abs(info["skew_angle"]) > 3  # Should detect ~5 degree skew
        assert "deskew" in info["steps_applied"]
    
    def test_noise_removal(self):
        """Test noise removal."""
        # Create noisy image
        img = np.ones((200, 200), dtype=np.uint8) * 255
        noise = np.random.randint(0, 50, (200, 200), dtype=np.uint8)
        noisy_img = cv2.add(img, noise)
        
        preprocessor = OCRPreprocessor()
        processed, info = preprocessor.preprocess_image(
            noisy_img, preprocessing_steps=["denoise"]
        )
        
        # Check that noise was reduced
        assert np.std(processed) < np.std(noisy_img)
        assert "denoise" in info["steps_applied"]
    
    def test_quality_assessment(self):
        """Test image quality assessment."""
        # Good quality image
        good_img = np.ones((300, 300), dtype=np.uint8) * 255
        cv2.rectangle(good_img, (50, 50), (250, 250), 0, 2)
        
        preprocessor = OCRPreprocessor()
        _, info = preprocessor.preprocess_image(good_img, auto_enhance=False)
        
        assert info["quality_score_before"] > 0.5
        assert info["quality_score_after"] > 0.5
    
    def test_upscaling(self):
        """Test image upscaling."""
        # Small image
        small_img = np.ones((100, 100), dtype=np.uint8) * 255
        
        preprocessor = OCRPreprocessor(target_dpi=300)
        processed, info = preprocessor.preprocess_image(
            small_img, current_dpi=100, preprocessing_steps=["upscale"]
        )
        
        # Should be upscaled 3x
        assert processed.shape[0] == 300
        assert processed.shape[1] == 300
        assert "upscale_3.0x" in info["steps_applied"]


class TestOCRConfidenceScorer:
    """Test OCR confidence scoring."""
    
    def test_scorer_initialization(self):
        """Test scorer initialization."""
        scorer = OCRConfidenceScorer(language="en")
        assert scorer.language == "en"
        assert len(scorer.common_words) > 0
    
    def test_good_quality_text(self):
        """Test scoring of good quality text."""
        good_text = """This is a well-formatted document with proper sentences.
        It contains common English words and follows standard grammar rules.
        The text is clear and easy to read without any OCR errors."""
        
        scorer = OCRConfidenceScorer()
        result = scorer.score_ocr_output(good_text)
        
        assert result["overall_confidence"] > 0.7
        assert result["quality_assessment"] in ["good", "excellent"]
        assert len(result["issues"]) < 2
    
    def test_poor_quality_text(self):
        """Test scoring of poor quality text."""
        poor_text = """Th1s ls a p00rly f0rmatted d0cument w1th many err0rs|||
        It c0nta1ns rnany 0CR err0rs and str@nge ch@r@cters|||
        The text 1s d1ff1cult t0 read and underst@nd|||"""
        
        scorer = OCRConfidenceScorer()
        result = scorer.score_ocr_output(poor_text)
        
        assert result["overall_confidence"] < 0.5
        assert result["quality_assessment"] in ["poor", "very_poor"]
        assert len(result["issues"]) > 2
    
    def test_email_detection(self):
        """Test email-specific confidence scoring."""
        email_text = """From: john.doe@example.com
To: jane.smith@example.com
Subject: Meeting Tomorrow

Dear Jane,

Please confirm our meeting tomorrow at 2 PM.

Best regards,
John"""
        
        scorer = OCRConfidenceScorer()
        result = scorer.score_ocr_output(email_text, document_type="email")
        
        assert "email_confidence" in result["scores"]
        assert result["scores"]["email_confidence"] > 0.8
    
    def test_language_detection(self):
        """Test language confidence scoring."""
        english_text = "The quick brown fox jumps over the lazy dog."
        non_english_text = "Das ist ein deutscher Text mit deutschen Wörtern."
        
        scorer = OCRConfidenceScorer()
        
        eng_result = scorer.score_ocr_output(english_text, expected_language="en")
        assert eng_result["scores"]["language_confidence"] > 0.7
        
        non_eng_result = scorer.score_ocr_output(non_english_text, expected_language="en")
        assert non_eng_result["scores"]["language_confidence"] < 0.3
    
    def test_issue_identification(self):
        """Test identification of specific issues."""
        text_with_issues = """This text has rnany OCR artifacts|||
        There are also verylongwordswithoutspaces and
        
        
        
        excessive whitespace."""
        
        scorer = OCRConfidenceScorer()
        result = scorer.score_ocr_output(text_with_issues)
        
        assert "ocr_artifacts_detected" in result["issues"]
        assert "word_boundary_errors" in result["issues"]
        assert "excessive_line_breaks" in result["issues"]


class TestOCRPostProcessor:
    """Test OCR post-processing."""
    
    def test_postprocessor_initialization(self):
        """Test postprocessor initialization."""
        postprocessor = OCRPostProcessor(language="en")
        assert postprocessor.language == "en"
        assert len(postprocessor.ocr_corrections) > 0
    
    def test_basic_corrections(self):
        """Test basic OCR corrections."""
        text_with_errors = "The cloor is rnade of woocl. Tlne window is c1ean."
        
        postprocessor = OCRPostProcessor()
        processed, info = postprocessor.process_text(text_with_errors)
        
        assert info["corrections_made"] > 0
        assert "ocr_corrections" in info["steps_applied"]
        # Should fix some common errors
        assert "rnade" not in processed  # Should be "made"
    
    def test_spacing_corrections(self):
        """Test spacing and formatting corrections."""
        text_with_spacing_issues = "This is a sentence.There is no space after period.Also,no space after comma."
        
        postprocessor = OCRPostProcessor()
        processed, info = postprocessor.process_text(text_with_spacing_issues)
        
        assert ". There" in processed  # Space added after period
        assert ", " in processed  # Space added after comma
        assert "formatting" in info["steps_applied"]
    
    def test_aggressive_corrections(self):
        """Test aggressive correction mode."""
        poor_text = "th e quick br own fox jumps ov er the lazy dog"
        
        postprocessor = OCRPostProcessor()
        processed, info = postprocessor.process_text(
            poor_text, apply_aggressive=True
        )
        
        assert "the" in processed  # "th e" -> "the"
        assert "brown" in processed  # "br own" -> "brown"
        assert "over" in processed  # "ov er" -> "over"
        assert "aggressive_corrections" in info["steps_applied"]
    
    def test_domain_specific_corrections(self):
        """Test domain-specific corrections."""
        email_text = "From: john @ grnail . com"
        
        postprocessor = OCRPostProcessor()
        processed, info = postprocessor.process_text(
            email_text, document_type="email"
        )
        
        assert "@gmail.com" in processed  # Fixed email domain
        assert "email_corrections" in info["steps_applied"]
    
    def test_correction_suggestions(self):
        """Test getting correction suggestions without applying."""
        text = "This has sorne OCR errors in tl1e text."
        
        postprocessor = OCRPostProcessor()
        suggestions = postprocessor.get_correction_suggestions(text)
        
        assert len(suggestions) > 0
        assert any(s["type"] == "ocr_error" for s in suggestions)


class TestAdaptiveOCRConfiguration:
    """Test adaptive OCR configuration."""
    
    def test_configurator_initialization(self):
        """Test configurator initialization."""
        configurator = AdaptiveOCRConfigurator()
        assert len(configurator.quality_profiles) == 5
        assert DocumentQuality.EXCELLENT in configurator.quality_profiles
    
    def test_document_analysis(self):
        """Test document characteristic analysis."""
        # Create sample pages
        good_page = np.ones((1000, 800), dtype=np.uint8) * 255
        cv2.putText(good_page, "Clear Text", (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)
        
        poor_page = np.ones((1000, 800), dtype=np.uint8) * 200
        noise = np.random.randint(0, 100, (1000, 800), dtype=np.uint8)
        poor_page = cv2.add(poor_page, noise)
        
        configurator = AdaptiveOCRConfigurator()
        
        # Analyze good document
        good_chars = configurator.analyze_document([good_page], 1)
        assert good_chars.quality in [DocumentQuality.GOOD, DocumentQuality.EXCELLENT]
        
        # Analyze poor document
        poor_chars = configurator.analyze_document([poor_page], 1)
        assert poor_chars.quality in [DocumentQuality.POOR, DocumentQuality.FAIR]
    
    def test_config_generation(self):
        """Test OCR config generation based on characteristics."""
        configurator = AdaptiveOCRConfigurator()
        
        # Good quality document
        good_chars = DocumentCharacteristics(
            quality=DocumentQuality.GOOD,
            page_count=10,
            estimated_dpi=200
        )
        good_config = configurator.generate_config(good_chars)
        
        assert good_config.target_dpi == 200
        assert len(good_config.preprocessing_steps) < 3
        assert not good_config.apply_aggressive_corrections
        
        # Poor quality document
        poor_chars = DocumentCharacteristics(
            quality=DocumentQuality.POOR,
            page_count=10,
            estimated_dpi=150,
            noise_level=50,
            skew_angle=5
        )
        poor_config = configurator.generate_config(poor_chars)
        
        assert poor_config.target_dpi == 300
        assert "denoise" in poor_config.preprocessing_steps
        assert "deskew" in poor_config.preprocessing_steps
        assert poor_config.apply_aggressive_corrections
    
    def test_large_document_optimization(self):
        """Test optimization for large documents."""
        configurator = AdaptiveOCRConfigurator()
        
        large_doc_chars = DocumentCharacteristics(
            quality=DocumentQuality.FAIR,
            page_count=200
        )
        
        config = configurator.generate_config(large_doc_chars)
        
        assert config.page_batch_size <= 2  # Small batches for large docs
        assert not config.enable_multi_engine_voting  # Disabled for performance


class TestOCRConfig:
    """Test OCR configuration model."""
    
    def test_default_config(self):
        """Test default OCR configuration."""
        config = OCRConfig()
        
        assert config.enable_ocr is True
        assert config.ocr_engine == "auto"
        assert config.ocr_languages == ["en"]
        assert config.enable_preprocessing is True
        assert config.target_dpi == 300
        assert config.confidence_threshold == 0.7
    
    def test_custom_config(self):
        """Test custom OCR configuration."""
        config = OCRConfig(
            enable_ocr=False,
            ocr_engine="tesseract-cli",
            target_dpi=600,
            preprocessing_steps=["deskew", "denoise"],
            apply_aggressive_corrections=True
        )
        
        assert config.enable_ocr is False
        assert config.ocr_engine == "tesseract-cli"
        assert config.target_dpi == 600
        assert config.preprocessing_steps == ["deskew", "denoise"]
        assert config.apply_aggressive_corrections is True
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = OCRConfig(
            ocr_languages=["en", "es"],
            preprocessing_steps=["upscale", "denoise", "deskew"]
        )
        
        # Serialize to dict
        config_dict = config.dict()
        assert config_dict["ocr_languages"] == ["en", "es"]
        assert "upscale" in config_dict["preprocessing_steps"]
        
        # Create from dict
        new_config = OCRConfig(**config_dict)
        assert new_config.ocr_languages == config.ocr_languages
        assert new_config.preprocessing_steps == config.preprocessing_steps