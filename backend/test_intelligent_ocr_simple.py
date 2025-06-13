#!/usr/bin/env python3
"""Simple test of intelligent OCR strategy."""

import logging
from pathlib import Path

from src.core.intelligent_ocr_strategy import IntelligentOCRStrategy
from src.core.pipeline_config import PipelineProfiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_ocr_strategy_planning():
    """Test the OCR strategy planning."""
    
    # Test file path
    test_file = Path("../tests/test_files/Test_PDF_Set_1.pdf")
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    logger.info("="*70)
    logger.info("Testing OCR Strategy Planning")
    logger.info("="*70)
    
    # Test different detection methods
    test_cases = [
        (["text"], "Text-only detection"),
        (["text", "visual"], "Text + Visual detection"),
        (["text", "visual", "llm"], "Full detection with LLM")
    ]
    
    for methods, description in test_cases:
        logger.info(f"\n{description}:")
        logger.info("-" * 50)
        
        strategy = IntelligentOCRStrategy(methods)
        plan = strategy.plan_ocr_strategy(test_file)
        
        logger.info(f"Total pages: {plan['total_pages']}")
        logger.info(f"Quality summary:")
        logger.info(f"  - High quality: {plan['quality_summary']['high_quality_pages']}")
        logger.info(f"  - Medium quality: {plan['quality_summary']['medium_quality_pages']}")
        logger.info(f"  - Fast: {plan['quality_summary']['fast_pages']}")
        logger.info(f"  - Skipped: {plan['quality_summary']['skipped_pages']}")
        logger.info(f"Estimated time: {plan['estimated_time']:.1f} seconds")
        
        # Show processing order (first 10 pages)
        logger.info(f"Processing order (first 10): {plan['processing_order'][:10]}")
        
        # Show page strategies for first few pages
        logger.info("Page strategies (first 5 pages):")
        for i in range(min(5, plan['total_pages'])):
            strategy_info = plan['page_strategies'][i]
            logger.info(f"  Page {i+1}: {strategy_info['strategy']} - {strategy_info['reason']}")


def test_config_profiles():
    """Test different configuration profiles."""
    
    logger.info("\n" + "="*70)
    logger.info("Configuration Profiles")
    logger.info("="*70)
    
    profiles = [
        ("Splitter Detection", PipelineProfiles.get_splitter_detection_config()),
        ("LLM Detection", PipelineProfiles.get_llm_detection_config()),
        ("RAG Extraction", PipelineProfiles.get_rag_extraction_config())
    ]
    
    for name, config in profiles:
        logger.info(f"\n{name}:")
        logger.info(f"  Engine: {config.ocr_engine}")
        logger.info(f"  Target DPI: {config.target_dpi}")
        logger.info(f"  Full page OCR: {config.force_full_page_ocr}")
        logger.info(f"  Preprocessing: {config.enable_preprocessing}")
        if config.enable_preprocessing:
            logger.info(f"  Steps: {', '.join(config.preprocessing_steps)}")
        logger.info(f"  Post-processing: {config.enable_postprocessing}")
        logger.info(f"  Max time per page: {config.max_processing_time}s")


if __name__ == "__main__":
    test_ocr_strategy_planning()
    test_config_profiles()