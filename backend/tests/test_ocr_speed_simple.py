#!/usr/bin/env python3
"""Simple OCR speed test - process just a few pages."""

import logging
import time
from pathlib import Path

from src.core.unified_document_processor import UnifiedDocumentProcessor, ProcessingMode
from src.core.pipeline_config import PipelineProfiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_single_page_speed():
    """Test OCR speed on single pages with different configs."""
    
    test_file = Path("../tests/test_files/Test_PDF_Set_1.pdf")
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    logger.info("="*70)
    logger.info("Single Page OCR Speed Test")
    logger.info("="*70)
    
    # Extract first page as test
    import fitz
    doc = fitz.open(str(test_file))
    
    # Save first 3 pages as separate PDFs
    test_pages = []
    for i in range(min(3, len(doc))):
        page_doc = fitz.open()
        page_doc.insert_pdf(doc, from_page=i, to_page=i)
        page_path = Path(f"test_page_{i+1}.pdf")
        page_doc.save(str(page_path))
        page_doc.close()
        test_pages.append(page_path)
    
    doc.close()
    
    # Test configurations
    configs = [
        ("Fast/Splitter", PipelineProfiles.get_splitter_detection_config()),
        ("LLM Quality", PipelineProfiles.get_llm_detection_config()),
    ]
    
    for config_name, config in configs:
        logger.info(f"\n{config_name} Configuration:")
        logger.info(f"  Engine: {config.ocr_engine}")
        logger.info(f"  DPI: {config.target_dpi}")
        logger.info(f"  Preprocessing: {config.preprocessing_steps if config.enable_preprocessing else 'None'}")
        logger.info("\n  Page timings:")
        
        processor = UnifiedDocumentProcessor(
            mode=ProcessingMode.ENHANCED,
            ocr_config=config,
            enable_adaptive=False  # Use fixed config
        )
        
        total_time = 0
        for page_path in test_pages:
            start = time.time()
            
            try:
                doc = processor.process_document(page_path)
                elapsed = time.time() - start
                total_time += elapsed
                
                words = doc.page_info[0].word_count if doc.page_info else 0
                confidence = doc.page_info[0].ocr_confidence if doc.page_info and doc.page_info[0].ocr_confidence else 0
                
                logger.info(f"    {page_path.name}: {elapsed:.2f}s ({words} words, {confidence:.0%} conf)")
                
            except Exception as e:
                logger.error(f"    {page_path.name}: Error - {e}")
        
        avg_time = total_time / len(test_pages) if test_pages else 0
        logger.info(f"  Average: {avg_time:.2f}s per page")
    
    # Cleanup
    for page_path in test_pages:
        page_path.unlink()


def test_intelligent_strategy_overhead():
    """Test overhead of intelligent OCR strategy."""
    
    test_file = Path("../tests/test_files/Test_PDF_Set_1.pdf")
    if not test_file.exists():
        return
    
    logger.info("\n" + "="*70)
    logger.info("Intelligent Strategy Overhead Test")
    logger.info("="*70)
    
    from src.core.intelligent_ocr_strategy import IntelligentOCRStrategy
    
    # Test strategy planning time
    strategies = [
        (["text"], "Text only"),
        (["text", "visual", "llm"], "Full LLM")
    ]
    
    for methods, desc in strategies:
        strategy = IntelligentOCRStrategy(methods)
        
        start = time.time()
        plan = strategy.plan_ocr_strategy(test_file)
        planning_time = time.time() - start
        
        logger.info(f"\n{desc}:")
        logger.info(f"  Planning time: {planning_time:.3f}s")
        logger.info(f"  Pages to process:")
        logger.info(f"    High quality: {plan['quality_summary']['high_quality_pages']}")
        logger.info(f"    Medium: {plan['quality_summary']['medium_quality_pages']}")
        logger.info(f"    Fast: {plan['quality_summary']['fast_pages']}")
        logger.info(f"  Estimated total time: {plan['estimated_time']:.1f}s")


def compare_boundary_detection_speed():
    """Compare boundary detection speed with different strategies."""
    
    test_file = Path("../tests/test_files/Test_PDF_Set_1.pdf")
    if not test_file.exists():
        return
    
    logger.info("\n" + "="*70)
    logger.info("Boundary Detection Speed Comparison")
    logger.info("="*70)
    
    from src.core.hybrid_boundary_detector import HybridBoundaryDetector, VisualProcessingConfig
    
    # Test first 5 pages only for speed
    page_range = (1, 5)
    
    configs = [
        (
            VisualProcessingConfig(
                enable_visual_features=False,
                enable_intelligent_ocr=False
            ),
            "Standard Text Detection"
        ),
        (
            VisualProcessingConfig(
                enable_visual_features=False,
                enable_intelligent_ocr=True,
                enable_llm=False
            ),
            "Intelligent OCR (no LLM)"
        ),
        (
            VisualProcessingConfig(
                enable_visual_features=False,
                enable_intelligent_ocr=True,
                enable_llm=True
            ),
            "Intelligent OCR (with LLM)"
        ),
    ]
    
    for config, desc in configs:
        detector = HybridBoundaryDetector(config=config)
        
        logger.info(f"\n{desc}:")
        start = time.time()
        
        try:
            boundaries = detector.detect_boundaries(test_file, page_range=page_range)
            elapsed = time.time() - start
            
            logger.info(f"  Time: {elapsed:.2f}s")
            logger.info(f"  Boundaries found: {len(boundaries)}")
            logger.info(f"  Time per page: {elapsed / (page_range[1] - page_range[0] + 1):.2f}s")
            
        except Exception as e:
            logger.error(f"  Error: {e}")


if __name__ == "__main__":
    test_single_page_speed()
    test_intelligent_strategy_overhead()
    compare_boundary_detection_speed()