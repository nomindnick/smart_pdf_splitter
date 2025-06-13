#!/usr/bin/env python3
"""Test OCR pipeline performance - quality and speed evaluation."""

import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List

from src.core.unified_document_processor import UnifiedDocumentProcessor, ProcessingMode
from src.core.ocr_config import OCRConfig
from src.core.pipeline_config import PipelineProfiles
from src.core.hybrid_boundary_detector import HybridBoundaryDetector, VisualProcessingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def measure_processing_time(processor: UnifiedDocumentProcessor, test_file: Path, mode_name: str) -> Dict[str, Any]:
    """Measure processing time and gather stats."""
    start_time = time.time()
    
    try:
        document = processor.process_document(test_file, return_quality_report=True)
        processing_time = time.time() - start_time
        
        # Get processing stats
        stats = processor.get_processing_stats()
        
        # Calculate quality metrics
        total_words = sum(page.word_count for page in document.page_info)
        avg_confidence = stats.get('average_confidence', 0)
        
        # Get page quality reports if available
        quality_reports = []
        if document.metadata and 'page_quality_reports' in document.metadata.custom_fields:
            quality_reports = document.metadata.custom_fields['page_quality_reports']
        
        return {
            'mode': mode_name,
            'success': True,
            'processing_time': processing_time,
            'pages_processed': document.total_pages,
            'time_per_page': processing_time / document.total_pages if document.total_pages > 0 else 0,
            'total_words': total_words,
            'avg_confidence': avg_confidence,
            'ocr_performed': stats.get('ocr_performed', 0),
            'pages_preprocessed': stats.get('pages_preprocessed', 0),
            'total_corrections': stats.get('total_corrections', 0),
            'quality_reports': quality_reports,
            'stats': stats
        }
    except Exception as e:
        logger.error(f"Error processing with {mode_name}: {e}")
        return {
            'mode': mode_name,
            'success': False,
            'error': str(e),
            'processing_time': time.time() - start_time
        }


def test_processing_modes():
    """Test different processing modes and compare performance."""
    test_file = Path("../tests/test_files/Test_PDF_Set_1.pdf")
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    logger.info("="*70)
    logger.info("OCR Pipeline Performance Test")
    logger.info("="*70)
    
    # Test configurations
    test_configs = [
        # (mode, config, description)
        (ProcessingMode.BASIC, None, "Basic (Docling only)"),
        (ProcessingMode.ENHANCED, PipelineProfiles.get_splitter_detection_config(), "Enhanced (Fast/Splitter)"),
        (ProcessingMode.ENHANCED, PipelineProfiles.get_llm_detection_config(), "Enhanced (LLM Quality)"),
        (ProcessingMode.SMART, PipelineProfiles.get_splitter_detection_config(), "Smart (Adaptive Fast)"),
        (ProcessingMode.SMART, PipelineProfiles.get_llm_detection_config(), "Smart (Adaptive LLM)"),
    ]
    
    results = []
    
    for mode, config, description in test_configs:
        logger.info(f"\nTesting: {description}")
        logger.info("-" * 50)
        
        # Create processor
        processor = UnifiedDocumentProcessor(
            mode=mode,
            ocr_config=config,
            enable_adaptive=True,
            max_ocr_pages=10  # Limit for testing
        )
        
        # Measure performance
        result = measure_processing_time(processor, test_file, description)
        results.append(result)
        
        if result['success']:
            logger.info(f"✓ Completed in {result['processing_time']:.2f}s")
            logger.info(f"  - Pages: {result['pages_processed']}")
            logger.info(f"  - Time per page: {result['time_per_page']:.2f}s")
            logger.info(f"  - Total words: {result['total_words']}")
            logger.info(f"  - OCR confidence: {result['avg_confidence']:.2%}")
            logger.info(f"  - Pages with OCR: {result['ocr_performed']}")
            logger.info(f"  - Corrections made: {result['total_corrections']}")
        else:
            logger.error(f"✗ Failed: {result['error']}")
    
    # Compare results
    logger.info("\n" + "="*70)
    logger.info("Performance Comparison")
    logger.info("="*70)
    
    # Find baseline (basic mode)
    baseline = next((r for r in results if r['mode'] == "Basic (Docling only)" and r['success']), None)
    
    if baseline:
        logger.info(f"\nBaseline: {baseline['mode']}")
        logger.info(f"Time: {baseline['processing_time']:.2f}s")
        
        logger.info("\nRelative Performance:")
        for result in results:
            if result['success'] and result != baseline:
                time_ratio = result['processing_time'] / baseline['processing_time']
                quality_diff = result['avg_confidence'] - baseline.get('avg_confidence', 0)
                
                logger.info(f"\n{result['mode']}:")
                logger.info(f"  Time: {result['processing_time']:.2f}s ({time_ratio:.1f}x baseline)")
                logger.info(f"  Quality: {result['avg_confidence']:.2%} ({'+' if quality_diff >= 0 else ''}{quality_diff:.2%} vs baseline)")
                logger.info(f"  Words extracted: {result['total_words']} ({result['total_words'] - baseline.get('total_words', 0):+d})")
    
    # Save detailed results
    with open('ocr_pipeline_performance_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nDetailed results saved to: ocr_pipeline_performance_results.json")


def test_intelligent_ocr_performance():
    """Test intelligent OCR strategy performance."""
    test_file = Path("../tests/test_files/Test_PDF_Set_1.pdf")
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    logger.info("\n" + "="*70)
    logger.info("Intelligent OCR Strategy Performance")
    logger.info("="*70)
    
    # Test with and without intelligent OCR
    configs = [
        (False, "Standard processing"),
        (True, "Intelligent OCR enabled")
    ]
    
    for enable_intelligent, description in configs:
        logger.info(f"\n{description}:")
        logger.info("-" * 50)
        
        config = VisualProcessingConfig(
            enable_visual_features=False,
            enable_intelligent_ocr=enable_intelligent,
            enable_llm=True  # Enable LLM to trigger intelligent strategy
        )
        
        detector = HybridBoundaryDetector(config=config)
        
        start_time = time.time()
        boundaries = detector.detect_boundaries(test_file, page_range=(1, 10))  # First 10 pages
        processing_time = time.time() - start_time
        
        logger.info(f"✓ Completed in {processing_time:.2f}s")
        logger.info(f"  - Boundaries detected: {len(boundaries)}")
        
        # Show boundary details
        for i, boundary in enumerate(boundaries[:3]):  # First 3 boundaries
            logger.info(f"\n  Boundary {i+1}:")
            logger.info(f"    Pages: {boundary.start_page}-{boundary.end_page}")
            logger.info(f"    Confidence: {boundary.confidence:.2%}")
            logger.info(f"    Type: {boundary.document_type}")


def test_page_quality_analysis():
    """Analyze quality metrics for different pages."""
    test_file = Path("../tests/test_files/Test_PDF_Set_1.pdf")
    if not test_file.exists():
        return
    
    logger.info("\n" + "="*70)
    logger.info("Page Quality Analysis")
    logger.info("="*70)
    
    # Process with quality analysis
    processor = UnifiedDocumentProcessor(
        mode=ProcessingMode.ENHANCED,
        ocr_config=PipelineProfiles.get_llm_detection_config(),
        enable_adaptive=True,
        max_ocr_pages=5  # Analyze first 5 pages in detail
    )
    
    document = processor.process_document(test_file, return_quality_report=True)
    
    # Analyze page quality
    if document.metadata and 'page_quality_reports' in document.metadata.custom_fields:
        quality_reports = document.metadata.custom_fields['page_quality_reports']
        
        logger.info(f"\nAnalyzed {len(quality_reports)} pages:")
        
        for report in quality_reports[:5]:  # Show first 5
            page_num = report.get('page_number', 0)
            confidence = report.get('confidence', {})
            
            logger.info(f"\nPage {page_num}:")
            
            if 'confidence' in report:
                overall = confidence.get('overall_confidence', 0)
                logger.info(f"  Overall confidence: {overall:.2%}")
                logger.info(f"  Quality: {confidence.get('quality_assessment', 'unknown')}")
                
                if 'issues' in confidence:
                    issues = confidence['issues']
                    if issues:
                        logger.info(f"  Issues: {', '.join(issues[:3])}")
                
                if 'postprocessing' in report:
                    corrections = report['postprocessing'].get('corrections_made', 0)
                    if corrections > 0:
                        logger.info(f"  Corrections applied: {corrections}")


if __name__ == "__main__":
    test_processing_modes()
    test_intelligent_ocr_performance()
    test_page_quality_analysis()