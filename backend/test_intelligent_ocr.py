#!/usr/bin/env python3
"""Test the intelligent OCR strategy integration."""

import logging
from pathlib import Path
import json

from src.core.hybrid_boundary_detector import HybridBoundaryDetector, VisualProcessingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_intelligent_ocr_detection():
    """Test boundary detection with intelligent OCR strategy."""
    
    # Test file path
    test_file = Path("../tests/test_files/Test_PDF_Set_1.pdf")
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    logger.info("=" * 70)
    logger.info("Testing Intelligent OCR Strategy")
    logger.info("=" * 70)
    
    # Configure detector with intelligent OCR enabled
    config = VisualProcessingConfig(
        enable_visual_features=False,  # Test text-only first
        enable_intelligent_ocr=True,
        enable_llm=False  # For now
    )
    
    detector = HybridBoundaryDetector(config=config)
    
    # Detect boundaries
    logger.info("\nDetecting boundaries with intelligent OCR...")
    boundaries = detector.detect_boundaries(test_file)
    
    logger.info(f"\nDetected {len(boundaries)} documents")
    
    # Display results
    for i, boundary in enumerate(boundaries):
        logger.info(f"\nDocument {i+1}:")
        logger.info(f"  Pages: {boundary.start_page}-{boundary.end_page}")
        logger.info(f"  Confidence: {boundary.confidence:.2f}")
        logger.info(f"  Type: {boundary.document_type}")
        logger.info(f"  Detection method: {boundary.metadata.get('detection_method', 'unknown')}")
        
        # Show signals
        if boundary.signals:
            logger.info("  Signals:")
            for signal in boundary.signals[:3]:  # Show first 3 signals
                logger.info(f"    - {signal.type.value}: {signal.description[:60]}...")
    
    # Get detection summary
    summary = detector.get_detection_summary(boundaries)
    logger.info("\n" + "="*70)
    logger.info("Detection Summary:")
    logger.info(json.dumps(summary, indent=2))
    
    # Compare with ground truth
    ground_truth_file = Path("../tests/test_files/Test_PDF_Set_Ground_Truth.json")
    if ground_truth_file.exists():
        with open(ground_truth_file) as f:
            ground_truth = json.load(f)
        
        logger.info("\n" + "="*70)
        logger.info("Comparison with Ground Truth:")
        logger.info(f"Expected documents: {len(ground_truth['documents'])}")
        logger.info(f"Detected documents: {len(boundaries)}")
        
        # Check accuracy
        correct = 0
        for expected in ground_truth['documents']:
            # Find matching boundary
            for boundary in boundaries:
                if (boundary.start_page == expected['start_page'] and 
                    boundary.end_page == expected['end_page']):
                    correct += 1
                    break
        
        accuracy = correct / len(ground_truth['documents']) * 100
        logger.info(f"Boundary accuracy: {accuracy:.1f}% ({correct}/{len(ground_truth['documents'])})")


def test_with_visual_features():
    """Test with visual features enabled."""
    
    test_file = Path("../tests/test_files/Test_PDF_Set_1.pdf")
    if not test_file.exists():
        return
    
    logger.info("\n" + "="*70)
    logger.info("Testing with Visual Features + Intelligent OCR")
    logger.info("=" * 70)
    
    # Configure with visual features
    config = VisualProcessingConfig(
        enable_visual_features=True,
        enable_intelligent_ocr=True,
        enable_llm=False
    )
    
    detector = HybridBoundaryDetector(config=config)
    
    # Detect boundaries
    boundaries = detector.detect_boundaries(test_file)
    
    logger.info(f"\nDetected {len(boundaries)} documents with hybrid approach")
    
    # Show detection methods used
    summary = detector.get_detection_summary(boundaries)
    logger.info("\nDetection methods:")
    for method, count in summary['detection_methods'].items():
        logger.info(f"  {method}: {count}")


if __name__ == "__main__":
    test_intelligent_ocr_detection()
    test_with_visual_features()