#!/usr/bin/env python3
"""Simple comprehensive test against ground truth data."""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from src.core.hybrid_boundary_detector import HybridBoundaryDetector, VisualProcessingConfig
from src.core.models import Boundary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthDocument:
    """Ground truth document info."""
    start_page: int
    end_page: int
    doc_type: str
    summary: str


def parse_ground_truth(ground_truth_path: Path) -> List[GroundTruthDocument]:
    """Parse ground truth JSON into document boundaries."""
    with open(ground_truth_path, 'r') as f:
        data = json.load(f)
    
    documents = []
    for doc in data['documents']:
        # Parse page range (e.g., "1-4" or "13")
        pages = doc['pages']
        if '-' in pages:
            start, end = pages.split('-')
            start_page = int(start)
            end_page = int(end)
        else:
            start_page = end_page = int(pages)
        
        documents.append(GroundTruthDocument(
            start_page=start_page,
            end_page=end_page,
            doc_type=doc['type'],
            summary=doc['summary']
        ))
    
    return documents


def evaluate_accuracy(
    detected: List[Boundary], 
    ground_truth: List[GroundTruthDocument]
) -> Dict[str, Any]:
    """Evaluate accuracy of detected boundaries against ground truth."""
    
    # Metrics
    exact_matches = 0
    partial_matches = 0
    missed = 0
    
    # Match each ground truth doc
    matched_gt = set()
    
    for gt in ground_truth:
        found_match = False
        
        for det in detected:
            # Check if boundaries match
            if det.start_page == gt.start_page and det.end_page == gt.end_page:
                exact_matches += 1
                found_match = True
                break
            # Check partial match (overlapping)
            elif (det.start_page <= gt.end_page and det.end_page >= gt.start_page):
                partial_matches += 1
                found_match = True
                break
        
        if not found_match:
            missed += 1
    
    # Calculate metrics
    total = len(ground_truth)
    accuracy = (exact_matches / total * 100) if total > 0 else 0
    
    return {
        'total_ground_truth': total,
        'total_detected': len(detected),
        'exact_matches': exact_matches,
        'partial_matches': partial_matches,
        'missed': missed,
        'accuracy_percent': accuracy
    }


def run_simple_test():
    """Run simple test against ground truth."""
    
    # File paths
    test_pdf = Path("../tests/test_files/Test_PDF_Set_1.pdf")
    ground_truth_file = Path("../tests/test_files/Test_PDF_Set_Ground_Truth.json")
    
    if not test_pdf.exists() or not ground_truth_file.exists():
        logger.error("Test files not found!")
        return
    
    # Load ground truth
    ground_truth = parse_ground_truth(ground_truth_file)
    logger.info(f"Loaded ground truth: {len(ground_truth)} documents")
    
    logger.info("="*80)
    logger.info("SIMPLE GROUND TRUTH TEST")
    logger.info("="*80)
    
    # Simple text-only detection
    config = VisualProcessingConfig(
        enable_visual_features=False,
        enable_intelligent_ocr=False,
        enable_llm=False
    )
    
    detector = HybridBoundaryDetector(config=config)
    
    logger.info("Running text-only boundary detection...")
    start_time = time.time()
    
    try:
        boundaries = detector.detect_boundaries(test_pdf)
        processing_time = time.time() - start_time
        
        logger.info(f"✓ Completed in {processing_time:.2f} seconds")
        logger.info(f"  Detected {len(boundaries)} boundaries")
        
        # Evaluate accuracy
        accuracy = evaluate_accuracy(boundaries, ground_truth)
        
        logger.info(f"\nResults:")
        logger.info(f"  Ground truth documents: {accuracy['total_ground_truth']}")
        logger.info(f"  Detected boundaries: {accuracy['total_detected']}")
        logger.info(f"  Exact matches: {accuracy['exact_matches']}")
        logger.info(f"  Partial matches: {accuracy['partial_matches']}")
        logger.info(f"  Missed: {accuracy['missed']}")
        logger.info(f"  Accuracy: {accuracy['accuracy_percent']:.1f}%")
        
        # Show detected boundaries
        logger.info(f"\nDetected Boundaries:")
        for i, boundary in enumerate(boundaries):
            logger.info(f"  {i+1}. Pages {boundary.start_page}-{boundary.end_page} "
                      f"(type: {boundary.document_type}, confidence: {boundary.confidence:.2%})")
        
        # Show ground truth for comparison
        logger.info(f"\nGround Truth:")
        for i, gt in enumerate(ground_truth):
            logger.info(f"  {i+1}. Pages {gt.start_page}-{gt.end_page} ({gt.doc_type})")
        
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_simple_test()