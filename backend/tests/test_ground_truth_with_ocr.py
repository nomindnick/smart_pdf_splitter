#!/usr/bin/env python3
"""Test with OCR enabled for better accuracy."""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from src.core.hybrid_boundary_detector import HybridBoundaryDetector, VisualProcessingConfig
from src.core.models import Boundary
from src.core.pipeline_config import PipelineProfiles

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


def evaluate_accuracy(detected: List[Boundary], ground_truth: List[GroundTruthDocument]) -> Dict[str, Any]:
    """Evaluate accuracy of detected boundaries against ground truth."""
    
    exact_matches = 0
    partial_matches = 0
    missed = 0
    
    matched_gt = set()
    
    for i, gt in enumerate(ground_truth):
        found_match = False
        
        for det in detected:
            if det.start_page == gt.start_page and det.end_page == gt.end_page:
                exact_matches += 1
                found_match = True
                matched_gt.add(i)
                break
            elif (det.start_page <= gt.end_page and det.end_page >= gt.start_page):
                partial_matches += 1
                found_match = True
                matched_gt.add(i)
                break
        
        if not found_match:
            missed += 1
    
    total = len(ground_truth)
    accuracy = (exact_matches / total * 100) if total > 0 else 0
    
    # Calculate false positives
    false_positives = len(detected) - len(matched_gt)
    
    return {
        'total_ground_truth': total,
        'total_detected': len(detected),
        'exact_matches': exact_matches,
        'partial_matches': partial_matches,
        'missed': missed,
        'false_positives': false_positives,
        'accuracy_percent': accuracy
    }


def run_ocr_test():
    """Run test with OCR enabled."""
    
    test_pdf = Path("../tests/test_files/Test_PDF_Set_1.pdf")
    ground_truth_file = Path("../tests/test_files/Test_PDF_Set_Ground_Truth.json")
    
    if not test_pdf.exists() or not ground_truth_file.exists():
        logger.error("Test files not found!")
        return
    
    ground_truth = parse_ground_truth(ground_truth_file)
    logger.info(f"Loaded ground truth: {len(ground_truth)} documents")
    
    logger.info("="*80)
    logger.info("GROUND TRUTH TEST WITH OCR")
    logger.info("="*80)
    
    # Test with intelligent OCR but no LLM (for speed)
    config = VisualProcessingConfig(
        enable_visual_features=True,
        enable_intelligent_ocr=True,
        enable_llm=False
    )
    
    # Create detector
    detector = HybridBoundaryDetector(config=config)
    
    logger.info("Running boundary detection with intelligent OCR...")
    logger.info("This may take 5-10 minutes for 36 pages...")
    start_time = time.time()
    
    try:
        # Process first 10 pages for faster testing
        boundaries = detector.detect_boundaries(test_pdf, page_range=(1, 10))
        processing_time = time.time() - start_time
        
        logger.info(f"\n✓ Completed in {processing_time:.2f} seconds ({processing_time/60:.1f} minutes)")
        logger.info(f"  Time per page: {processing_time/10:.2f}s")
        logger.info(f"  Detected {len(boundaries)} boundaries")
        
        # Filter ground truth to first 10 pages
        gt_first_10 = [gt for gt in ground_truth if gt.start_page <= 10]
        
        accuracy = evaluate_accuracy(boundaries, gt_first_10)
        
        logger.info(f"\nResults (First 10 Pages):")
        logger.info(f"  Ground truth documents: {accuracy['total_ground_truth']}")
        logger.info(f"  Detected boundaries: {accuracy['total_detected']}")
        logger.info(f"  Exact matches: {accuracy['exact_matches']}")
        logger.info(f"  Partial matches: {accuracy['partial_matches']}")
        logger.info(f"  Missed: {accuracy['missed']}")
        logger.info(f"  False positives: {accuracy['false_positives']}")
        logger.info(f"  Accuracy: {accuracy['accuracy_percent']:.1f}%")
        
        logger.info(f"\nDetected Boundaries:")
        for i, boundary in enumerate(boundaries):
            logger.info(f"  {i+1}. Pages {boundary.start_page}-{boundary.end_page} "
                      f"(type: {boundary.document_type}, confidence: {boundary.confidence:.2%})")
        
        logger.info(f"\nGround Truth (First 10 Pages):")
        for i, gt in enumerate(gt_first_10):
            logger.info(f"  {i+1}. Pages {gt.start_page}-{gt.end_page} ({gt.doc_type})")
        
        # Show improvement
        logger.info(f"\nImprovement over text-only:")
        logger.info(f"  Boundaries reduced from 10 to {len(boundaries)} "
                   f"({(10-len(boundaries))/10*100:.0f}% reduction)")
        
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_ocr_test()