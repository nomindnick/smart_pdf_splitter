#!/usr/bin/env python3
"""Test with ALL capabilities enabled: OCR, Visual, and LLM detection."""

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
    match_details = []
    
    for i, gt in enumerate(ground_truth):
        found_match = False
        
        for j, det in enumerate(detected):
            if det.start_page == gt.start_page and det.end_page == gt.end_page:
                exact_matches += 1
                found_match = True
                matched_gt.add(i)
                match_details.append({
                    'ground_truth': f"Pages {gt.start_page}-{gt.end_page} ({gt.doc_type})",
                    'detected': f"Pages {det.start_page}-{det.end_page} (conf: {det.confidence:.2%})",
                    'match_type': 'exact'
                })
                break
            elif (det.start_page <= gt.end_page and det.end_page >= gt.start_page):
                partial_matches += 1
                found_match = True
                matched_gt.add(i)
                overlap_start = max(det.start_page, gt.start_page)
                overlap_end = min(det.end_page, gt.end_page)
                match_details.append({
                    'ground_truth': f"Pages {gt.start_page}-{gt.end_page} ({gt.doc_type})",
                    'detected': f"Pages {det.start_page}-{det.end_page} (overlap: {overlap_start}-{overlap_end})",
                    'match_type': 'partial'
                })
                break
        
        if not found_match:
            missed += 1
            match_details.append({
                'ground_truth': f"Pages {gt.start_page}-{gt.end_page} ({gt.doc_type})",
                'detected': 'MISSED',
                'match_type': 'missed'
            })
    
    total = len(ground_truth)
    accuracy = (exact_matches / total * 100) if total > 0 else 0
    
    # Calculate false positives
    false_positives = len(detected) - len(matched_gt)
    
    # Calculate precision and recall
    true_positives = exact_matches + partial_matches
    precision = true_positives / len(detected) if len(detected) > 0 else 0
    recall = true_positives / total if total > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'total_ground_truth': total,
        'total_detected': len(detected),
        'exact_matches': exact_matches,
        'partial_matches': partial_matches,
        'missed': missed,
        'false_positives': false_positives,
        'accuracy_percent': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'match_details': match_details
    }


def run_full_capabilities_test():
    """Run test with ALL capabilities enabled."""
    
    test_pdf = Path("../tests/test_files/Test_PDF_Set_1.pdf")
    ground_truth_file = Path("../tests/test_files/Test_PDF_Set_Ground_Truth.json")
    
    if not test_pdf.exists() or not ground_truth_file.exists():
        logger.error("Test files not found!")
        return
    
    ground_truth = parse_ground_truth(ground_truth_file)
    logger.info(f"Loaded ground truth: {len(ground_truth)} documents")
    
    logger.info("="*80)
    logger.info("FULL CAPABILITIES TEST - OCR + Visual + LLM")
    logger.info("="*80)
    
    # Enable ALL capabilities
    config = VisualProcessingConfig(
        enable_visual_features=True,
        enable_intelligent_ocr=True,
        enable_llm=True,
        llm_confidence_threshold=0.7,
        visual_confidence_threshold=0.5,
        enable_picture_classification=True
    )
    
    detector = HybridBoundaryDetector(config=config)
    
    logger.info("Configuration:")
    logger.info("  - Visual Features: ENABLED")
    logger.info("  - Intelligent OCR: ENABLED")
    logger.info("  - LLM Detection: ENABLED")
    logger.info("  - Picture Classification: ENABLED")
    logger.info("")
    logger.info("Processing first 10 pages for demonstration...")
    logger.info("This may take 10-15 minutes with LLM enabled...")
    
    start_time = time.time()
    
    try:
        # Process first 10 pages
        boundaries = detector.detect_boundaries(test_pdf, page_range=(1, 10))
        processing_time = time.time() - start_time
        
        logger.info(f"\n✓ Completed in {processing_time:.2f} seconds ({processing_time/60:.1f} minutes)")
        logger.info(f"  Time per page: {processing_time/10:.2f}s")
        logger.info(f"  Detected {len(boundaries)} boundaries")
        
        # Filter ground truth to first 10 pages
        gt_first_10 = [gt for gt in ground_truth if gt.start_page <= 10]
        
        accuracy = evaluate_accuracy(boundaries, gt_first_10)
        
        logger.info(f"\n📊 Results (First 10 Pages):")
        logger.info(f"  Ground truth documents: {accuracy['total_ground_truth']}")
        logger.info(f"  Detected boundaries: {accuracy['total_detected']}")
        logger.info(f"  ✅ Exact matches: {accuracy['exact_matches']}")
        logger.info(f"  🔶 Partial matches: {accuracy['partial_matches']}")
        logger.info(f"  ❌ Missed: {accuracy['missed']}")
        logger.info(f"  ⚠️  False positives: {accuracy['false_positives']}")
        logger.info(f"  📈 Accuracy: {accuracy['accuracy_percent']:.1f}%")
        logger.info(f"  📊 Precision: {accuracy['precision']:.2%}")
        logger.info(f"  📊 Recall: {accuracy['recall']:.2%}")
        logger.info(f"  📊 F1 Score: {accuracy['f1_score']:.2%}")
        
        logger.info(f"\n🔍 Detected Boundaries:")
        for i, boundary in enumerate(boundaries):
            signals = ", ".join([s.type.value for s in boundary.signals[:3]])
            logger.info(f"  {i+1}. Pages {boundary.start_page}-{boundary.end_page} "
                      f"(type: {boundary.document_type}, conf: {boundary.confidence:.2%}, "
                      f"signals: {signals})")
        
        logger.info(f"\n📋 Match Details:")
        for detail in accuracy['match_details']:
            emoji = "✅" if detail['match_type'] == 'exact' else "🔶" if detail['match_type'] == 'partial' else "❌"
            logger.info(f"  {emoji} {detail['ground_truth']} → {detail['detected']}")
        
        logger.info(f"\n📄 Ground Truth (First 10 Pages):")
        for i, gt in enumerate(gt_first_10):
            logger.info(f"  {i+1}. Pages {gt.start_page}-{gt.end_page} ({gt.doc_type})")
        
        # Save detailed results
        results = {
            'test_file': str(test_pdf),
            'processing_time_seconds': processing_time,
            'pages_processed': 10,
            'time_per_page': processing_time / 10,
            'capabilities': {
                'visual_features': True,
                'intelligent_ocr': True,
                'llm_detection': True,
                'picture_classification': True
            },
            'accuracy_metrics': accuracy,
            'boundaries': [
                {
                    'start_page': b.start_page,
                    'end_page': b.end_page,
                    'confidence': b.confidence,
                    'document_type': b.document_type,
                    'signals': [s.type.value for s in b.signals]
                }
                for b in boundaries
            ]
        }
        
        with open('full_capabilities_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n💾 Detailed results saved to: full_capabilities_results.json")
        
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_full_capabilities_test()