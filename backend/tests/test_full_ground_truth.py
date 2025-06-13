#!/usr/bin/env python3
"""Comprehensive test against ground truth data."""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from src.core.hybrid_boundary_detector import HybridBoundaryDetector, VisualProcessingConfig
from src.core.models import Boundary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    
    # Convert detected boundaries to comparable format
    detected_docs = []
    for i, boundary in enumerate(detected):
        detected_docs.append({
            'start': boundary.start_page,
            'end': boundary.end_page,
            'type': boundary.document_type,
            'confidence': boundary.confidence
        })
    
    # Metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    partial_matches = 0
    exact_matches = 0
    
    # Track which ground truth docs were matched
    matched_gt = set()
    
    # Check each detected boundary
    for det in detected_docs:
        best_match = None
        best_overlap = 0
        
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
                
            # Calculate overlap
            overlap_start = max(det['start'], gt.start_page)
            overlap_end = min(det['end'], gt.end_page)
            
            if overlap_start <= overlap_end:
                overlap = overlap_end - overlap_start + 1
                total_pages = max(det['end'], gt.end_page) - min(det['start'], gt.start_page) + 1
                overlap_ratio = overlap / total_pages
                
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_match = i
        
        if best_match is not None and best_overlap > 0.5:
            matched_gt.add(best_match)
            true_positives += 1
            
            # Check if exact match
            gt = ground_truth[best_match]
            if det['start'] == gt.start_page and det['end'] == gt.end_page:
                exact_matches += 1
            else:
                partial_matches += 1
        else:
            false_positives += 1
    
    # Check for missed documents
    false_negatives = len(ground_truth) - len(matched_gt)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'total_ground_truth': len(ground_truth),
        'total_detected': len(detected_docs),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'exact_matches': exact_matches,
        'partial_matches': partial_matches,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'matched_details': []
    }


def run_full_test():
    """Run full test against ground truth."""
    
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
    logger.info("FULL PDF SPLITTER TEST - 36 pages, 14 documents")
    logger.info("="*80)
    
    # Test configurations
    test_configs = [
        # Basic text-only detection
        {
            'name': 'Text-Only Detection',
            'config': VisualProcessingConfig(
                enable_visual_features=False,
                enable_intelligent_ocr=False,
                enable_llm=False
            )
        },
        # Text + Visual
        {
            'name': 'Text + Visual Detection',
            'config': VisualProcessingConfig(
                enable_visual_features=True,
                enable_intelligent_ocr=False,
                enable_llm=False
            )
        },
        # Intelligent OCR (no LLM)
        {
            'name': 'Intelligent OCR (No LLM)',
            'config': VisualProcessingConfig(
                enable_visual_features=True,
                enable_intelligent_ocr=True,
                enable_llm=False
            )
        },
        # Full detection with LLM
        {
            'name': 'Full Detection (with LLM)',
            'config': VisualProcessingConfig(
                enable_visual_features=True,
                enable_intelligent_ocr=True,
                enable_llm=True,
                llm_confidence_threshold=0.7
            )
        }
    ]
    
    results = []
    
    for test_case in test_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {test_case['name']}")
        logger.info(f"{'='*60}")
        
        detector = HybridBoundaryDetector(config=test_case['config'])
        
        # Time the detection
        start_time = time.time()
        
        try:
            boundaries = detector.detect_boundaries(test_pdf)
            processing_time = time.time() - start_time
            
            # Evaluate accuracy
            accuracy = evaluate_accuracy(boundaries, ground_truth)
            
            # Log results
            logger.info(f"\n✓ Completed in {processing_time:.2f} seconds ({processing_time/60:.1f} minutes)")
            logger.info(f"  Time per page: {processing_time/36:.2f}s")
            
            logger.info(f"\nAccuracy Metrics:")
            logger.info(f"  Documents detected: {accuracy['total_detected']} / {accuracy['total_ground_truth']}")
            logger.info(f"  Exact matches: {accuracy['exact_matches']}")
            logger.info(f"  Partial matches: {accuracy['partial_matches']}")
            logger.info(f"  False positives: {accuracy['false_positives']}")
            logger.info(f"  Missed documents: {accuracy['false_negatives']}")
            logger.info(f"  Precision: {accuracy['precision']:.2%}")
            logger.info(f"  Recall: {accuracy['recall']:.2%}")
            logger.info(f"  F1 Score: {accuracy['f1_score']:.2%}")
            
            # Show detected boundaries
            logger.info(f"\nDetected Boundaries:")
            for i, boundary in enumerate(boundaries[:10]):  # Show first 10
                logger.info(f"  {i+1}. Pages {boundary.start_page}-{boundary.end_page} "
                          f"({boundary.document_type}, conf: {boundary.confidence:.2%})")
            
            if len(boundaries) > 10:
                logger.info(f"  ... and {len(boundaries) - 10} more")
            
            results.append({
                'name': test_case['name'],
                'processing_time': processing_time,
                'accuracy': accuracy,
                'boundaries': boundaries
            })
            
        except Exception as e:
            logger.error(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    logger.info("\n" + "="*80)
    logger.info("SUMMARY COMPARISON")
    logger.info("="*80)
    
    logger.info(f"\n{'Method':<25} {'Time':<10} {'F1 Score':<10} {'Exact':<10} {'Partial':<10} {'Missed':<10}")
    logger.info("-" * 85)
    
    for result in results:
        if 'accuracy' in result:
            acc = result['accuracy']
            logger.info(f"{result['name']:<25} "
                      f"{result['processing_time']:.1f}s{'':<5} "
                      f"{acc['f1_score']:.2%}{'':<5} "
                      f"{acc['exact_matches']:<10} "
                      f"{acc['partial_matches']:<10} "
                      f"{acc['false_negatives']:<10}")
    
    # Save detailed results
    output_file = Path('full_test_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'test_file': str(test_pdf),
            'ground_truth_documents': len(ground_truth),
            'results': [
                {
                    'name': r['name'],
                    'processing_time_seconds': r['processing_time'],
                    'accuracy_metrics': r['accuracy'],
                    'boundaries_count': len(r.get('boundaries', []))
                }
                for r in results if 'accuracy' in r
            ]
        }, f, indent=2)
    
    logger.info(f"\nDetailed results saved to: {output_file}")
    
    # Show ground truth for reference
    logger.info("\n" + "="*80)
    logger.info("GROUND TRUTH DOCUMENTS")
    logger.info("="*80)
    
    for i, gt in enumerate(ground_truth):
        logger.info(f"{i+1}. Pages {gt.start_page}-{gt.end_page}: {gt.doc_type}")


if __name__ == "__main__":
    logger.info("Starting comprehensive ground truth test...")
    logger.info("This may take 10-15 minutes to complete. Please be patient.")
    run_full_test()