#!/usr/bin/env python3
"""
Test script to verify boundary detection against ground truth data.
"""

import json
import sys
from pathlib import Path

# Add backend/src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.document_processor import DocumentProcessor
from src.core.boundary_detector import BoundaryDetector


def load_ground_truth():
    """Load ground truth data."""
    ground_truth_path = Path(__file__).parent.parent / "tests" / "test_files" / "Test_PDF_Set_Ground_Truth.json"
    with open(ground_truth_path, 'r') as f:
        return json.load(f)


def parse_page_range(page_range_str):
    """Parse page range string like '1-4' or '13' into start, end tuple."""
    if '-' in page_range_str:
        start, end = map(int, page_range_str.split('-'))
        return start, end
    else:
        page = int(page_range_str)
        return page, page


def main():
    """Run boundary detection and compare with ground truth."""
    print("Loading ground truth data...")
    ground_truth = load_ground_truth()
    
    # Get expected boundaries from ground truth
    expected_boundaries = []
    for doc in ground_truth['documents']:
        start, end = parse_page_range(doc['pages'])
        expected_boundaries.append({
            'start': start,
            'end': end,
            'type': doc['type'],
            'summary': doc['summary'][:50] + '...' if len(doc['summary']) > 50 else doc['summary']
        })
    
    print(f"\nExpected {len(expected_boundaries)} documents:")
    for i, boundary in enumerate(expected_boundaries, 1):
        print(f"  Document {i}: Pages {boundary['start']}-{boundary['end']} ({boundary['type']})")
    
    # Process the test PDF
    pdf_path = Path(__file__).parent.parent / "tests" / "test_files" / "Test_PDF_Set_1.pdf"
    
    print(f"\nProcessing PDF: {pdf_path}")
    print("This may take a moment...")
    
    # Initialize document processor with simple configuration
    processor = DocumentProcessor(
        enable_ocr=False,
        page_batch_size=4,
        max_memory_mb=4096
    )
    
    # Process document
    try:
        pages = list(processor.process_document(str(pdf_path)))
        print(f"Extracted {len(pages)} pages")
    except Exception as e:
        print(f"Error processing document: {e}")
        return
    
    # Initialize boundary detector
    detector = BoundaryDetector(
        min_confidence=0.6,
        min_signals=1,
        enable_visual_analysis=True
    )
    
    # Detect boundaries
    print("\nDetecting boundaries...")
    boundaries = detector.detect_boundaries(pages)
    
    print(f"\nDetected {len(boundaries)} documents:")
    for i, boundary in enumerate(boundaries, 1):
        doc_type = boundary.document_type.value if boundary.document_type else "Unknown"
        print(f"  Document {i}: Pages {boundary.start_page}-{boundary.end_page} "
              f"(confidence: {boundary.confidence:.2f}, type: {doc_type})")
        
        # Show signals
        for signal in boundary.signals[:3]:  # Show first 3 signals
            print(f"    - {signal.type.value}: {signal.description} (conf: {signal.confidence:.2f})")
    
    # Compare with ground truth
    print("\n" + "="*60)
    print("COMPARISON WITH GROUND TRUTH")
    print("="*60)
    
    # Check document count
    print(f"\nDocument count:")
    print(f"  Expected: {len(expected_boundaries)}")
    print(f"  Detected: {len(boundaries)}")
    print(f"  Difference: {len(boundaries) - len(expected_boundaries)}")
    
    # Check boundary alignment
    print("\nBoundary alignment:")
    
    # Find matches
    matches = 0
    near_matches = 0
    
    for expected in expected_boundaries:
        # Check for exact match
        exact_match = any(
            b.start_page == expected['start'] and b.end_page == expected['end']
            for b in boundaries
        )
        
        # Check for near match (within 1 page)
        near_match = any(
            abs(b.start_page - expected['start']) <= 1 and 
            abs(b.end_page - expected['end']) <= 1
            for b in boundaries
        )
        
        if exact_match:
            matches += 1
            status = "✓ EXACT"
        elif near_match:
            near_matches += 1
            status = "~ NEAR"
        else:
            status = "✗ MISSED"
        
        print(f"  Pages {expected['start']}-{expected['end']} ({expected['type']}): {status}")
    
    # Calculate accuracy
    accuracy = matches / len(expected_boundaries) * 100
    near_accuracy = (matches + near_matches) / len(expected_boundaries) * 100
    
    print(f"\nAccuracy:")
    print(f"  Exact matches: {matches}/{len(expected_boundaries)} ({accuracy:.1f}%)")
    print(f"  Near matches: {near_matches}/{len(expected_boundaries)}")
    print(f"  Total accuracy: {matches + near_matches}/{len(expected_boundaries)} ({near_accuracy:.1f}%)")
    
    # Show false positives
    print("\nFalse positives (detected but not in ground truth):")
    false_positives = 0
    for boundary in boundaries:
        is_expected = any(
            abs(boundary.start_page - exp['start']) <= 1
            for exp in expected_boundaries
        )
        if not is_expected:
            false_positives += 1
            doc_type = boundary.document_type.value if boundary.document_type else "Unknown"
            print(f"  Pages {boundary.start_page}-{boundary.end_page} ({doc_type})")
    
    if false_positives == 0:
        print("  None")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Ground truth documents: {len(expected_boundaries)}")
    print(f"Detected documents: {len(boundaries)}")
    print(f"Exact accuracy: {accuracy:.1f}%")
    print(f"Near accuracy: {near_accuracy:.1f}%")
    print(f"False positives: {false_positives}")


if __name__ == "__main__":
    main()