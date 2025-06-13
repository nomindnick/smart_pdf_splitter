"""
Comprehensive integration test that validates boundary detection against ground truth data.

This test processes the full Test_PDF_Set_1.pdf and compares detected boundaries
against the ground truth JSON to ensure we correctly identify all 14 documents.
"""

import pytest
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

from src.core.document_processor import DocumentProcessor
from src.core.simple_document_processor import SimpleDocumentProcessor
from src.core.boundary_detector import BoundaryDetector
from src.core.models import Boundary, PageInfo


class TestGroundTruthValidation:
    """Test suite that validates detection accuracy against ground truth data."""
    
    @pytest.fixture
    def test_pdf_path(self):
        """Path to test PDF file."""
        return Path("/home/nick/Projects/smart_pdf_splitter/tests/test_files/Test_PDF_Set_1.pdf")
    
    @pytest.fixture
    def ground_truth_path(self):
        """Path to ground truth JSON file."""
        return Path("/home/nick/Projects/smart_pdf_splitter/tests/test_files/Test_PDF_Set_Ground_Truth.json")
    
    @pytest.fixture
    def ground_truth_data(self, ground_truth_path):
        """Load and return ground truth data."""
        if not ground_truth_path.exists():
            pytest.skip(f"Ground truth file not found at {ground_truth_path}")
        
        with open(ground_truth_path, 'r') as f:
            return json.load(f)
    
    @pytest.fixture
    def expected_boundaries(self, ground_truth_data) -> List[Tuple[int, int]]:
        """Extract expected boundaries from ground truth data."""
        boundaries = []
        for doc in ground_truth_data['documents']:
            pages = doc['pages']
            if '-' in pages:
                start, end = map(int, pages.split('-'))
                boundaries.append((start, end))
            else:
                page = int(pages)
                boundaries.append((page, page))
        return boundaries
    
    def test_ground_truth_structure(self, ground_truth_data):
        """Test that ground truth data has expected structure."""
        assert 'documents' in ground_truth_data
        assert len(ground_truth_data['documents']) == 14
        
        # Verify each document has required fields
        for doc in ground_truth_data['documents']:
            assert 'pages' in doc
            assert 'type' in doc
            assert 'summary' in doc
    
    def test_process_full_pdf(self, test_pdf_path):
        """Test that we can process the entire test PDF."""
        if not test_pdf_path.exists():
            pytest.skip(f"Test PDF not found at {test_pdf_path}")
        
        # Process document (PDF needs OCR as it's scanned)
        processor = DocumentProcessor(enable_ocr=True, page_batch_size=5)
        pages = list(processor.process_document(test_pdf_path))
        
        # Should have 36 pages based on ground truth
        assert len(pages) == 36, f"Expected 36 pages, got {len(pages)}"
        
        # Verify basic page properties
        for i, page in enumerate(pages):
            assert page.page_number == i + 1
            assert page.text_content is not None
            assert page.width > 0
            assert page.height > 0
    
    def test_detect_all_boundaries(self, test_pdf_path, expected_boundaries):
        """Test that boundary detection finds all 14 documents."""
        if not test_pdf_path.exists():
            pytest.skip(f"Test PDF not found at {test_pdf_path}")
        
        # Process document
        processor = DocumentProcessor(enable_ocr=False, page_batch_size=10)
        pages = list(processor.process_document(test_pdf_path))
        
        # Detect boundaries
        detector = BoundaryDetector(min_confidence=0.5, min_signals=1)
        detected_boundaries = detector.detect_boundaries(pages)
        
        # Should detect 14 documents
        assert len(detected_boundaries) == 14, \
            f"Expected 14 documents, detected {len(detected_boundaries)}"
        
        # Verify detected boundaries match expected
        for i, (detected, expected) in enumerate(zip(detected_boundaries, expected_boundaries)):
            assert detected.start_page == expected[0], \
                f"Document {i+1}: Expected start page {expected[0]}, got {detected.start_page}"
            assert detected.end_page == expected[1], \
                f"Document {i+1}: Expected end page {expected[1]}, got {detected.end_page}"
    
    def test_boundary_confidence_scores(self, test_pdf_path):
        """Test that detected boundaries have reasonable confidence scores."""
        if not test_pdf_path.exists():
            pytest.skip(f"Test PDF not found at {test_pdf_path}")
        
        # Process and detect
        processor = DocumentProcessor(enable_ocr=False, page_batch_size=10)
        pages = list(processor.process_document(test_pdf_path))
        detector = BoundaryDetector(min_confidence=0.5, min_signals=1)
        boundaries = detector.detect_boundaries(pages)
        
        # Check confidence scores
        for i, boundary in enumerate(boundaries):
            assert 0.0 <= boundary.confidence <= 1.0, \
                f"Boundary {i+1} has invalid confidence: {boundary.confidence}"
            
            # Most boundaries should have decent confidence
            if i > 0:  # Skip first boundary which always has confidence 1.0
                assert boundary.confidence >= 0.5, \
                    f"Boundary {i+1} has low confidence: {boundary.confidence}"
    
    def test_document_type_detection(self, test_pdf_path, ground_truth_data):
        """Test that document types are correctly identified."""
        if not test_pdf_path.exists():
            pytest.skip(f"Test PDF not found at {test_pdf_path}")
        
        # Process and detect
        processor = DocumentProcessor(enable_ocr=False, page_batch_size=10)
        pages = list(processor.process_document(test_pdf_path))
        detector = BoundaryDetector(min_confidence=0.5, min_signals=1)
        boundaries = detector.detect_boundaries(pages)
        
        # Map ground truth types
        gt_types = [doc['type'].lower() for doc in ground_truth_data['documents']]
        
        # Check document types
        correct_types = 0
        for i, (boundary, gt_type) in enumerate(zip(boundaries, gt_types)):
            if boundary.document_type:
                detected_type = boundary.document_type.value.lower()
                
                # Check for match (allowing some flexibility)
                if 'email' in gt_type and 'email' in detected_type:
                    correct_types += 1
                elif 'invoice' in gt_type and 'invoice' in detected_type:
                    correct_types += 1
                elif 'submittal' in gt_type and ('form' in detected_type or 'report' in detected_type):
                    correct_types += 1
                elif 'request for information' in gt_type and 'report' in detected_type:
                    correct_types += 1
                elif 'application' in gt_type and ('form' in detected_type or 'invoice' in detected_type):
                    correct_types += 1
                elif 'cost proposal' in gt_type and ('invoice' in detected_type or 'report' in detected_type):
                    correct_types += 1
                elif 'schedule' in gt_type and ('report' in detected_type or 'form' in detected_type):
                    correct_types += 1
                elif 'plans' in gt_type and 'report' in detected_type:
                    correct_types += 1
        
        # Should correctly identify at least 70% of document types
        accuracy = correct_types / len(boundaries)
        assert accuracy >= 0.7, f"Document type accuracy too low: {accuracy:.2%}"
    
    def test_signal_diversity(self, test_pdf_path):
        """Test that multiple signal types are being used for detection."""
        if not test_pdf_path.exists():
            pytest.skip(f"Test PDF not found at {test_pdf_path}")
        
        # Process and detect
        processor = DocumentProcessor(enable_ocr=False, page_batch_size=10)
        pages = list(processor.process_document(test_pdf_path))
        detector = BoundaryDetector(min_confidence=0.5, min_signals=1)
        boundaries = detector.detect_boundaries(pages)
        
        # Collect all signal types used
        signal_types = set()
        for boundary in boundaries:
            for signal in boundary.signals:
                signal_types.add(signal.type.value)
        
        # Should use multiple signal types
        assert len(signal_types) >= 3, \
            f"Only {len(signal_types)} signal types used: {signal_types}"
        
        # Should include common signals
        expected_signals = {'email_header', 'document_header', 'page_number_reset'}
        found_signals = signal_types & expected_signals
        assert len(found_signals) >= 2, \
            f"Missing common signals. Found: {found_signals}"
    
    def test_page_range_consistency(self, test_pdf_path):
        """Test that detected boundaries have consistent page ranges."""
        if not test_pdf_path.exists():
            pytest.skip(f"Test PDF not found at {test_pdf_path}")
        
        # Process and detect
        processor = DocumentProcessor(enable_ocr=False, page_batch_size=10)
        pages = list(processor.process_document(test_pdf_path))
        detector = BoundaryDetector(min_confidence=0.5, min_signals=1)
        boundaries = detector.detect_boundaries(pages)
        
        # Check page range consistency
        for i in range(len(boundaries)):
            boundary = boundaries[i]
            
            # Start page should be <= end page
            assert boundary.start_page <= boundary.end_page, \
                f"Boundary {i+1} has invalid page range: {boundary.start_page}-{boundary.end_page}"
            
            # No gaps between boundaries
            if i > 0:
                prev_boundary = boundaries[i-1]
                assert boundary.start_page == prev_boundary.end_page + 1, \
                    f"Gap between boundaries {i} and {i+1}"
        
        # Last boundary should end at last page
        assert boundaries[-1].end_page == 36, \
            f"Last boundary should end at page 36, ends at {boundaries[-1].end_page}"
    
    def test_detection_report(self, test_pdf_path, ground_truth_data, expected_boundaries):
        """Generate a detailed report comparing detection results with ground truth."""
        if not test_pdf_path.exists():
            pytest.skip(f"Test PDF not found at {test_pdf_path}")
        
        # Process and detect
        processor = DocumentProcessor(enable_ocr=False, page_batch_size=10)
        pages = list(processor.process_document(test_pdf_path))
        detector = BoundaryDetector(min_confidence=0.5, min_signals=1)
        boundaries = detector.detect_boundaries(pages)
        
        # Generate comparison report
        print("\n" + "="*80)
        print("BOUNDARY DETECTION VALIDATION REPORT")
        print("="*80)
        print(f"Total pages in PDF: {len(pages)}")
        print(f"Expected documents: {len(expected_boundaries)}")
        print(f"Detected documents: {len(boundaries)}")
        print()
        
        # Compare each boundary
        print("Boundary Comparison:")
        print("-"*80)
        print(f"{'Doc':<4} {'Expected':<12} {'Detected':<12} {'Match':<8} {'Type':<20} {'Confidence':<10}")
        print("-"*80)
        
        for i in range(max(len(expected_boundaries), len(boundaries))):
            if i < len(expected_boundaries) and i < len(boundaries):
                exp_start, exp_end = expected_boundaries[i]
                det = boundaries[i]
                exp_range = f"{exp_start}-{exp_end}" if exp_start != exp_end else str(exp_start)
                det_range = f"{det.start_page}-{det.end_page}" if det.start_page != det.end_page else str(det.start_page)
                match = "✓" if (det.start_page == exp_start and det.end_page == exp_end) else "✗"
                doc_type = det.document_type.value if det.document_type else "Unknown"
                gt_type = ground_truth_data['documents'][i]['type']
                
                print(f"{i+1:<4} {exp_range:<12} {det_range:<12} {match:<8} {doc_type:<20} {det.confidence:<10.2f}")
                
                # Print ground truth type if different
                if doc_type.lower() not in gt_type.lower():
                    print(f"     (Ground truth: {gt_type})")
            elif i < len(expected_boundaries):
                exp_start, exp_end = expected_boundaries[i]
                exp_range = f"{exp_start}-{exp_end}" if exp_start != exp_end else str(exp_start)
                print(f"{i+1:<4} {exp_range:<12} {'MISSING':<12} {'✗':<8}")
            else:
                det = boundaries[i]
                det_range = f"{det.start_page}-{det.end_page}"
                print(f"{i+1:<4} {'N/A':<12} {det_range:<12} {'EXTRA':<8}")
        
        print("-"*80)
        
        # Summary statistics
        correct_boundaries = sum(1 for i in range(min(len(expected_boundaries), len(boundaries)))
                               if boundaries[i].start_page == expected_boundaries[i][0]
                               and boundaries[i].end_page == expected_boundaries[i][1])
        
        accuracy = correct_boundaries / len(expected_boundaries) * 100
        print(f"\nAccuracy: {correct_boundaries}/{len(expected_boundaries)} ({accuracy:.1f}%)")
        
        # Signal analysis
        all_signals = {}
        for boundary in boundaries:
            for signal in boundary.signals:
                signal_type = signal.type.value
                all_signals[signal_type] = all_signals.get(signal_type, 0) + 1
        
        print(f"\nSignal Usage:")
        for signal_type, count in sorted(all_signals.items(), key=lambda x: x[1], reverse=True):
            print(f"  {signal_type}: {count}")
        
        print("="*80)
        
        # Assert for test passing
        assert len(boundaries) == len(expected_boundaries), \
            f"Document count mismatch: expected {len(expected_boundaries)}, got {len(boundaries)}"
        assert accuracy >= 85.0, f"Accuracy too low: {accuracy:.1f}%"


if __name__ == "__main__":
    # Run with verbose output to see the report
    pytest.main([__file__, "-v", "-s"])