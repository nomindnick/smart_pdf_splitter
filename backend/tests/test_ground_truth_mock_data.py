"""
Test boundary detection against ground truth using mock data.

Since the test PDF is scanned and requires OCR (which is slow),
this test creates mock page data based on the ground truth JSON
to verify that our boundary detection logic works correctly.
"""

import pytest
import json
from pathlib import Path
from typing import List, Tuple

from src.core.boundary_detector import BoundaryDetector
from src.core.models import PageInfo, DocumentType


class TestGroundTruthWithMockData:
    """Test boundary detection using mock data based on ground truth."""
    
    @pytest.fixture
    def ground_truth_path(self):
        """Path to ground truth JSON file."""
        return Path("/home/nick/Projects/smart_pdf_splitter/tests/test_files/Test_PDF_Set_Ground_Truth.json")
    
    @pytest.fixture
    def ground_truth_data(self, ground_truth_path):
        """Load and return ground truth data."""
        with open(ground_truth_path, 'r') as f:
            return json.load(f)
    
    @pytest.fixture
    def mock_pages(self, ground_truth_data) -> List[PageInfo]:
        """Create mock pages based on ground truth data."""
        pages = []
        
        # Create pages based on ground truth documents
        for doc in ground_truth_data['documents']:
            # Parse page range
            pages_str = doc['pages']
            if '-' in pages_str:
                start, end = map(int, pages_str.split('-'))
            else:
                start = end = int(pages_str)
            
            # Get document type and create appropriate content
            doc_type = doc['type'].lower()
            
            for page_num in range(start, end + 1):
                # Create page content based on document type and position
                if page_num == start:  # First page of document
                    if 'email' in doc_type:
                        text = f"From: sender@example.com\\nTo: recipient@example.com\\nSubject: Test Email\\nDate: March 1, 2024\\n\\n{doc['summary'][:100]}"
                    elif 'invoice' in doc_type:
                        text = f"INVOICE #12345\\nDate: March 1, 2024\\n\\n{doc['summary'][:100]}"
                    elif 'submittal' in doc_type:
                        text = f"Submittal Transmittal #0007\\nDate: 2/29/2024\\n\\n{doc['summary'][:100]}"
                    elif 'schedule' in doc_type:
                        text = f"SCHEDULE OF VALUES\\nContract Amount: $1,459,395.00\\n\\n{doc['summary'][:100]}"
                    elif 'application' in doc_type:
                        text = f"APPLICATION FOR PAYMENT\\nPeriod Ending: 1/31/2024\\n\\n{doc['summary'][:100]}"
                    elif 'request for information' in doc_type:
                        text = f"REQUEST FOR INFORMATION #7\\nDate: March 28, 2024\\n\\n{doc['summary'][:100]}"
                    elif 'cost proposal' in doc_type:
                        text = f"COST PROPOSAL #2\\nDate: 4/5/2024\\n\\n{doc['summary'][:100]}"
                    elif 'plans' in doc_type:
                        text = f"STRUCTURAL ENGINEERING DRAWINGS\\nSheet SI-1.1\\n\\n{doc['summary'][:100]}"
                    else:
                        text = f"Document Type: {doc_type}\\n\\n{doc['summary'][:100]}"
                else:
                    # Continuation pages
                    if page_num == end and (end - start) > 0:
                        text = f"Page {page_num - start + 1} of {end - start + 1}\\n\\nContinued from previous page..."
                    else:
                        text = f"Continued from page {page_num - 1}\\n\\nDocument content here..."
                
                # Add page
                pages.append(PageInfo(
                    page_number=page_num,
                    width=612,
                    height=792,
                    text_content=text,
                    word_count=len(text.split()),
                    has_images='plans' in doc_type or 'invoice' in doc_type,
                    has_tables='schedule' in doc_type or 'invoice' in doc_type or 'application' in doc_type
                ))
        
        return pages
    
    def test_mock_data_creation(self, mock_pages, ground_truth_data):
        """Test that mock data was created correctly."""
        # Should have 36 pages total
        assert len(mock_pages) == 36
        
        # Check page numbers are sequential
        for i, page in enumerate(mock_pages):
            assert page.page_number == i + 1
        
        # Check that we have appropriate content
        assert all(page.text_content for page in mock_pages)
        assert all(page.word_count > 0 for page in mock_pages)
    
    def test_boundary_detection_on_mock_data(self, mock_pages, ground_truth_data):
        """Test boundary detection on mock data."""
        detector = BoundaryDetector(min_confidence=0.5, min_signals=1)
        boundaries = detector.detect_boundaries(mock_pages)
        
        # Should detect correct number of documents
        expected_count = len(ground_truth_data['documents'])
        assert len(boundaries) == expected_count, \
            f"Expected {expected_count} documents, detected {len(boundaries)}"
    
    def test_boundary_positions(self, mock_pages, ground_truth_data):
        """Test that boundaries are detected at correct positions."""
        detector = BoundaryDetector(min_confidence=0.5, min_signals=1)
        boundaries = detector.detect_boundaries(mock_pages)
        
        # Extract expected boundaries from ground truth
        expected_boundaries = []
        for doc in ground_truth_data['documents']:
            pages = doc['pages']
            if '-' in pages:
                start, end = map(int, pages.split('-'))
            else:
                start = end = int(pages)
            expected_boundaries.append((start, end))
        
        # Compare detected vs expected
        for i, (detected, expected) in enumerate(zip(boundaries, expected_boundaries)):
            assert detected.start_page == expected[0], \
                f"Document {i+1}: Expected start {expected[0]}, got {detected.start_page}"
            assert detected.end_page == expected[1], \
                f"Document {i+1}: Expected end {expected[1]}, got {detected.end_page}"
    
    def test_document_type_detection(self, mock_pages, ground_truth_data):
        """Test document type detection accuracy."""
        detector = BoundaryDetector(min_confidence=0.5, min_signals=1)
        boundaries = detector.detect_boundaries(mock_pages)
        
        # Check document types
        type_matches = 0
        for i, (boundary, gt_doc) in enumerate(zip(boundaries, ground_truth_data['documents'])):
            if boundary.document_type:
                detected = boundary.document_type.value.lower()
                expected = gt_doc['type'].lower()
                
                # Flexible matching
                if ('email' in expected and 'email' in detected) or \
                   ('invoice' in expected and 'invoice' in detected) or \
                   ('form' in expected and detected in ['form', 'report']) or \
                   ('report' in expected and detected in ['report', 'form']):
                    type_matches += 1
        
        accuracy = type_matches / len(boundaries)
        assert accuracy >= 0.5, f"Document type accuracy too low: {accuracy:.2%}"
    
    def test_signal_generation(self, mock_pages):
        """Test that appropriate signals are generated."""
        detector = BoundaryDetector(min_confidence=0.5, min_signals=1)
        boundaries = detector.detect_boundaries(mock_pages)
        
        # Collect all signal types
        signal_types = set()
        for boundary in boundaries:
            for signal in boundary.signals:
                signal_types.add(signal.type.value)
        
        # Should have multiple signal types
        assert len(signal_types) >= 2, f"Only {len(signal_types)} signal types found"
        
        # Should include email and document headers
        assert 'email_header' in signal_types or 'document_header' in signal_types
    
    def test_detection_report(self, mock_pages, ground_truth_data):
        """Generate a detailed detection report."""
        detector = BoundaryDetector(min_confidence=0.5, min_signals=1)
        boundaries = detector.detect_boundaries(mock_pages)
        
        print("\n" + "="*80)
        print("MOCK DATA BOUNDARY DETECTION REPORT")
        print("="*80)
        print(f"Total pages: {len(mock_pages)}")
        print(f"Expected documents: {len(ground_truth_data['documents'])}")
        print(f"Detected documents: {len(boundaries)}")
        print()
        
        # Extract expected boundaries
        expected_boundaries = []
        for doc in ground_truth_data['documents']:
            pages = doc['pages']
            if '-' in pages:
                start, end = map(int, pages.split('-'))
            else:
                start = end = int(pages)
            expected_boundaries.append((start, end, doc['type']))
        
        # Compare results
        print("Document Comparison:")
        print("-"*80)
        print(f"{'Doc':<4} {'Expected':<12} {'Detected':<12} {'Match':<8} {'Type Match':<12}")
        print("-"*80)
        
        for i in range(max(len(expected_boundaries), len(boundaries))):
            if i < len(expected_boundaries) and i < len(boundaries):
                exp_start, exp_end, exp_type = expected_boundaries[i]
                det = boundaries[i]
                
                exp_range = f"{exp_start}-{exp_end}" if exp_start != exp_end else str(exp_start)
                det_range = f"{det.start_page}-{det.end_page}" if det.start_page != det.end_page else str(det.start_page)
                
                pos_match = "✓" if (det.start_page == exp_start and det.end_page == exp_end) else "✗"
                
                det_type = det.document_type.value if det.document_type else "Unknown"
                type_match = "✓" if det_type.lower() in exp_type.lower() or exp_type.lower() in det_type.lower() else "~"
                
                print(f"{i+1:<4} {exp_range:<12} {det_range:<12} {pos_match:<8} {type_match:<12}")
                
                # Show confidence and signals
                signals = [s.type.value for s in det.signals]
                print(f"     Confidence: {det.confidence:.2f}, Signals: {', '.join(signals)}")
        
        print("-"*80)
        
        # Summary
        pos_correct = sum(1 for i in range(min(len(expected_boundaries), len(boundaries)))
                         if boundaries[i].start_page == expected_boundaries[i][0]
                         and boundaries[i].end_page == expected_boundaries[i][1])
        
        accuracy = pos_correct / len(expected_boundaries) * 100 if expected_boundaries else 0
        print(f"\nPosition Accuracy: {pos_correct}/{len(expected_boundaries)} ({accuracy:.1f}%)")
        print("="*80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])