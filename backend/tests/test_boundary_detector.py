"""Test boundary detection against ground truth data."""

import json
import os
from pathlib import Path

import pytest

from src.core.models import Boundary, DocumentType


@pytest.fixture
def ground_truth_data():
    """Load ground truth data for testing."""
    test_file = Path(__file__).parent.parent.parent / "tests" / "test_files" / "Test_PDF_Set_Ground_Truth.json"
    with open(test_file, "r") as f:
        return json.load(f)


@pytest.fixture
def test_pdf_path():
    """Get path to test PDF file."""
    return Path(__file__).parent.parent.parent / "tests" / "test_files" / "Test_PDF_Set_1.pdf"


def parse_page_range(page_range: str) -> tuple[int, int]:
    """Parse page range string (e.g., '1-4' or '5') into start and end pages."""
    if "-" in page_range:
        start, end = page_range.split("-")
        return int(start), int(end)
    else:
        page = int(page_range)
        return page, page


@pytest.mark.requires_pdf
class TestBoundaryDetection:
    """Test boundary detection against known ground truth."""
    
    def test_ground_truth_structure(self, ground_truth_data):
        """Test that ground truth data has expected structure."""
        assert "documents" in ground_truth_data
        assert len(ground_truth_data["documents"]) == 14
        
        # Check first document
        first_doc = ground_truth_data["documents"][0]
        assert "pages" in first_doc
        assert "type" in first_doc
        assert "summary" in first_doc
        assert first_doc["pages"] == "1-4"
        assert first_doc["type"] == "Email Chain"
    
    def test_pdf_exists(self, test_pdf_path):
        """Test that test PDF file exists."""
        assert test_pdf_path.exists()
        assert test_pdf_path.suffix == ".pdf"
    
    @pytest.mark.skip(reason="BoundaryDetector not implemented yet")
    def test_detect_all_boundaries(self, test_pdf_path, ground_truth_data):
        """Test that all 14 documents are correctly detected."""
        # This test will be implemented when we create the BoundaryDetector
        from src.core.boundary_detector import BoundaryDetector
        
        detector = BoundaryDetector()
        boundaries = detector.detect_boundaries(test_pdf_path)
        
        # Should detect 14 documents
        assert len(boundaries) == 14
        
        # Check each boundary matches ground truth
        for i, (detected, expected) in enumerate(zip(boundaries, ground_truth_data["documents"])):
            start, end = parse_page_range(expected["pages"])
            assert detected.start_page == start, f"Document {i}: start page mismatch"
            assert detected.end_page == end, f"Document {i}: end page mismatch"
    
    def test_email_detection(self, ground_truth_data):
        """Test email document type detection."""
        email_docs = [doc for doc in ground_truth_data["documents"] if "Email" in doc["type"]]
        assert len(email_docs) >= 4  # We know there are at least 4 email-related documents
        
        # First document should be an email chain
        assert ground_truth_data["documents"][0]["type"] == "Email Chain"
        assert ground_truth_data["documents"][0]["pages"] == "1-4"
    
    def test_invoice_detection(self, ground_truth_data):
        """Test invoice document type detection."""
        invoice_docs = [doc for doc in ground_truth_data["documents"] if "Invoice" in doc["type"]]
        assert len(invoice_docs) >= 1  # At least one invoice in the test set
    
    def test_total_page_coverage(self, ground_truth_data):
        """Test that all pages are covered by boundaries."""
        all_pages = set()
        
        for doc in ground_truth_data["documents"]:
            start, end = parse_page_range(doc["pages"])
            for page in range(start, end + 1):
                assert page not in all_pages, f"Page {page} is in multiple documents"
                all_pages.add(page)
        
        # Check if pages are consecutive (no gaps)
        page_list = sorted(all_pages)
        assert page_list == list(range(1, max(all_pages) + 1))
        
        # Total unique pages
        print(f"Total pages covered: {len(all_pages)}")


@pytest.mark.skip(reason="Components not implemented yet")
class TestBoundaryDetectorComponents:
    """Test individual components of boundary detection."""
    
    def test_email_header_detection(self):
        """Test detection of email headers."""
        from src.core.boundary_detector import detect_email_headers
        
        text = "From: John Doe <john@example.com>\nTo: Jane Smith\nSubject: Test Email"
        assert detect_email_headers(text) == True
        
        text = "This is regular text without email headers"
        assert detect_email_headers(text) == False
    
    def test_page_number_reset_detection(self):
        """Test detection of page number resets."""
        from src.core.boundary_detector import detect_page_number_reset
        
        # Page 1 indicates potential new document
        assert detect_page_number_reset("Page 1 of 5") == True
        assert detect_page_number_reset("1") == True
        
        # Other page numbers don't indicate reset
        assert detect_page_number_reset("Page 2 of 5") == False
        assert detect_page_number_reset("Page 15") == False
    
    def test_confidence_scoring(self):
        """Test confidence score calculation."""
        from src.core.boundary_detector import calculate_confidence
        from src.core.models import Signal, SignalType
        
        signals = [
            Signal(type=SignalType.EMAIL_HEADER, confidence=0.9, page_number=1),
            Signal(type=SignalType.PAGE_NUMBER_RESET, confidence=0.8, page_number=1),
        ]
        
        confidence = calculate_confidence(signals)
        assert 0.8 <= confidence <= 1.0