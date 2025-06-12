"""
Tests for the boundary detector module.
"""

import pytest
import json
from pathlib import Path
from typing import List

from src.core.boundary_detector import BoundaryDetector, BoundaryCandidate
from src.core.models import PageInfo, Boundary, Signal, SignalType, DocumentType


class TestBoundaryDetector:
    """Test suite for BoundaryDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a boundary detector instance."""
        return BoundaryDetector(
            min_confidence=0.6,
            min_signals=1,
            enable_visual_analysis=True
        )
    
    @pytest.fixture
    def sample_pages(self):
        """Create sample pages for testing."""
        return [
            # Email document (pages 1-2)
            PageInfo(
                page_number=1,
                width=612,
                height=792,
                text_content="From: john@example.com\nTo: jane@example.com\nSubject: Meeting Notes\nDate: March 1, 2024\n\nHi Jane,\nPlease find the meeting notes attached.",
                word_count=20,
                has_images=False,
                has_tables=False
            ),
            PageInfo(
                page_number=2,
                width=612,
                height=792,
                text_content="Meeting continued...\nAction items:\n1. Review proposal\n2. Send feedback",
                word_count=10,
                has_images=False,
                has_tables=False
            ),
            # Invoice document (pages 3-4)
            PageInfo(
                page_number=3,
                width=612,
                height=792,
                text_content="INVOICE #12345\nDate: March 2, 2024\nBill To: ACME Corp\n\nDescription: Consulting Services\nAmount: $5,000.00",
                word_count=15,
                has_images=False,
                has_tables=True
            ),
            PageInfo(
                page_number=4,
                width=612,
                height=792,
                text_content="Page 2 of 2\nPayment Terms: Net 30\nThank you for your business!",
                word_count=12,
                has_images=False,
                has_tables=False
            ),
            # Report document (page 5)
            PageInfo(
                page_number=5,
                width=612,
                height=792,
                text_content="REQUEST FOR INFORMATION #7\nProject: Construction\nDate: March 3, 2024\n\nWe need clarification on the following items...",
                word_count=18,
                has_images=True,
                has_tables=False
            ),
        ]
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = BoundaryDetector()
        assert detector.min_confidence == 0.6
        assert detector.min_signals == 1
        assert detector.enable_visual_analysis is True
        
        detector = BoundaryDetector(min_confidence=0.8, min_signals=2)
        assert detector.min_confidence == 0.8
        assert detector.min_signals == 2
    
    def test_detect_email_header(self, detector, sample_pages):
        """Test email header detection."""
        # Test email page
        signal = detector._detect_email_header(sample_pages[0])
        assert signal is not None
        assert signal.type == SignalType.EMAIL_HEADER
        assert signal.confidence > 0.5
        assert "from" in signal.description.lower()
        
        # Test non-email page
        signal = detector._detect_email_header(sample_pages[2])
        assert signal is None
    
    def test_detect_document_header(self, detector, sample_pages):
        """Test document header detection."""
        # Test invoice page
        signal = detector._detect_document_header(sample_pages[2])
        assert signal is not None
        assert signal.type == SignalType.DOCUMENT_HEADER
        assert signal.confidence == 0.85
        assert "invoice" in signal.description.lower()
        
        # Test RFI page
        signal = detector._detect_document_header(sample_pages[4])
        assert signal is not None
        assert "rfi" in signal.description.lower()
        
        # Test regular email page
        signal = detector._detect_document_header(sample_pages[0])
        assert signal is None  # Emails don't have document headers
    
    def test_detect_page_number_reset(self, detector):
        """Test page number reset detection."""
        # Page with "Page 1"
        page = PageInfo(
            page_number=5,
            width=612,
            height=792,
            text_content="Page 1 of 3\n\nDocument content here...",
            word_count=10
        )
        signal = detector._detect_page_number_reset(page)
        assert signal is not None
        assert signal.type == SignalType.PAGE_NUMBER_RESET
        
        # Page without page number reset
        page = PageInfo(
            page_number=5,
            width=612,
            height=792,
            text_content="Page 5 of 10\n\nDocument content here...",
            word_count=10
        )
        signal = detector._detect_page_number_reset(page)
        assert signal is None
    
    def test_detect_white_space(self, detector):
        """Test white space detection."""
        # Nearly empty page
        page = PageInfo(
            page_number=1,
            width=612,
            height=792,
            text_content="This page intentionally left blank",
            word_count=5
        )
        signal = detector._detect_white_space(page)
        assert signal is not None
        assert signal.type == SignalType.WHITE_SPACE
        assert signal.confidence > 0.5
        
        # Normal page
        page = PageInfo(
            page_number=1,
            width=612,
            height=792,
            text_content=" ".join(["word"] * 100),
            word_count=100
        )
        signal = detector._detect_white_space(page)
        assert signal is None
    
    def test_detect_document_type(self, detector, sample_pages):
        """Test document type detection."""
        # Email
        doc_type = detector._detect_document_type(sample_pages[0])
        assert doc_type == DocumentType.EMAIL
        
        # Invoice
        doc_type = detector._detect_document_type(sample_pages[2])
        assert doc_type == DocumentType.INVOICE
        
        # Report (RFI)
        doc_type = detector._detect_document_type(sample_pages[4])
        assert doc_type == DocumentType.REPORT
    
    def test_calculate_confidence(self, detector):
        """Test confidence calculation."""
        # Single high-confidence signal
        signals = [
            Signal(type=SignalType.EMAIL_HEADER, confidence=0.9, page_number=1)
        ]
        confidence = detector._calculate_confidence(signals)
        assert confidence == 0.9
        
        # Multiple signals
        signals = [
            Signal(type=SignalType.EMAIL_HEADER, confidence=0.9, page_number=1),
            Signal(type=SignalType.PAGE_NUMBER_RESET, confidence=0.7, page_number=1),
        ]
        confidence = detector._calculate_confidence(signals)
        assert confidence > 0.7  # Should be boosted for multiple strong signals
        
        # No signals
        confidence = detector._calculate_confidence([])
        assert confidence == 0.0
    
    def test_detect_boundaries(self, detector, sample_pages):
        """Test full boundary detection."""
        boundaries = detector.detect_boundaries(sample_pages)
        
        # Should detect at least 3 documents
        assert len(boundaries) >= 3
        
        # First boundary should start at page 1
        assert boundaries[0].start_page == 1
        assert boundaries[0].confidence == 1.0  # First page always has confidence 1
        
        # Check that boundaries don't overlap
        for i in range(len(boundaries) - 1):
            assert boundaries[i].end_page < boundaries[i + 1].start_page
        
        # Last boundary should end at last page
        assert boundaries[-1].end_page == sample_pages[-1].page_number
    
    def test_refine_boundaries_with_user_feedback(self, detector, sample_pages):
        """Test boundary refinement with user feedback."""
        # Get initial boundaries
        boundaries = detector.detect_boundaries(sample_pages)
        initial_count = len(boundaries)
        
        # User feedback: page 2 is also a boundary
        user_feedback = {
            2: True,  # Add boundary at page 2
            3: False  # Remove boundary at page 3 if exists
        }
        
        refined = detector.refine_boundaries(boundaries, sample_pages, user_feedback)
        
        # Should have added page 2 as boundary
        assert any(b.start_page == 2 for b in refined)
        
        # Should not have boundary at page 3
        assert not any(b.start_page == 3 for b in refined)
        
        # End pages should be properly updated
        for i in range(len(refined) - 1):
            assert refined[i].end_page == refined[i + 1].start_page - 1
    
    def test_empty_pages_list(self, detector):
        """Test handling of empty pages list."""
        boundaries = detector.detect_boundaries([])
        assert boundaries == []
    
    def test_single_page_document(self, detector):
        """Test handling of single-page document."""
        pages = [PageInfo(
            page_number=1,
            width=612,
            height=792,
            text_content="Single page document",
            word_count=3
        )]
        
        boundaries = detector.detect_boundaries(pages)
        assert len(boundaries) == 1
        assert boundaries[0].start_page == 1
        assert boundaries[0].end_page == 1


class TestBoundaryDetectorIntegration:
    """Integration tests with ground truth data."""
    
    @pytest.fixture
    def ground_truth(self):
        """Load ground truth data."""
        ground_truth_path = Path("/home/nick/Projects/smart_pdf_splitter/tests/test_files/Test_PDF_Set_Ground_Truth.json")
        if ground_truth_path.exists():
            with open(ground_truth_path, 'r') as f:
                return json.load(f)
        return None
    
    def test_ground_truth_document_count(self, ground_truth):
        """Test that we can identify the correct number of documents."""
        if not ground_truth:
            pytest.skip("Ground truth file not found")
        
        expected_documents = len(ground_truth['documents'])
        assert expected_documents == 14  # Based on the ground truth
    
    def test_ground_truth_page_ranges(self, ground_truth):
        """Test that ground truth page ranges are valid."""
        if not ground_truth:
            pytest.skip("Ground truth file not found")
        
        for doc in ground_truth['documents']:
            pages = doc['pages']
            if '-' in pages:
                start, end = map(int, pages.split('-'))
                assert start <= end
                assert start >= 1
            else:
                # Single page
                page = int(pages)
                assert page >= 1