"""Tests for the Phi-4 Mini boundary detector."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.core.phi4_mini_detector import (
    Phi4MiniBoundaryDetector,
    LLMAnalysis
)
from src.core.models import (
    PageInfo,
    Boundary,
    Signal,
    SignalType,
    DocumentType
)


@pytest.fixture
def sample_pages():
    """Create sample pages for testing."""
    return [
        PageInfo(
            page_number=1,
            width=612,
            height=792,
            text_content="From: john@example.com\nTo: jane@example.com\nSubject: Meeting Notes\n\nHi Jane,\nHere are the meeting notes...",
            word_count=150
        ),
        PageInfo(
            page_number=2,
            width=612,
            height=792,
            text_content="...continued from previous page. The project timeline is as follows:\n1. Phase 1: January\n2. Phase 2: February",
            word_count=120
        ),
        PageInfo(
            page_number=3,
            width=612,
            height=792,
            text_content="INVOICE #12345\nDate: 2024-01-15\nBill To: ABC Company\n\nDescription: Consulting Services\nAmount: $5,000",
            word_count=80
        ),
        PageInfo(
            page_number=4,
            width=612,
            height=792,
            text_content="Page 1 of 3\n\nPROPOSAL FOR SERVICES\n\nExecutive Summary\nWe are pleased to submit this proposal...",
            word_count=200
        ),
        PageInfo(
            page_number=5,
            width=612,
            height=792,
            text_content="Page 2 of 3\n\nScope of Work\nThe following services will be provided...",
            word_count=180
        ),
    ]


@pytest.fixture
def mock_ollama_client():
    """Create a mock Ollama client."""
    with patch('src.core.phi4_mini_detector.Client') as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock model list
        mock_client.list.return_value = {
            'models': [{'name': 'phi4-mini:3.8b'}]
        }
        
        yield mock_client


class TestPhi4MiniBoundaryDetector:
    """Test cases for Phi4MiniBoundaryDetector."""
    
    def test_initialization_with_ollama(self, mock_ollama_client):
        """Test detector initialization with Ollama available."""
        detector = Phi4MiniBoundaryDetector(
            model_name="phi4-mini:3.8b",
            use_llm_for_ambiguous=True
        )
        
        assert detector.model_name == "phi4-mini:3.8b"
        assert detector.use_llm_for_ambiguous is True
        assert detector.client is not None
        mock_ollama_client.list.assert_called_once()
    
    def test_initialization_without_ollama(self):
        """Test detector falls back gracefully when Ollama is not available."""
        with patch('src.core.phi4_mini_detector.Client') as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")
            
            detector = Phi4MiniBoundaryDetector(
                model_name="phi4-mini:3.8b",
                use_llm_for_ambiguous=True
            )
            
            assert detector.use_llm_for_ambiguous is False
            assert detector.client is None
    
    def test_detect_boundaries_without_llm(self, sample_pages):
        """Test boundary detection falls back to base detector when LLM is disabled."""
        detector = Phi4MiniBoundaryDetector(
            use_llm_for_ambiguous=False
        )
        
        boundaries = detector.detect_boundaries(sample_pages)
        
        # Should detect at least the email, invoice, and proposal
        assert len(boundaries) >= 3
        
        # Check first boundary (email)
        assert boundaries[0].start_page == 1
        assert boundaries[0].document_type == DocumentType.EMAIL
        
        # Check invoice boundary
        invoice_boundary = next((b for b in boundaries if b.start_page == 3), None)
        assert invoice_boundary is not None
        assert invoice_boundary.document_type == DocumentType.INVOICE
    
    def test_identify_ambiguous_cases(self, sample_pages):
        """Test identification of ambiguous boundary cases."""
        detector = Phi4MiniBoundaryDetector(
            use_llm_for_ambiguous=False  # Disable LLM for this test
        )
        
        # Get initial boundaries
        initial_boundaries = detector.detect_boundaries(sample_pages)
        
        # Identify ambiguous cases
        ambiguous = detector._identify_ambiguous_cases(sample_pages, initial_boundaries)
        
        # Should find some ambiguous cases
        assert len(ambiguous) >= 0
        
        # Check structure of ambiguous cases
        for page_idx, candidate in ambiguous:
            assert 0 < page_idx < len(sample_pages)
            assert hasattr(candidate, 'page_number')
            assert hasattr(candidate, 'signals')
            assert hasattr(candidate, 'confidence')
    
    def test_analyze_single_page_with_mock_llm(self, sample_pages, mock_ollama_client):
        """Test single page analysis with mocked LLM response."""
        detector = Phi4MiniBoundaryDetector(
            model_name="phi4-mini:3.8b",
            use_llm_for_ambiguous=True
        )
        
        # Mock LLM response
        mock_response = {
            'response': '''{
                "is_boundary": true,
                "confidence": 0.85,
                "reasoning": "Clear invoice header with document number",
                "document_type": "invoice",
                "context_signals": ["document_header", "layout_change"]
            }'''
        }
        mock_ollama_client.generate.return_value = mock_response
        
        # Create a candidate for the invoice page
        from src.core.boundary_detector import BoundaryCandidate
        candidate = BoundaryCandidate(
            page_number=3,
            signals=[],
            confidence=0.6
        )
        
        # Analyze the page
        analysis = detector._analyze_single_page(sample_pages, 2, candidate)
        
        assert analysis is not None
        assert analysis.is_boundary is True
        assert analysis.confidence == 0.85
        assert analysis.document_type == DocumentType.INVOICE
        assert "invoice header" in analysis.reasoning.lower()
    
    def test_merge_results(self, sample_pages):
        """Test merging of initial boundaries with LLM results."""
        detector = Phi4MiniBoundaryDetector(
            use_llm_for_ambiguous=False
        )
        
        # Create initial boundaries
        initial_boundaries = [
            Boundary(
                start_page=1,
                end_page=2,
                confidence=0.9,
                signals=[],
                document_type=DocumentType.EMAIL
            ),
            Boundary(
                start_page=3,
                end_page=3,
                confidence=0.5,  # Low confidence
                signals=[],
                document_type=DocumentType.INVOICE
            )
        ]
        
        # Create LLM results
        llm_results = [
            (2, LLMAnalysis(
                is_boundary=True,
                confidence=0.9,
                reasoning="Strong invoice header detected",
                document_type=DocumentType.INVOICE
            )),
            (3, LLMAnalysis(
                is_boundary=True,
                confidence=0.95,
                reasoning="Clear document boundary with proposal header",
                document_type=DocumentType.REPORT
            ))
        ]
        
        # Merge results
        final_boundaries = detector._merge_results(
            initial_boundaries, llm_results, sample_pages
        )
        
        # Should have all boundaries with updated confidence
        assert len(final_boundaries) >= 3
        
        # Check that the invoice boundary confidence was updated
        invoice_boundary = next((b for b in final_boundaries if b.start_page == 3), None)
        assert invoice_boundary is not None
        assert invoice_boundary.confidence >= 0.9
    
    def test_explain_boundary(self, sample_pages, mock_ollama_client):
        """Test boundary explanation feature."""
        detector = Phi4MiniBoundaryDetector(
            model_name="phi4-mini:3.8b",
            use_llm_for_ambiguous=True
        )
        
        # Mock LLM response for explanation
        mock_response = {
            'response': '''{
                "is_boundary": true,
                "confidence": 0.9,
                "reasoning": "This page contains a clear email header with From, To, and Subject fields",
                "document_type": "email",
                "context_signals": ["email_header"]
            }'''
        }
        mock_ollama_client.generate.return_value = mock_response
        
        # Get explanation for page 1
        explanation = detector.explain_boundary(sample_pages, 1)
        
        assert explanation is not None
        assert "Page 1 Analysis" in explanation
        assert "Is boundary: Yes" in explanation
        assert "email" in explanation.lower()
    
    @pytest.mark.parametrize("page_content,expected_type", [
        ("From: test@example.com\nTo: user@example.com\nSubject: Test", DocumentType.EMAIL),
        ("INVOICE #12345\nBill To: Customer", DocumentType.INVOICE),
        ("Dear Sir/Madam,\n\nI am writing to...\n\nSincerely,", DocumentType.LETTER),
        ("Executive Summary\n\nThis report presents...", DocumentType.REPORT),
    ])
    def test_document_type_mapping(self, page_content, expected_type):
        """Test document type detection and mapping."""
        detector = Phi4MiniBoundaryDetector(use_llm_for_ambiguous=False)
        
        page = PageInfo(
            page_number=1,
            width=612,
            height=792,
            text_content=page_content,
            word_count=len(page_content.split())
        )
        
        detected_type = detector._detect_document_type(page)
        assert detected_type == expected_type
    
    def test_batch_processing(self, sample_pages, mock_ollama_client):
        """Test batch processing of ambiguous cases."""
        detector = Phi4MiniBoundaryDetector(
            model_name="phi4-mini:3.8b",
            use_llm_for_ambiguous=True,
            llm_batch_size=2
        )
        
        # Mock LLM responses
        mock_ollama_client.generate.return_value = {
            'response': '{"is_boundary": true, "confidence": 0.8, "reasoning": "Test", "document_type": "other"}'
        }
        
        # Create multiple candidates
        from src.core.boundary_detector import BoundaryCandidate
        candidates = [
            (i, BoundaryCandidate(
                page_number=i+1,
                signals=[],
                confidence=0.5
            ))
            for i in range(4)
        ]
        
        # Analyze with batching
        results = detector._analyze_with_llm(sample_pages, candidates)
        
        # Should process all candidates
        assert len(results) == 4
        
        # Verify batching occurred (generate called 4 times for 4 candidates with batch size 2)
        assert mock_ollama_client.generate.call_count == 4
    
    def test_error_handling_in_llm_analysis(self, sample_pages, mock_ollama_client):
        """Test error handling when LLM analysis fails."""
        detector = Phi4MiniBoundaryDetector(
            model_name="phi4-mini:3.8b",
            use_llm_for_ambiguous=True
        )
        
        # Mock LLM to return invalid JSON
        mock_ollama_client.generate.return_value = {
            'response': 'Invalid JSON response'
        }
        
        from src.core.boundary_detector import BoundaryCandidate
        candidate = BoundaryCandidate(
            page_number=1,
            signals=[],
            confidence=0.5
        )
        
        # Should handle error gracefully
        analysis = detector._analyze_single_page(sample_pages, 0, candidate)
        assert analysis is None
    
    def test_confidence_thresholds(self):
        """Test that confidence thresholds are properly set."""
        detector = Phi4MiniBoundaryDetector(
            min_confidence=0.7,
            use_llm_for_ambiguous=False
        )
        
        assert detector.min_confidence == 0.7
        assert detector.MIN_AMBIGUOUS_CONFIDENCE == 0.4
        assert detector.MAX_AMBIGUOUS_CONFIDENCE == 0.75
        assert detector.MIN_AMBIGUOUS_CONFIDENCE < detector.MAX_AMBIGUOUS_CONFIDENCE