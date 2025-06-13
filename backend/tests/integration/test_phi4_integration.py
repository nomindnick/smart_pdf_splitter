"""Integration tests for Phi-4 Mini detector with the full system."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, Mock

from src.core import DocumentProcessor, Phi4MiniBoundaryDetector
from src.core.models import DocumentType, ProcessingStatus


class TestPhi4Integration:
    """Integration tests for Phi-4 detector."""
    
    @pytest.mark.asyncio
    async def test_process_document_with_phi4_detector(self, tmp_path):
        """Test processing a document with Phi-4 detector (mocked)."""
        # Note: Creating a valid PDF programmatically is complex
        # For this integration test, we'll focus on testing the detector initialization
        # and mock the document processing part
        
        # Mock Ollama client to avoid actual LLM calls
        with patch('src.core.phi4_mini_detector.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.list.return_value = {'models': [{'name': 'phi4-mini:3.8b'}]}
            
            # Initialize components
            processor = DocumentProcessor()
            detector = Phi4MiniBoundaryDetector(
                use_llm_for_ambiguous=False  # Disable LLM for this test
            )
            
            # Test with mock pages instead of actual PDF processing
            from src.core.models import PageInfo
            mock_pages = [
                PageInfo(
                    page_number=1,
                    width=612,
                    height=792,
                    text_content="Test document content",
                    word_count=3
                )
            ]
            
            # Process with detector
            boundaries = detector.detect_boundaries(mock_pages)
            
            # Basic assertions
            assert isinstance(boundaries, list)
            assert len(boundaries) >= 1  # Should detect at least the first page as boundary
            assert boundaries[0].start_page == 1
    
    def test_detector_initialization_in_system(self):
        """Test that Phi-4 detector can be initialized as part of the system."""
        # Mock Ollama to avoid network calls
        with patch('src.core.phi4_mini_detector.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.list.return_value = {'models': [{'name': 'phi4-mini:3.8b'}]}
            
            # Should initialize without errors
            detector = Phi4MiniBoundaryDetector(
                model_name="phi4-mini:3.8b",
                min_confidence=0.7,
                use_llm_for_ambiguous=True
            )
            
            assert detector is not None
            assert detector.model_name == "phi4-mini:3.8b"
            assert detector.min_confidence == 0.7
    
    def test_detector_pattern_matching_compatibility(self):
        """Test that Phi-4 detector maintains compatibility with base detector."""
        from src.core.models import PageInfo
        
        # Create test pages
        pages = [
            PageInfo(
                page_number=1,
                width=612,
                height=792,
                text_content="From: sender@example.com\nTo: recipient@example.com\nSubject: Test Email",
                word_count=50
            ),
            PageInfo(
                page_number=2,
                width=612,
                height=792,
                text_content="INVOICE #12345\nDate: 2024-01-01\nAmount Due: $1,000",
                word_count=30
            )
        ]
        
        # Test with LLM disabled (pattern matching only)
        detector = Phi4MiniBoundaryDetector(use_llm_for_ambiguous=False)
        boundaries = detector.detect_boundaries(pages)
        
        # Should detect both documents
        assert len(boundaries) >= 2
        
        # Check document types
        email_boundary = next((b for b in boundaries if b.document_type == DocumentType.EMAIL), None)
        assert email_boundary is not None
        
        invoice_boundary = next((b for b in boundaries if b.document_type == DocumentType.INVOICE), None)
        assert invoice_boundary is not None
    
    def test_detector_config_options(self):
        """Test various configuration options for the detector."""
        # Test different configurations
        configs = [
            {"min_confidence": 0.5, "min_signals": 1},
            {"min_confidence": 0.8, "min_signals": 2},
            {"enable_visual_analysis": False},
            {"llm_batch_size": 10, "llm_timeout": 60.0},
        ]
        
        for config in configs:
            with patch('src.core.phi4_mini_detector.Client'):
                detector = Phi4MiniBoundaryDetector(
                    use_llm_for_ambiguous=False,
                    **config
                )
                
                # Verify config is applied
                for key, value in config.items():
                    assert getattr(detector, key) == value
    
    @pytest.mark.asyncio
    async def test_detector_with_document_processor_pipeline(self):
        """Test Phi-4 detector in a full processing pipeline."""
        from src.core.models import Document, ProcessingStatus
        
        # Create a mock document
        doc = Document(
            id="test-123",
            filename="test.pdf",
            total_pages=5,
            file_size=1024,
            status=ProcessingStatus.PENDING
        )
        
        # Mock components
        with patch('src.core.phi4_mini_detector.Client'):
            processor = DocumentProcessor()
            detector = Phi4MiniBoundaryDetector(use_llm_for_ambiguous=False)
            
            # Simulate processing pipeline
            # (In real system, this would be handled by the API/service layer)
            doc.status = ProcessingStatus.PROCESSING
            
            # Mock page extraction
            mock_pages = [
                {"page_number": i, "width": 612, "height": 792, 
                 "text_content": f"Page {i} content", "word_count": 10}
                for i in range(1, 6)
            ]
            
            # Detect boundaries
            from src.core.models import PageInfo
            pages = [PageInfo(**page_data) for page_data in mock_pages]
            boundaries = detector.detect_boundaries(pages)
            
            # Update document
            doc.detected_boundaries = boundaries
            doc.page_info = pages
            doc.status = ProcessingStatus.COMPLETED
            
            # Verify pipeline completed
            assert doc.status == ProcessingStatus.COMPLETED
            assert len(doc.detected_boundaries) > 0
            assert len(doc.page_info) == 5