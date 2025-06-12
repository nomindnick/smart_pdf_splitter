"""
Unit tests for the document processor module with proper mocking.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock

from src.core.document_processor import DocumentProcessor
from src.core.models import PageInfo, BoundingBox


class TestDocumentProcessor:
    """Test suite for DocumentProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a document processor instance."""
        return DocumentProcessor(
            enable_ocr=True,
            ocr_languages=["en"],
            page_batch_size=4
        )
    
    def test_initialization(self):
        """Test processor initialization with various configurations."""
        # Default initialization
        processor = DocumentProcessor()
        assert processor.enable_ocr is True
        assert processor.ocr_languages == ["en"]
        assert processor.page_batch_size == 4
        assert processor.max_memory_mb == 4096
        
        # Custom initialization
        processor = DocumentProcessor(
            enable_ocr=False,
            ocr_languages=["en", "es"],
            page_batch_size=8,
            max_memory_mb=8192
        )
        assert processor.enable_ocr is False
        assert processor.ocr_languages == ["en", "es"]
        assert processor.page_batch_size == 8
        assert processor.max_memory_mb == 8192
    
    def test_create_pipeline_options(self, processor):
        """Test pipeline options creation."""
        options = processor._create_pipeline_options()
        
        # Check that options object is created
        assert options is not None
        assert hasattr(options, 'do_ocr')
        assert options.do_ocr == processor.enable_ocr
    
    @patch('src.core.document_processor.DocumentConverter')
    def test_process_document(self, mock_converter_class, processor):
        """Test processing a document from file path."""
        # Create mock converter
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        
        # Create mock result
        mock_result = Mock()
        mock_doc = Mock()
        mock_doc.num_pages = 2
        # Mock iterate_items to return (item, level) tuples
        item1 = Mock()
        item1.export_to_text.return_value = "Page 1 text"
        item1.bounding_box = Mock(x0=0, y0=0, x1=100, y1=50)
        item1.__class__.__name__ = 'TextItem'
        
        item2 = Mock()
        item2.export_to_text.return_value = "Page 2 text"
        item2.bounding_box = Mock(x0=0, y0=0, x1=100, y1=50)
        item2.__class__.__name__ = 'TextItem'
        
        mock_doc.iterate_items.side_effect = [
            # Page 1 items (returns list of (item, level) tuples)
            [(item1, 0)],
            # Page 2 items
            [(item2, 0)]
        ]
        mock_result.document = mock_doc
        mock_result.pages = [Mock(), Mock()]  # Add pages for metadata extraction
        mock_converter.convert.return_value = mock_result
        
        # Reinitialize processor with mocked converter
        processor.converter = mock_converter
        
        # Process document
        file_path = Path("/tmp/test.pdf")
        pages = list(processor.process_document(file_path))
        
        # Verify
        assert len(pages) == 2
        assert pages[0].page_number == 1
        assert pages[0].text_content == "Page 1 text"
        assert pages[1].page_number == 2
        assert pages[1].text_content == "Page 2 text"
        
        # Verify converter was called correctly
        mock_converter.convert.assert_called_once_with(
            source=str(file_path),
            page_range=None
        )
    
    @patch('src.core.document_processor.DocumentConverter')
    def test_process_document_with_error(self, mock_converter_class, processor):
        """Test error handling in document processing."""
        # Setup mock to raise exception
        mock_converter = Mock()
        mock_converter.convert.side_effect = Exception("Conversion failed")
        mock_converter_class.return_value = mock_converter
        processor.converter = mock_converter
        
        # Test that exception is raised
        file_path = Path("/tmp/test.pdf")
        with pytest.raises(Exception, match="Conversion failed"):
            list(processor.process_document(file_path))
    
    def test_extract_page_info_with_empty_page(self, processor):
        """Test extracting info from empty page."""
        # Create mock document with no items
        mock_doc = Mock()
        mock_doc.num_pages = 1
        mock_doc.iterate_items.return_value = []
        
        # Create mock result
        mock_result = Mock()
        mock_result.pages = [Mock()]  # Add pages
        
        # Extract page info
        page_info = processor._extract_page_info(mock_doc, 1, mock_result)
        
        # Verify
        assert page_info is not None
        assert page_info.page_number == 1
        assert page_info.text_content == ""
        assert len(page_info.text_blocks) == 0
        assert len(page_info.images) == 0
        assert len(page_info.tables) == 0
    
    @patch('src.core.document_processor.DocumentConverter')
    def test_get_document_metadata(self, mock_converter_class, processor):
        """Test getting document metadata."""
        # Setup mock
        mock_converter = Mock()
        mock_result = Mock()
        mock_doc = Mock()
        mock_doc.num_pages = 3
        mock_result.document = mock_doc
        
        # Mock metadata - try both locations
        mock_result.input = Mock()
        mock_result.input.metadata = {
            'title': 'Test Document',
            'author': 'Test Author',
            'subject': 'Test Subject',
            'creation_date': '2024-01-01',
            'modification_date': '2024-01-02'
        }
        # Also add to document for backward compatibility
        mock_doc.metadata = mock_result.input.metadata
        
        mock_converter.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter
        processor.converter = mock_converter
        
        # Get metadata
        file_path = Path("/tmp/test.pdf")
        metadata = processor.get_document_metadata(file_path)
        
        # Verify
        assert metadata['total_pages'] == 3
        assert metadata['title'] == 'Test Document'
        assert metadata['author'] == 'Test Author'
        assert metadata['subject'] == 'Test Subject'
        assert metadata['creation_date'] == '2024-01-01'
        assert metadata['modification_date'] == '2024-01-02'