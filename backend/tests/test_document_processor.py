"""
Tests for the document processor module.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

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
    
    @pytest.fixture
    def mock_conversion_result(self):
        """Create a mock conversion result."""
        result = Mock()
        
        # Mock document
        doc = Mock()
        doc.pages = [Mock(), Mock()]  # 2 pages
        
        # Mock elements on page 1
        element1 = Mock()
        element1.page_num = 1
        element1.text = "This is page 1 text"
        element1.element_type = "text"
        element1.bbox = Mock(x0=0, y0=0, x1=100, y1=50)
        
        # Mock elements on page 2
        element2 = Mock()
        element2.page_num = 2
        element2.text = "This is page 2 text"
        element2.element_type = "text"
        element2.bbox = Mock(x0=0, y0=0, x1=100, y1=50)
        
        # Setup iterate_elements
        doc.iterate_elements.return_value = [element1, element2]
        
        result.document = doc
        return result
    
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
    
    @patch('src.core.document_processor.DocumentConverter')
    def test_process_document(self, mock_converter_class, processor, mock_conversion_result):
        """Test processing a document from file path."""
        # Setup mocks
        mock_converter = Mock()
        mock_converter.convert.return_value = mock_conversion_result
        mock_converter_class.return_value = mock_converter
        
        # Reinitialize processor to use mocked converter
        processor.converter = mock_converter
        
        # Process document
        file_path = Path("/tmp/test.pdf")
        pages = list(processor.process_document(file_path))
        
        # Verify
        assert len(pages) == 2
        assert pages[0].page_number == 1
        assert pages[0].text_content == "This is page 1 text"
        assert pages[1].page_number == 2
        assert pages[1].text_content == "This is page 2 text"
        
        # Verify converter was called correctly
        mock_converter.convert.assert_called_once_with(
            source=str(file_path),
            page_range=None
        )
    
    @patch('src.core.document_processor.DocumentConverter')
    def test_process_document_with_page_range(self, mock_converter_class, processor, mock_conversion_result):
        """Test processing specific page range."""
        # Setup mocks
        mock_converter = Mock()
        mock_converter.convert.return_value = mock_conversion_result
        mock_converter_class.return_value = mock_converter
        
        # Reinitialize processor to use mocked converter
        processor.converter = mock_converter
        
        # Process document with page range
        file_path = Path("/tmp/test.pdf")
        pages = list(processor.process_document(file_path, page_range=(1, 1)))
        
        # Verify converter was called with page range
        mock_converter.convert.assert_called_once_with(
            source=str(file_path),
            page_range=(1, 1)
        )
    
    @patch('src.core.document_processor.BytesIO')
    @patch('src.core.document_processor.DocumentStream')
    @patch('src.core.document_processor.DocumentConverter')
    def test_process_document_stream(self, mock_converter_class, mock_stream_class, mock_bytesio, processor, mock_conversion_result):
        """Test processing a document from bytes stream."""
        # Setup mocks
        mock_converter = Mock()
        mock_converter.convert.return_value = mock_conversion_result
        mock_converter_class.return_value = mock_converter
        
        mock_stream = Mock()
        mock_stream_class.return_value = mock_stream
        
        mock_bytes_stream = Mock()
        mock_bytesio.return_value = mock_bytes_stream
        
        # Reinitialize processor to use mocked converter
        processor.converter = mock_converter
        
        # Process document stream
        pdf_bytes = b"fake pdf content"
        pages = list(processor.process_document_stream(pdf_bytes, "test.pdf"))
        
        # Verify
        assert len(pages) == 2
        mock_bytesio.assert_called_once_with(pdf_bytes)
        mock_stream_class.assert_called_once_with(name="test.pdf", stream=mock_bytes_stream)
        mock_converter.convert.assert_called_once_with(mock_stream)
    
    def test_extract_page_info(self, processor):
        """Test extracting information from a single page."""
        # Create mock document
        doc = Mock()
        
        # Mock page
        page = Mock()
        page.size = Mock(width=612, height=792)
        doc.pages = [page]
        
        # Mock text element
        text_elem = Mock()
        text_elem.page_num = 1
        text_elem.text = "Sample text"
        text_elem.element_type = "text"
        text_elem.bbox = Mock(x0=10, y0=20, x1=100, y1=40)
        
        # Mock image element
        image_elem = Mock()
        image_elem.page_num = 1
        image_elem.element_type = "image"
        image_elem.bbox = Mock(x0=200, y0=300, x1=400, y1=500)
        
        # Mock table element
        table_elem = Mock()
        table_elem.page_num = 1
        table_elem.element_type = "table"
        table_elem.text = "Table data"
        table_elem.bbox = Mock(x0=50, y0=100, x1=500, y1=200)
        
        doc.iterate_elements.return_value = [text_elem, image_elem, table_elem]
        
        # Extract page info
        page_info = processor._extract_page_info(doc, 1)
        
        # Verify
        assert page_info is not None
        assert page_info.page_number == 1
        assert page_info.text_content == "Sample text\nTable data"
        assert len(page_info.text_blocks) == 2
        assert len(page_info.images) == 1
        assert len(page_info.tables) == 1
        assert page_info.metadata['width'] == 612
        assert page_info.metadata['height'] == 792
    
    @patch('src.core.document_processor.DocumentConverter')
    def test_extract_text_from_region(self, mock_converter_class, processor):
        """Test extracting text from a specific region."""
        # Setup mock conversion result
        result = Mock()
        doc = Mock()
        
        # Mock elements - one inside bbox, one outside
        elem_inside = Mock()
        elem_inside.page_num = 1
        elem_inside.text = "Inside region"
        elem_inside.bbox = Mock(x0=50, y0=50, x1=150, y1=100)
        
        elem_outside = Mock()
        elem_outside.page_num = 1
        elem_outside.text = "Outside region"
        elem_outside.bbox = Mock(x0=300, y0=300, x1=400, y1=400)
        
        doc.iterate_elements.return_value = [elem_inside, elem_outside]
        result.document = doc
        
        # Setup converter mock
        mock_converter = Mock()
        mock_converter.convert.return_value = result
        mock_converter_class.return_value = mock_converter
        processor.converter = mock_converter
        
        # Extract text from region
        file_path = Path("/tmp/test.pdf")
        bbox = BoundingBox(x0=0, y0=0, x1=200, y1=200)
        text = processor.extract_text_from_region(file_path, 1, bbox)
        
        # Verify
        assert text == "Inside region"
        mock_converter.convert.assert_called_once_with(
            source=str(file_path),
            page_range=(1, 1)
        )
    
    @patch('src.core.document_processor.DocumentConverter')
    def test_get_document_metadata(self, mock_converter_class, processor):
        """Test getting document metadata."""
        # Setup mock
        result = Mock()
        doc = Mock()
        doc.pages = [Mock(), Mock(), Mock()]  # 3 pages
        
        # Mock metadata
        metadata = Mock()
        metadata.title = "Test Document"
        metadata.author = "Test Author"
        metadata.subject = "Test Subject"
        metadata.creation_date = "2024-01-01"
        metadata.modification_date = "2024-01-02"
        doc.metadata = metadata
        
        result.document = doc
        
        mock_converter = Mock()
        mock_converter.convert.return_value = result
        mock_converter_class.return_value = mock_converter
        processor.converter = mock_converter
        
        # Get metadata
        file_path = Path("/tmp/test.pdf")
        metadata = processor.get_document_metadata(file_path)
        
        # Verify
        assert metadata['total_pages'] == 3
        assert metadata['title'] == "Test Document"
        assert metadata['author'] == "Test Author"
        assert metadata['subject'] == "Test Subject"
        assert metadata['creation_date'] == "2024-01-01"
        assert metadata['modification_date'] == "2024-01-02"
    
    @patch('src.core.document_processor.DocumentConverter')
    def test_error_handling(self, mock_converter_class, processor):
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
    
    def test_pipeline_options_creation(self, processor):
        """Test pipeline options are created correctly."""
        options = processor._create_pipeline_options()
        
        assert isinstance(options, Mock) or hasattr(options, 'do_ocr')
        if processor.enable_ocr:
            assert options.do_ocr is True
            assert options.ocr_options.lang == processor.ocr_languages