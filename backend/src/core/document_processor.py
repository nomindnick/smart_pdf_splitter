"""
Document processor module using Docling for PDF parsing and text extraction.

This module provides memory-efficient PDF processing capabilities with page-by-page
analysis, text extraction, and metadata extraction for boundary detection.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator, Tuple
from io import BytesIO

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import (
    InputFormat,
    DocumentStream
)
from docling.datamodel.document import ConversionResult
from docling_core.types.doc.document import DoclingDocument
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractOcrOptions
)
from docling.utils.utils import chunkify

from .models import PageInfo, BoundingBox, ProcessingStatus

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Handles PDF document processing using Docling library.
    
    Provides memory-efficient page-by-page processing with text extraction,
    metadata extraction, and structure analysis for boundary detection.
    """
    
    def __init__(
        self,
        enable_ocr: bool = True,
        ocr_languages: List[str] = None,
        page_batch_size: int = 4,
        max_memory_mb: int = 4096  # 4GB default limit
    ):
        """
        Initialize document processor with configuration.
        
        Args:
            enable_ocr: Whether to enable OCR for scanned documents
            ocr_languages: List of languages for OCR (default: ["en"])
            page_batch_size: Number of pages to process in batch
            max_memory_mb: Maximum memory usage in MB
        """
        self.enable_ocr = enable_ocr
        self.ocr_languages = ocr_languages or ["en"]
        self.page_batch_size = page_batch_size
        self.max_memory_mb = max_memory_mb
        
        # Configure pipeline options
        self.pipeline_options = self._create_pipeline_options()
        
        # Initialize converter
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options
                )
            }
        )
        
    def _create_pipeline_options(self) -> PdfPipelineOptions:
        """Create pipeline options with memory-efficient settings."""
        options = PdfPipelineOptions()
        
        # OCR configuration
        if self.enable_ocr:
            options.do_ocr = True
            options.ocr_options = TesseractOcrOptions(
                kind='tesserocr',
                lang=self.ocr_languages
            )
        else:
            options.do_ocr = False
        
        # Configure for memory efficiency
        options.generate_page_images = False  # Don't generate images to save memory
        options.generate_picture_images = False
        options.generate_table_images = False
            
        return options
    
    def process_document(
        self,
        file_path: Path,
        page_range: Optional[Tuple[int, int]] = None
    ) -> Iterator[PageInfo]:
        """
        Process a PDF document and yield page information.
        
        Args:
            file_path: Path to PDF file
            page_range: Optional tuple of (start_page, end_page) 1-indexed
            
        Yields:
            PageInfo objects for each processed page
        """
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Convert document with optional page range
            if page_range:
                # TODO: Check if installed version supports page_range
                logger.warning("page_range parameter may not be supported in this version of Docling")
            
            result = self.converter.convert(
                source=str(file_path)
            )
            
            # Extract pages from result
            yield from self._extract_pages_from_result(result)
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    def process_document_stream(
        self,
        pdf_bytes: bytes,
        filename: str = "document.pdf"
    ) -> Iterator[PageInfo]:
        """
        Process a PDF from bytes stream for memory efficiency.
        
        Args:
            pdf_bytes: PDF file content as bytes
            filename: Optional filename for the document
            
        Yields:
            PageInfo objects for each processed page
        """
        logger.info(f"Processing document stream: {filename}")
        
        try:
            # Create stream from bytes
            stream = BytesIO(pdf_bytes)
            source = DocumentStream(name=filename, stream=stream)
            
            # Convert document from stream
            result = self.converter.convert(source)
            
            # Extract pages from result
            yield from self._extract_pages_from_result(result)
            
        except Exception as e:
            logger.error(f"Error processing document stream: {e}")
            raise
    
    def _extract_pages_from_result(
        self,
        result: ConversionResult
    ) -> Iterator[PageInfo]:
        """Extract page information from conversion result."""
        if not result.document:
            logger.warning("No document found in conversion result")
            return
            
        doc: DoclingDocument = result.document
        
        # Process pages in batches for memory efficiency
        total_pages = doc.num_pages if hasattr(doc, 'num_pages') else len(result.pages)
        page_numbers = list(range(1, total_pages + 1))
        
        for page_batch in chunkify(page_numbers, self.page_batch_size):
            for page_num in page_batch:
                try:
                    page_info = self._extract_page_info(doc, page_num, result)
                    if page_info:
                        yield page_info
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    continue
    
    def _extract_page_info(
        self,
        doc: DoclingDocument,
        page_num: int,
        result: ConversionResult
    ) -> Optional[PageInfo]:
        """
        Extract information from a single page.
        
        Args:
            doc: Docling document object
            page_num: Page number (1-indexed)
            
        Returns:
            PageInfo object with extracted data
        """
        try:
            # Get page content
            page_idx = page_num - 1
            
            # Extract text content from page
            page_text = ""
            text_blocks = []
            images = []
            tables = []
            
            # Iterate through document items on this page
            for item, level in doc.iterate_items(page_no=page_num):
                # Get text content
                text = item.export_to_text() if hasattr(item, 'export_to_text') else str(item.text if hasattr(item, 'text') else '')
                if text:
                    page_text += text + "\n"
                    
                    # Create text block with bounding box if available
                    if hasattr(item, 'bounding_box') and item.bounding_box:
                        bbox = BoundingBox(
                            x=item.bounding_box.x0,
                            y=item.bounding_box.y0,
                            width=item.bounding_box.x1 - item.bounding_box.x0,
                            height=item.bounding_box.y1 - item.bounding_box.y0
                        )
                        text_blocks.append({
                            'text': text,
                            'bbox': bbox,
                            'type': type(item).__name__.lower()
                        })
                
                # Check for specific item types
                item_type = type(item).__name__.lower()
                if 'picture' in item_type and hasattr(item, 'bounding_box') and item.bounding_box:
                    images.append({
                        'bbox': BoundingBox(
                            x=item.bounding_box.x0,
                            y=item.bounding_box.y0,
                            width=item.bounding_box.x1 - item.bounding_box.x0,
                            height=item.bounding_box.y1 - item.bounding_box.y0
                        )
                    })
                elif 'table' in item_type:
                    table_data = {
                        'text': text,
                    }
                    if hasattr(item, 'bounding_box') and item.bounding_box:
                        table_data['bbox'] = BoundingBox(
                            x=item.bounding_box.x0,
                            y=item.bounding_box.y0,
                            width=item.bounding_box.x1 - item.bounding_box.x0,
                            height=item.bounding_box.y1 - item.bounding_box.y0
                        )
                    tables.append(table_data)
            
            # Get page metadata from result
            page_metadata = {}
            if hasattr(result, 'pages') and result.pages and page_idx < len(result.pages):
                page = result.pages[page_idx]
                if hasattr(page, 'size'):
                    page_metadata['width'] = page.size.width
                    page_metadata['height'] = page.size.height
                elif hasattr(page, 'width') and hasattr(page, 'height'):
                    page_metadata['width'] = page.width
                    page_metadata['height'] = page.height
            
            # Get page dimensions with defaults
            width = page_metadata.get('width', 612.0)  # Default US Letter width
            height = page_metadata.get('height', 792.0)  # Default US Letter height
            
            # Count words
            word_count = len(page_text.split()) if page_text else 0
            
            return PageInfo(
                page_number=page_num,
                width=width,
                height=height,
                text_content=page_text.strip(),
                word_count=word_count,
                has_images=len(images) > 0,
                has_tables=len(tables) > 0,
                layout_elements=text_blocks + images + tables
            )
            
        except Exception as e:
            logger.error(f"Error extracting info from page {page_num}: {e}")
            return None
    
    def extract_text_from_region(
        self,
        file_path: Path,
        page_num: int,
        bbox: BoundingBox
    ) -> Optional[str]:
        """
        Extract text from a specific region of a page.
        
        Args:
            file_path: Path to PDF file
            page_num: Page number (1-indexed)
            bbox: Bounding box defining the region
            
        Returns:
            Extracted text or None if error
        """
        try:
            # Process just the specific page
            # TODO: Use page_range when supported
            result = self.converter.convert(
                source=str(file_path)
            )
            
            if not result.document:
                return None
                
            doc = result.document
            extracted_text = ""
            
            # Find text elements within the bounding box
            for item, level in doc.iterate_items(page_no=page_num):
                if hasattr(item, 'bounding_box') and item.bounding_box:
                    elem_bbox = item.bounding_box
                    # Check if element is within the specified region
                    # Convert BoundingBox format (x, y, width, height) to compare with element bbox
                    bbox_x1 = bbox.x + bbox.width
                    bbox_y1 = bbox.y + bbox.height
                    
                    if (elem_bbox.x0 >= bbox.x and elem_bbox.y0 >= bbox.y and
                        elem_bbox.x1 <= bbox_x1 and elem_bbox.y1 <= bbox_y1):
                        text = item.export_to_text() if hasattr(item, 'export_to_text') else str(item.text if hasattr(item, 'text') else '')
                        if text:
                            extracted_text += text + "\n"
            
            return extracted_text.strip() if extracted_text else None
            
        except Exception as e:
            logger.error(f"Error extracting text from region: {e}")
            return None
    
    def get_document_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Get document-level metadata.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with document metadata
        """
        try:
            result = self.converter.convert(
                source=str(file_path)
            )
            
            metadata = {
                'total_pages': 0,
                'title': None,
                'author': None,
                'subject': None,
                'creation_date': None,
                'modification_date': None
            }
            
            if result.document:
                doc = result.document
                metadata['total_pages'] = doc.num_pages if hasattr(doc, 'num_pages') else len(result.pages)
                
                # Extract PDF metadata if available from result
                if hasattr(result, 'input') and hasattr(result.input, 'metadata'):
                    meta = result.input.metadata
                    metadata['title'] = getattr(meta, 'title', None)
                    metadata['author'] = getattr(meta, 'author', None)
                    metadata['subject'] = getattr(meta, 'subject', None)
                    metadata['creation_date'] = getattr(meta, 'creation_date', None)
                    metadata['modification_date'] = getattr(meta, 'modification_date', None)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting document metadata: {e}")
            return {}