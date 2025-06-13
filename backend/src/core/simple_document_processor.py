"""
Simple document processor using PyMuPDF for testing purposes.

This processor extracts text using PyMuPDF's layout-preserving text extraction,
which can work better for scanned PDFs that have been pre-processed.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Iterator, Optional, Tuple
import logging

from .models import PageInfo

logger = logging.getLogger(__name__)


class SimpleDocumentProcessor:
    """Simple document processor using PyMuPDF."""
    
    def __init__(self):
        """Initialize the processor."""
        pass
    
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
        try:
            doc = fitz.open(str(file_path))
            
            # Determine pages to process
            start_page = 1 if page_range is None else page_range[0]
            end_page = len(doc) if page_range is None else min(page_range[1], len(doc))
            
            for page_num in range(start_page, end_page + 1):
                page_idx = page_num - 1
                page = doc[page_idx]
                
                # Extract text with layout preservation
                text = page.get_text("text")
                
                # If no text found, try blocks method
                if not text.strip():
                    blocks = page.get_text("blocks")
                    text_blocks = []
                    for block in blocks:
                        if block[6] == 0:  # Text block (not image)
                            text_blocks.append(block[4])
                    text = "\n".join(text_blocks)
                
                # Get page dimensions
                rect = page.rect
                width = rect.width
                height = rect.height
                
                # Count words
                word_count = len(text.split()) if text else 0
                
                # Check for images and tables (basic detection)
                has_images = len(list(page.get_images())) > 0
                
                # Simple table detection - look for tab-separated content
                lines = text.split('\n')
                has_tables = any('\t' in line or '  ' in line for line in lines)
                
                yield PageInfo(
                    page_number=page_num,
                    width=width,
                    height=height,
                    text_content=text,
                    word_count=word_count,
                    has_images=has_images,
                    has_tables=has_tables
                )
                
            doc.close()
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    def get_document_metadata(self, file_path: Path) -> dict:
        """Get basic metadata about the document."""
        try:
            doc = fitz.open(str(file_path))
            metadata = {
                'total_pages': len(doc),
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'keywords': doc.metadata.get('keywords', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
            }
            doc.close()
            return metadata
        except Exception as e:
            logger.error(f"Error getting metadata: {e}")
            return {'total_pages': 0}