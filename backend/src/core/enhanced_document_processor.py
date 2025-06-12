"""
Enhanced document processor with visual feature extraction capabilities.

This module extends the base document processor to include visual analysis
using Docling's advanced features.
"""

import logging
from typing import List, Optional, Iterator, Tuple, Dict, Any
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .document_processor import DocumentProcessor
from .visual_processor import VisualFeatureProcessor
from .models import PageInfo, PageVisualInfo, ProcessingStatus

logger = logging.getLogger(__name__)


class EnhancedDocumentProcessor(DocumentProcessor):
    """
    Enhanced document processor with visual feature extraction capabilities.
    """
    
    def __init__(
        self,
        enable_ocr: bool = True,
        enable_visual_features: bool = True,
        enable_vlm: bool = True,
        visual_memory_limit_mb: int = 2048,
        max_parallel_pages: int = 2,
        **kwargs
    ):
        """
        Initialize enhanced processor.
        
        Args:
            enable_ocr: Enable OCR processing
            enable_visual_features: Enable visual feature extraction
            enable_vlm: Enable Vision Language Model
            visual_memory_limit_mb: Memory limit for visual processing
            max_parallel_pages: Maximum pages to process in parallel
            **kwargs: Additional arguments for base processor
        """
        super().__init__(enable_ocr=enable_ocr, **kwargs)
        
        self.enable_visual_features = enable_visual_features
        self.enable_vlm = enable_vlm
        self.visual_memory_limit_mb = visual_memory_limit_mb
        self.max_parallel_pages = max_parallel_pages
        
        # Initialize visual processor if enabled
        if self.enable_visual_features:
            self.visual_processor = VisualFeatureProcessor(
                enable_vlm=enable_vlm,
                memory_limit_mb=visual_memory_limit_mb
            )
        else:
            self.visual_processor = None
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_parallel_pages)
    
    def process_document_with_visual(
        self,
        file_path: Path,
        page_range: Optional[Tuple[int, int]] = None
    ) -> Iterator[PageVisualInfo]:
        """
        Process document with visual feature extraction.
        
        Args:
            file_path: Path to PDF file
            page_range: Optional page range to process
            
        Yields:
            PageVisualInfo objects with visual features
        """
        logger.info(f"Processing document with visual features: {file_path}")
        
        # First pass: Extract basic page info
        pages = list(self.process_document(file_path, page_range))
        
        if not self.enable_visual_features:
            # Convert to PageVisualInfo without visual features
            for page in pages:
                yield PageVisualInfo(**page.dict())
            return
        
        # Second pass: Extract visual features in batches
        for i in range(0, len(pages), self.page_batch_size):
            batch = pages[i:i + self.page_batch_size]
            
            # Process batch in parallel
            visual_pages = self._process_visual_batch(file_path, batch)
            
            for visual_page in visual_pages:
                yield visual_page
    
    def _process_visual_batch(
        self,
        file_path: Path,
        pages: List[PageInfo]
    ) -> List[PageVisualInfo]:
        """Process a batch of pages for visual features."""
        visual_pages = []
        
        try:
            # Extract visual features for each page
            futures = []
            
            for page in pages:
                future = self.executor.submit(
                    self._extract_visual_features_for_page,
                    file_path,
                    page
                )
                futures.append(future)
            
            # Collect results
            for future, page in zip(futures, pages):
                try:
                    visual_features, picture_classifications = future.result(timeout=30)
                    visual_page = PageVisualInfo(
                        **page.dict(),
                        visual_features=visual_features,
                        picture_classifications=picture_classifications or {}
                    )
                    visual_pages.append(visual_page)
                except Exception as e:
                    logger.error(f"Error extracting visual features for page {page.page_number}: {e}")
                    # Add page without visual features
                    visual_pages.append(PageVisualInfo(**page.dict()))
        
        except Exception as e:
            logger.error(f"Error processing visual batch: {e}")
            # Return pages without visual features
            visual_pages = [PageVisualInfo(**p.dict()) for p in pages]
        
        return visual_pages
    
    def _extract_visual_features_for_page(
        self,
        file_path: Path,
        page: PageInfo
    ):
        """Extract visual features for a single page."""
        try:
            # Convert with visual pipeline for this specific page
            result = self.visual_processor.converter.convert(
                source=str(file_path)
            )
            
            if result.document:
                # Extract visual features
                features = self.visual_processor.extract_visual_features(
                    result.document,
                    page.page_number
                )
                
                # Extract picture classifications if available
                classifications = self.visual_processor.extract_picture_classifications(
                    result.document,
                    page.page_number
                )
                
                return features, classifications
            
            return None, None
        
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            return None, None
    
    def process_document_stream_with_visual(
        self,
        pdf_bytes: bytes,
        filename: str = "document.pdf"
    ) -> Iterator[PageVisualInfo]:
        """
        Process a PDF from bytes stream with visual features.
        
        Args:
            pdf_bytes: PDF file content as bytes
            filename: Optional filename for the document
            
        Yields:
            PageVisualInfo objects with visual features
        """
        logger.info(f"Processing document stream with visual features: {filename}")
        
        # First pass: Extract basic page info
        pages = list(self.process_document_stream(pdf_bytes, filename))
        
        if not self.enable_visual_features:
            # Convert to PageVisualInfo without visual features
            for page in pages:
                yield PageVisualInfo(**page.dict())
            return
        
        # For stream processing, we need to save temporarily for visual processing
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = Path(tmp_file.name)
        
        try:
            # Process with visual features using temp file
            for i in range(0, len(pages), self.page_batch_size):
                batch = pages[i:i + self.page_batch_size]
                visual_pages = self._process_visual_batch(tmp_path, batch)
                
                for visual_page in visual_pages:
                    yield visual_page
        finally:
            # Clean up temp file
            if tmp_path.exists():
                os.unlink(tmp_path)
    
    def get_visual_summary(
        self,
        pages: List[PageVisualInfo]
    ) -> Dict[str, Any]:
        """
        Get a summary of visual features across all pages.
        
        Args:
            pages: List of pages with visual info
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_pages': len(pages),
            'pages_with_visual_features': 0,
            'common_font_sizes': {},
            'common_font_families': {},
            'layout_types': {'single_column': 0, 'multi_column': 0},
            'pages_with_images': 0,
            'pages_with_tables': 0,
            'pages_with_logos': 0,
            'orientation_changes': 0,
            'significant_layout_changes': []
        }
        
        prev_orientation = None
        prev_columns = None
        
        for i, page in enumerate(pages):
            if not page.visual_features:
                continue
            
            summary['pages_with_visual_features'] += 1
            features = page.visual_features
            
            # Count font usage
            if features.primary_font_size:
                size_key = f"{features.primary_font_size:.0f}pt"
                summary['common_font_sizes'][size_key] = summary['common_font_sizes'].get(size_key, 0) + 1
            
            if features.primary_font_family:
                summary['common_font_families'][features.primary_font_family] = \
                    summary['common_font_families'].get(features.primary_font_family, 0) + 1
            
            # Count layout types
            if features.num_columns > 1:
                summary['layout_types']['multi_column'] += 1
            else:
                summary['layout_types']['single_column'] += 1
            
            # Count visual elements
            if features.num_images > 0:
                summary['pages_with_images'] += 1
            if features.num_tables > 0:
                summary['pages_with_tables'] += 1
            if features.has_logo:
                summary['pages_with_logos'] += 1
            
            # Track orientation changes
            if prev_orientation and prev_orientation != features.orientation:
                summary['orientation_changes'] += 1
                summary['significant_layout_changes'].append({
                    'page': page.page_number,
                    'type': 'orientation_change',
                    'from': prev_orientation,
                    'to': features.orientation
                })
            prev_orientation = features.orientation
            
            # Track column changes
            if prev_columns and prev_columns != features.num_columns:
                summary['significant_layout_changes'].append({
                    'page': page.page_number,
                    'type': 'column_change',
                    'from': prev_columns,
                    'to': features.num_columns
                })
            prev_columns = features.num_columns
        
        return summary
    
    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)