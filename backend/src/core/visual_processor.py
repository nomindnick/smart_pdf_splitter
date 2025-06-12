"""
Visual feature processor module using Docling's advanced capabilities.

This module provides visual analysis features including layout detection,
picture classification, and visual boundary identification.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
from io import BytesIO

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureClassificationOptions,
    TableDetectionOptions
)
from docling.datamodel.base_models import InputFormat, PictureClassificationModel
from docling_core.types.doc.document import DoclingDocument

from .models import VisualFeatures, PageVisualInfo, BoundingBox

logger = logging.getLogger(__name__)


class VisualFeatureProcessor:
    """
    Processes visual features from PDF pages using Docling's advanced capabilities.
    """
    
    def __init__(
        self,
        enable_picture_classification: bool = True,
        enable_vlm: bool = True,
        enable_table_detection: bool = True,
        max_image_size: int = 1024,  # Max dimension for images
        memory_limit_mb: int = 2048  # 2GB for visual processing
    ):
        """
        Initialize visual feature processor.
        
        Args:
            enable_picture_classification: Enable Docling picture classification
            enable_vlm: Enable Vision Language Model features
            enable_table_detection: Enable table detection
            max_image_size: Maximum image dimension to process
            memory_limit_mb: Memory limit for visual processing
        """
        self.enable_picture_classification = enable_picture_classification
        self.enable_vlm = enable_vlm
        self.enable_table_detection = enable_table_detection
        self.max_image_size = max_image_size
        self.memory_limit_mb = memory_limit_mb
        
        # Configure pipeline for visual processing
        self.pipeline_options = self._create_visual_pipeline_options()
        
        # Initialize converter with visual features
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options
                )
            }
        )
    
    def _create_visual_pipeline_options(self) -> PdfPipelineOptions:
        """Create pipeline options for visual feature extraction."""
        options = PdfPipelineOptions()
        
        # Enable visual feature extraction
        options.generate_page_images = True
        options.generate_picture_images = self.enable_picture_classification
        options.generate_table_images = self.enable_table_detection
        
        # Configure picture classification if available
        # Note: This configuration may need adjustment based on installed Docling version
        if self.enable_picture_classification:
            try:
                options.picture_classification_options = PictureClassificationOptions(
                    enabled=True,
                    confidence_threshold=0.7
                )
            except Exception as e:
                logger.warning(f"Picture classification options not available: {e}")
                self.enable_picture_classification = False
        
        # Configure table detection
        if self.enable_table_detection:
            try:
                options.table_detection_options = TableDetectionOptions(
                    enabled=True,
                    detect_structure=True
                )
            except Exception as e:
                logger.warning(f"Table detection options not available: {e}")
                self.enable_table_detection = False
        
        # Set image generation parameters for memory efficiency
        options.page_image_resolution = min(150, self.max_image_size // 10)
        
        return options
    
    def extract_visual_features(
        self,
        page: DoclingDocument,
        page_num: int,
        page_image: Optional[Image.Image] = None
    ) -> VisualFeatures:
        """
        Extract visual features from a page.
        
        Args:
            page: Docling document page
            page_num: Page number (1-indexed)
            page_image: Optional pre-loaded page image
            
        Returns:
            VisualFeatures object
        """
        features = VisualFeatures()
        
        try:
            # Extract layout structure
            layout_info = self._analyze_layout_structure(page, page_num)
            features.num_columns = layout_info.get('num_columns', 1)
            features.text_alignment = layout_info.get('alignment', 'left')
            
            # Extract font information
            font_info = self._analyze_fonts(page, page_num)
            features.primary_font_size = font_info.get('primary_size')
            features.primary_font_family = font_info.get('primary_family')
            
            # Extract visual elements
            elements = self._count_visual_elements(page, page_num)
            features.num_images = elements.get('images', 0)
            features.num_tables = elements.get('tables', 0)
            features.num_charts = elements.get('charts', 0)
            
            # Analyze spacing and margins
            if page_image:
                spacing_info = self._analyze_spacing(page_image)
                features.avg_line_spacing = spacing_info.get('line_spacing')
                features.margin_top = spacing_info.get('margin_top')
                features.margin_bottom = spacing_info.get('margin_bottom')
                features.margin_left = spacing_info.get('margin_left')
                features.margin_right = spacing_info.get('margin_right')
            
            # Detect headers and footers
            header_footer = self._detect_header_footer(page, page_num)
            features.has_header = header_footer.get('has_header', False)
            features.has_footer = header_footer.get('has_footer', False)
            features.header_text = header_footer.get('header_text')
            features.footer_text = header_footer.get('footer_text')
            
            # Page orientation
            if hasattr(page, 'size'):
                features.aspect_ratio = page.size.width / page.size.height
                features.orientation = "landscape" if features.aspect_ratio > 1.2 else "portrait"
            
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
        
        return features
    
    def _analyze_layout_structure(
        self,
        page: DoclingDocument,
        page_num: int
    ) -> Dict[str, Any]:
        """Analyze the layout structure of a page."""
        layout_info = {
            'num_columns': 1,
            'alignment': 'left',
            'has_multi_column': False
        }
        
        try:
            # Get all text blocks with positions
            text_blocks = []
            for item, level in page.iterate_items(page_no=page_num):
                if hasattr(item, 'bounding_box') and item.bounding_box:
                    text_blocks.append({
                        'bbox': item.bounding_box,
                        'text': str(item.text) if hasattr(item, 'text') else '',
                        'level': level
                    })
            
            if not text_blocks:
                return layout_info
            
            # Analyze column structure by x-coordinate clustering
            x_positions = [block['bbox'].x0 for block in text_blocks]
            if len(set(x_positions)) > 1:
                # Simple column detection based on x-position clustering
                x_clusters = self._cluster_positions(x_positions, threshold=50)
                layout_info['num_columns'] = len(x_clusters)
                layout_info['has_multi_column'] = len(x_clusters) > 1
            
            # Detect text alignment
            layout_info['alignment'] = self._detect_alignment(text_blocks)
            
        except Exception as e:
            logger.error(f"Error analyzing layout structure: {e}")
        
        return layout_info
    
    def _analyze_fonts(
        self,
        page: DoclingDocument,
        page_num: int
    ) -> Dict[str, Any]:
        """Analyze font characteristics on the page."""
        font_info = {}
        
        try:
            # Collect font information from text elements
            font_sizes = []
            font_families = []
            
            for item, _ in page.iterate_items(page_no=page_num):
                # Extract font info if available in item metadata
                if hasattr(item, 'style') and item.style:
                    if hasattr(item.style, 'font_size'):
                        font_sizes.append(item.style.font_size)
                    if hasattr(item.style, 'font_family'):
                        font_families.append(item.style.font_family)
            
            # Determine primary font characteristics
            if font_sizes:
                # Most common font size
                font_info['primary_size'] = max(set(font_sizes), key=font_sizes.count)
            
            if font_families:
                # Most common font family
                font_info['primary_family'] = max(set(font_families), key=font_families.count)
            
        except Exception as e:
            logger.error(f"Error analyzing fonts: {e}")
        
        return font_info
    
    def _count_visual_elements(
        self,
        page: DoclingDocument,
        page_num: int
    ) -> Dict[str, int]:
        """Count different types of visual elements on the page."""
        counts = {
            'images': 0,
            'tables': 0,
            'charts': 0
        }
        
        try:
            for item, _ in page.iterate_items(page_no=page_num):
                item_type = type(item).__name__.lower()
                
                if 'picture' in item_type or 'image' in item_type:
                    counts['images'] += 1
                    # Check if it's a chart based on classification
                    if hasattr(item, 'classification'):
                        if 'chart' in str(item.classification).lower():
                            counts['charts'] += 1
                elif 'table' in item_type:
                    counts['tables'] += 1
        
        except Exception as e:
            logger.error(f"Error counting visual elements: {e}")
        
        return counts
    
    def _analyze_spacing(self, page_image: Image.Image) -> Dict[str, float]:
        """Analyze spacing and margins from page image."""
        spacing_info = {}
        
        try:
            # Convert to numpy array for analysis
            img_array = np.array(page_image.convert('L'))  # Convert to grayscale
            height, width = img_array.shape
            
            # Simple margin detection by finding content boundaries
            # Find first and last non-white rows/columns
            threshold = 250  # Near white
            
            # Top margin
            for i in range(height):
                if np.any(img_array[i] < threshold):
                    spacing_info['margin_top'] = i / height
                    break
            
            # Bottom margin
            for i in range(height - 1, -1, -1):
                if np.any(img_array[i] < threshold):
                    spacing_info['margin_bottom'] = (height - i) / height
                    break
            
            # Left margin
            for i in range(width):
                if np.any(img_array[:, i] < threshold):
                    spacing_info['margin_left'] = i / width
                    break
            
            # Right margin
            for i in range(width - 1, -1, -1):
                if np.any(img_array[:, i] < threshold):
                    spacing_info['margin_right'] = (width - i) / width
                    break
            
            # Estimate line spacing (simplified)
            # This would need more sophisticated text line detection
            spacing_info['line_spacing'] = 1.2  # Default estimate
            
        except Exception as e:
            logger.error(f"Error analyzing spacing: {e}")
        
        return spacing_info
    
    def _detect_header_footer(
        self,
        page: DoclingDocument,
        page_num: int
    ) -> Dict[str, Any]:
        """Detect headers and footers on the page."""
        info = {
            'has_header': False,
            'has_footer': False,
            'header_text': None,
            'footer_text': None
        }
        
        try:
            # Get page height for position calculation
            page_height = 792  # Default US Letter
            if hasattr(page, 'size'):
                page_height = page.size.height
            
            # Collect text elements by vertical position
            top_elements = []
            bottom_elements = []
            
            for item, _ in page.iterate_items(page_no=page_num):
                if hasattr(item, 'bounding_box') and item.bounding_box:
                    bbox = item.bounding_box
                    text = str(item.text) if hasattr(item, 'text') else ''
                    
                    # Top 10% of page
                    if bbox.y0 < page_height * 0.1:
                        top_elements.append(text)
                    
                    # Bottom 10% of page
                    if bbox.y1 > page_height * 0.9:
                        bottom_elements.append(text)
            
            # Detect headers
            if top_elements:
                info['has_header'] = True
                info['header_text'] = ' '.join(top_elements).strip()
            
            # Detect footers
            if bottom_elements:
                info['has_footer'] = True
                info['footer_text'] = ' '.join(bottom_elements).strip()
        
        except Exception as e:
            logger.error(f"Error detecting header/footer: {e}")
        
        return info
    
    def _cluster_positions(
        self,
        positions: List[float],
        threshold: float = 50
    ) -> List[List[float]]:
        """Simple clustering of positions."""
        if not positions:
            return []
        
        sorted_pos = sorted(positions)
        clusters = [[sorted_pos[0]]]
        
        for pos in sorted_pos[1:]:
            if pos - clusters[-1][-1] <= threshold:
                clusters[-1].append(pos)
            else:
                clusters.append([pos])
        
        return clusters
    
    def _detect_alignment(self, text_blocks: List[Dict]) -> str:
        """Detect predominant text alignment."""
        if not text_blocks:
            return 'left'
        
        # Simple heuristic based on x-positions
        # This would need more sophisticated analysis in production
        return 'left'
    
    def extract_picture_classifications(
        self,
        page: DoclingDocument,
        page_num: int
    ) -> Dict[str, float]:
        """
        Extract picture classifications for a page.
        
        Args:
            page: Docling document page
            page_num: Page number
            
        Returns:
            Dictionary of classification labels and confidence scores
        """
        classifications = {}
        
        try:
            # This would integrate with Docling's picture classification
            # For now, returning placeholder
            logger.debug(f"Picture classification for page {page_num}")
        except Exception as e:
            logger.error(f"Error extracting picture classifications: {e}")
        
        return classifications
    
    def analyze_with_vlm(
        self,
        page_image: Image.Image,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Analyze page with Vision Language Model.
        
        Args:
            page_image: Page image
            prompt: Analysis prompt
            
        Returns:
            VLM analysis results
        """
        results = {}
        
        try:
            # This would integrate with Docling's VLM capabilities
            # For now, returning placeholder
            logger.debug("VLM analysis requested")
        except Exception as e:
            logger.error(f"Error in VLM analysis: {e}")
        
        return results