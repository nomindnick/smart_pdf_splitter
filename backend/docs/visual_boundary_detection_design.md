# Visual Boundary Detection Design with Docling

## Overview

This document outlines the design for enhancing the Smart PDF Splitter with advanced visual boundary detection using Docling's visual features, picture classification, and Vision Language Model (VLM) capabilities. The design focuses on complementing the existing LLM-based detection with visual analysis while maintaining memory efficiency.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Enhanced Document Processor                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │ Visual Feature  │  │ Picture          │  │    VLM     │ │
│  │   Extractor     │  │ Classifier       │  │  Analyzer  │ │
│  └────────┬────────┘  └────────┬─────────┘  └─────┬──────┘ │
│           │                     │                    │       │
│  ┌────────▼─────────────────────▼────────────────────▼────┐ │
│  │             Visual Boundary Detector                    │ │
│  │  - Layout Change Detection                             │ │
│  │  - Visual Separator Detection                          │ │
│  │  - Document Structure Analysis                         │ │
│  └────────────────────────┬───────────────────────────────┘ │
│                           │                                  │
│  ┌────────────────────────▼───────────────────────────────┐ │
│  │              Confidence Score Aggregator                │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Visual Feature Extractor

Extracts visual features from each page including:
- Page layout structure (columns, headers, footers)
- Font characteristics (size, style, family)
- Color information
- Spacing patterns
- Image and graphic positions

### 2. Picture Classifier

Uses Docling's picture classification to:
- Identify logos and letterheads
- Detect signatures
- Classify charts, graphs, and diagrams
- Recognize form elements

### 3. VLM Analyzer

Leverages Vision Language Models to:
- Understand page semantics from visual layout
- Detect document type from visual appearance
- Identify visual boundaries and separators
- Analyze document flow and structure

### 4. Visual Boundary Detector

Combines all visual signals to detect boundaries based on:
- Significant layout changes between pages
- Visual separators (lines, boxes, headers)
- Document structure transitions
- White space patterns

## Implementation Design

### Enhanced Models

```python
# backend/src/core/models.py additions

class VisualSignalType(str, Enum):
    """Types of visual signals for boundary detection."""
    
    LAYOUT_STRUCTURE_CHANGE = "layout_structure_change"
    FONT_STYLE_CHANGE = "font_style_change"
    COLOR_SCHEME_CHANGE = "color_scheme_change"
    VISUAL_SEPARATOR_LINE = "visual_separator_line"
    HEADER_FOOTER_CHANGE = "header_footer_change"
    LOGO_DETECTION = "logo_detection"
    SIGNATURE_DETECTION = "signature_detection"
    PAGE_ORIENTATION_CHANGE = "page_orientation_change"
    COLUMN_LAYOUT_CHANGE = "column_layout_change"
    WHITESPACE_PATTERN = "whitespace_pattern"

class VisualFeatures(BaseModel):
    """Visual features extracted from a page."""
    
    # Layout features
    num_columns: int = Field(default=1, ge=1)
    primary_font_size: Optional[float] = None
    primary_font_family: Optional[str] = None
    text_alignment: Optional[str] = None  # left, center, right, justify
    
    # Color features
    background_color: Optional[str] = None
    primary_text_color: Optional[str] = None
    has_color_images: bool = False
    
    # Structural features
    has_header: bool = False
    has_footer: bool = False
    header_text: Optional[str] = None
    footer_text: Optional[str] = None
    
    # Visual elements
    num_images: int = Field(default=0, ge=0)
    num_tables: int = Field(default=0, ge=0)
    num_charts: int = Field(default=0, ge=0)
    has_logo: bool = False
    has_signature: bool = False
    
    # Spacing
    avg_line_spacing: Optional[float] = None
    margin_top: Optional[float] = None
    margin_bottom: Optional[float] = None
    margin_left: Optional[float] = None
    margin_right: Optional[float] = None
    
    # Page characteristics
    orientation: str = Field(default="portrait")  # portrait or landscape
    aspect_ratio: Optional[float] = None

class PageVisualInfo(PageInfo):
    """Extended page info with visual features."""
    
    visual_features: Optional[VisualFeatures] = None
    picture_classifications: Dict[str, float] = Field(default_factory=dict)
    vlm_analysis: Optional[Dict[str, Any]] = None
```

### Visual Feature Processor

```python
# backend/src/core/visual_processor.py

import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
from io import BytesIO

from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureClassificationOptions,
    TableDetectionOptions
)
from docling.datamodel.base_models import PictureClassificationModel
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
        
        # Configure picture classification
        if self.enable_picture_classification:
            options.picture_classification_options = PictureClassificationOptions(
                enabled=True,
                model=PictureClassificationModel.DOCLING_V1,
                confidence_threshold=0.7
            )
        
        # Configure table detection
        if self.enable_table_detection:
            options.table_detection_options = TableDetectionOptions(
                enabled=True,
                detect_structure=True
            )
        
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
```

### Visual Boundary Detector

```python
# backend/src/core/visual_boundary_detector.py

import logging
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass

from .models import (
    PageVisualInfo, 
    Boundary, 
    Signal, 
    SignalType,
    VisualSignalType,
    VisualFeatures
)
from .visual_processor import VisualFeatureProcessor

logger = logging.getLogger(__name__)


@dataclass
class VisualBoundaryCandidate:
    """Candidate boundary based on visual analysis."""
    page_number: int
    visual_signals: List[Signal]
    visual_confidence: float
    layout_change_score: float
    visual_separator_score: float
    

class VisualBoundaryDetector:
    """
    Detects document boundaries using visual features and layout analysis.
    """
    
    # Thresholds for visual changes
    LAYOUT_CHANGE_THRESHOLD = 0.3
    FONT_CHANGE_THRESHOLD = 2.0  # Font size difference in points
    MARGIN_CHANGE_THRESHOLD = 0.1  # 10% of page dimension
    COLOR_CHANGE_THRESHOLD = 0.2
    
    # Visual signal weights
    VISUAL_SIGNAL_WEIGHTS = {
        VisualSignalType.LAYOUT_STRUCTURE_CHANGE: 0.8,
        VisualSignalType.FONT_STYLE_CHANGE: 0.6,
        VisualSignalType.COLOR_SCHEME_CHANGE: 0.5,
        VisualSignalType.VISUAL_SEPARATOR_LINE: 0.9,
        VisualSignalType.HEADER_FOOTER_CHANGE: 0.7,
        VisualSignalType.LOGO_DETECTION: 0.8,
        VisualSignalType.SIGNATURE_DETECTION: 0.6,
        VisualSignalType.PAGE_ORIENTATION_CHANGE: 0.9,
        VisualSignalType.COLUMN_LAYOUT_CHANGE: 0.7,
        VisualSignalType.WHITESPACE_PATTERN: 0.5
    }
    
    def __init__(
        self,
        min_visual_confidence: float = 0.5,
        enable_vlm_analysis: bool = True,
        visual_processor: Optional[VisualFeatureProcessor] = None
    ):
        """
        Initialize visual boundary detector.
        
        Args:
            min_visual_confidence: Minimum confidence for visual boundaries
            enable_vlm_analysis: Whether to use VLM for analysis
            visual_processor: Optional visual feature processor instance
        """
        self.min_visual_confidence = min_visual_confidence
        self.enable_vlm_analysis = enable_vlm_analysis
        self.visual_processor = visual_processor or VisualFeatureProcessor()
    
    def detect_visual_boundaries(
        self,
        pages: List[PageVisualInfo],
        context_window: int = 2
    ) -> List[VisualBoundaryCandidate]:
        """
        Detect boundaries based on visual features.
        
        Args:
            pages: List of pages with visual information
            context_window: Number of pages to consider for context
            
        Returns:
            List of visual boundary candidates
        """
        if not pages or len(pages) < 2:
            return []
        
        candidates = []
        
        # Analyze each page transition
        for i in range(1, len(pages)):
            prev_page = pages[i - 1]
            curr_page = pages[i]
            
            visual_signals = []
            
            # Check layout structure changes
            layout_signal = self._detect_layout_change(prev_page, curr_page)
            if layout_signal:
                visual_signals.append(layout_signal)
            
            # Check font style changes
            font_signal = self._detect_font_change(prev_page, curr_page)
            if font_signal:
                visual_signals.append(font_signal)
            
            # Check color scheme changes
            color_signal = self._detect_color_change(prev_page, curr_page)
            if color_signal:
                visual_signals.append(color_signal)
            
            # Check for visual separators
            separator_signal = self._detect_visual_separator(curr_page)
            if separator_signal:
                visual_signals.append(separator_signal)
            
            # Check header/footer changes
            header_footer_signal = self._detect_header_footer_change(prev_page, curr_page)
            if header_footer_signal:
                visual_signals.append(header_footer_signal)
            
            # Check for logos (often indicate new document)
            logo_signal = self._detect_logo_change(prev_page, curr_page)
            if logo_signal:
                visual_signals.append(logo_signal)
            
            # Check page orientation
            orientation_signal = self._detect_orientation_change(prev_page, curr_page)
            if orientation_signal:
                visual_signals.append(orientation_signal)
            
            # Analyze with VLM if enabled
            if self.enable_vlm_analysis and visual_signals:
                vlm_signal = self._analyze_with_vlm(prev_page, curr_page)
                if vlm_signal:
                    visual_signals.append(vlm_signal)
            
            # Create candidate if significant visual changes detected
            if visual_signals:
                confidence = self._calculate_visual_confidence(visual_signals)
                
                if confidence >= self.min_visual_confidence:
                    candidate = VisualBoundaryCandidate(
                        page_number=curr_page.page_number,
                        visual_signals=visual_signals,
                        visual_confidence=confidence,
                        layout_change_score=self._calculate_layout_change_score(
                            prev_page, curr_page
                        ),
                        visual_separator_score=self._calculate_separator_score(curr_page)
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def _detect_layout_change(
        self,
        prev_page: PageVisualInfo,
        curr_page: PageVisualInfo
    ) -> Optional[Signal]:
        """Detect significant layout structure changes."""
        if not (prev_page.visual_features and curr_page.visual_features):
            return None
        
        prev_features = prev_page.visual_features
        curr_features = curr_page.visual_features
        
        # Check column layout change
        if prev_features.num_columns != curr_features.num_columns:
            return Signal(
                type=VisualSignalType.COLUMN_LAYOUT_CHANGE,
                confidence=0.8,
                page_number=curr_page.page_number,
                description=f"Column layout changed from {prev_features.num_columns} to {curr_features.num_columns}"
            )
        
        # Check alignment change
        if prev_features.text_alignment != curr_features.text_alignment:
            return Signal(
                type=VisualSignalType.LAYOUT_STRUCTURE_CHANGE,
                confidence=0.6,
                page_number=curr_page.page_number,
                description=f"Text alignment changed from {prev_features.text_alignment} to {curr_features.text_alignment}"
            )
        
        # Check margin changes
        margin_changes = self._calculate_margin_changes(prev_features, curr_features)
        if margin_changes > self.MARGIN_CHANGE_THRESHOLD:
            return Signal(
                type=VisualSignalType.LAYOUT_STRUCTURE_CHANGE,
                confidence=0.7,
                page_number=curr_page.page_number,
                description=f"Significant margin changes detected ({margin_changes:.0%})"
            )
        
        return None
    
    def _detect_font_change(
        self,
        prev_page: PageVisualInfo,
        curr_page: PageVisualInfo
    ) -> Optional[Signal]:
        """Detect significant font changes."""
        if not (prev_page.visual_features and curr_page.visual_features):
            return None
        
        prev_features = prev_page.visual_features
        curr_features = curr_page.visual_features
        
        # Check font size change
        if (prev_features.primary_font_size and curr_features.primary_font_size):
            size_diff = abs(prev_features.primary_font_size - curr_features.primary_font_size)
            if size_diff > self.FONT_CHANGE_THRESHOLD:
                return Signal(
                    type=VisualSignalType.FONT_STYLE_CHANGE,
                    confidence=0.7,
                    page_number=curr_page.page_number,
                    description=f"Font size changed by {size_diff:.1f} points"
                )
        
        # Check font family change
        if (prev_features.primary_font_family and curr_features.primary_font_family and
            prev_features.primary_font_family != curr_features.primary_font_family):
            return Signal(
                type=VisualSignalType.FONT_STYLE_CHANGE,
                confidence=0.8,
                page_number=curr_page.page_number,
                description=f"Font family changed from {prev_features.primary_font_family} to {curr_features.primary_font_family}"
            )
        
        return None
    
    def _detect_color_change(
        self,
        prev_page: PageVisualInfo,
        curr_page: PageVisualInfo
    ) -> Optional[Signal]:
        """Detect significant color scheme changes."""
        if not (prev_page.visual_features and curr_page.visual_features):
            return None
        
        prev_features = prev_page.visual_features
        curr_features = curr_page.visual_features
        
        # Check background color change
        if (prev_features.background_color and curr_features.background_color and
            prev_features.background_color != curr_features.background_color):
            return Signal(
                type=VisualSignalType.COLOR_SCHEME_CHANGE,
                confidence=0.6,
                page_number=curr_page.page_number,
                description="Background color changed"
            )
        
        # Check text color change
        if (prev_features.primary_text_color and curr_features.primary_text_color and
            prev_features.primary_text_color != curr_features.primary_text_color):
            return Signal(
                type=VisualSignalType.COLOR_SCHEME_CHANGE,
                confidence=0.5,
                page_number=curr_page.page_number,
                description="Primary text color changed"
            )
        
        return None
    
    def _detect_visual_separator(self, page: PageVisualInfo) -> Optional[Signal]:
        """Detect visual separators like lines or boxes."""
        if not page.visual_features:
            return None
        
        # Check for separator patterns in layout elements
        for element in page.layout_elements:
            if element.get('type') == 'line' or element.get('type') == 'separator':
                return Signal(
                    type=VisualSignalType.VISUAL_SEPARATOR_LINE,
                    confidence=0.9,
                    page_number=page.page_number,
                    description="Visual separator line detected"
                )
        
        return None
    
    def _detect_header_footer_change(
        self,
        prev_page: PageVisualInfo,
        curr_page: PageVisualInfo
    ) -> Optional[Signal]:
        """Detect changes in headers or footers."""
        if not (prev_page.visual_features and curr_page.visual_features):
            return None
        
        prev_features = prev_page.visual_features
        curr_features = curr_page.visual_features
        
        # Check header change
        if (prev_features.header_text != curr_features.header_text):
            return Signal(
                type=VisualSignalType.HEADER_FOOTER_CHANGE,
                confidence=0.8,
                page_number=curr_page.page_number,
                description="Header content changed"
            )
        
        # Check footer change
        if (prev_features.footer_text != curr_features.footer_text):
            return Signal(
                type=VisualSignalType.HEADER_FOOTER_CHANGE,
                confidence=0.7,
                page_number=curr_page.page_number,
                description="Footer content changed"
            )
        
        return None
    
    def _detect_logo_change(
        self,
        prev_page: PageVisualInfo,
        curr_page: PageVisualInfo
    ) -> Optional[Signal]:
        """Detect appearance of logos."""
        if not (prev_page.visual_features and curr_page.visual_features):
            return None
        
        # Logo on current page but not previous
        if (curr_page.visual_features.has_logo and 
            not prev_page.visual_features.has_logo):
            return Signal(
                type=VisualSignalType.LOGO_DETECTION,
                confidence=0.8,
                page_number=curr_page.page_number,
                description="New logo detected"
            )
        
        return None
    
    def _detect_orientation_change(
        self,
        prev_page: PageVisualInfo,
        curr_page: PageVisualInfo
    ) -> Optional[Signal]:
        """Detect page orientation changes."""
        if not (prev_page.visual_features and curr_page.visual_features):
            return None
        
        if (prev_page.visual_features.orientation != 
            curr_page.visual_features.orientation):
            return Signal(
                type=VisualSignalType.PAGE_ORIENTATION_CHANGE,
                confidence=0.9,
                page_number=curr_page.page_number,
                description=f"Page orientation changed from {prev_page.visual_features.orientation} to {curr_page.visual_features.orientation}"
            )
        
        return None
    
    def _analyze_with_vlm(
        self,
        prev_page: PageVisualInfo,
        curr_page: PageVisualInfo
    ) -> Optional[Signal]:
        """Use VLM to analyze visual boundary."""
        # This would integrate with Docling's VLM capabilities
        # For now, returning None as placeholder
        return None
    
    def _calculate_margin_changes(
        self,
        prev_features: VisualFeatures,
        curr_features: VisualFeatures
    ) -> float:
        """Calculate the magnitude of margin changes."""
        changes = []
        
        if prev_features.margin_top and curr_features.margin_top:
            changes.append(abs(prev_features.margin_top - curr_features.margin_top))
        if prev_features.margin_bottom and curr_features.margin_bottom:
            changes.append(abs(prev_features.margin_bottom - curr_features.margin_bottom))
        if prev_features.margin_left and curr_features.margin_left:
            changes.append(abs(prev_features.margin_left - curr_features.margin_left))
        if prev_features.margin_right and curr_features.margin_right:
            changes.append(abs(prev_features.margin_right - curr_features.margin_right))
        
        return max(changes) if changes else 0.0
    
    def _calculate_visual_confidence(self, signals: List[Signal]) -> float:
        """Calculate overall confidence from visual signals."""
        if not signals:
            return 0.0
        
        # Weighted average of signal confidences
        total_weight = 0.0
        weighted_sum = 0.0
        
        for signal in signals:
            weight = self.VISUAL_SIGNAL_WEIGHTS.get(signal.type, 0.5)
            weighted_sum += signal.confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        base_confidence = weighted_sum / total_weight
        
        # Boost for multiple strong signals
        strong_signals = sum(1 for s in signals if s.confidence >= 0.7)
        if strong_signals >= 2:
            base_confidence = min(1.0, base_confidence * 1.1)
        
        return base_confidence
    
    def _calculate_layout_change_score(
        self,
        prev_page: PageVisualInfo,
        curr_page: PageVisualInfo
    ) -> float:
        """Calculate a comprehensive layout change score."""
        if not (prev_page.visual_features and curr_page.visual_features):
            return 0.0
        
        scores = []
        
        # Column change
        if prev_page.visual_features.num_columns != curr_page.visual_features.num_columns:
            scores.append(0.8)
        
        # Visual element count changes
        prev_elements = (prev_page.visual_features.num_images + 
                        prev_page.visual_features.num_tables +
                        prev_page.visual_features.num_charts)
        curr_elements = (curr_page.visual_features.num_images + 
                        curr_page.visual_features.num_tables +
                        curr_page.visual_features.num_charts)
        
        if abs(prev_elements - curr_elements) > 2:
            scores.append(0.6)
        
        # Margin changes
        margin_change = self._calculate_margin_changes(
            prev_page.visual_features,
            curr_page.visual_features
        )
        if margin_change > 0.1:
            scores.append(margin_change)
        
        return max(scores) if scores else 0.0
    
    def _calculate_separator_score(self, page: PageVisualInfo) -> float:
        """Calculate score for visual separators on the page."""
        score = 0.0
        
        # Check for separator elements
        for element in page.layout_elements:
            if element.get('type') in ['line', 'separator', 'rule']:
                score = max(score, 0.8)
            elif element.get('type') == 'box' and element.get('is_separator'):
                score = max(score, 0.7)
        
        # Check for significant whitespace patterns
        if page.visual_features and page.visual_features.whitespace_pattern:
            score = max(score, 0.5)
        
        return score
```

### Enhanced Document Processor Integration

```python
# backend/src/core/enhanced_document_processor.py

import logging
from typing import List, Optional, Iterator, Tuple
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
                    visual_features = future.result(timeout=30)
                    visual_page = PageVisualInfo(
                        **page.dict(),
                        visual_features=visual_features
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
                classifications = {}
                if hasattr(result, 'picture_classifications'):
                    for pic_class in result.picture_classifications:
                        if pic_class.page_number == page.page_number:
                            classifications[pic_class.label] = pic_class.confidence
                
                return features
        
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            return None
    
    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
```

### Configuration Management

```python
# backend/src/core/config.py

from pydantic import BaseSettings, Field
from typing import List, Optional


class VisualProcessingConfig(BaseSettings):
    """Configuration for visual processing features."""
    
    # Feature toggles
    enable_visual_features: bool = Field(
        default=True,
        env="ENABLE_VISUAL_FEATURES",
        description="Enable visual feature extraction"
    )
    enable_picture_classification: bool = Field(
        default=True,
        env="ENABLE_PICTURE_CLASSIFICATION",
        description="Enable Docling picture classification"
    )
    enable_vlm: bool = Field(
        default=False,
        env="ENABLE_VLM",
        description="Enable Vision Language Model (requires more resources)"
    )
    
    # Resource limits
    visual_memory_limit_mb: int = Field(
        default=2048,
        env="VISUAL_MEMORY_LIMIT_MB",
        description="Memory limit for visual processing in MB"
    )
    max_image_dimension: int = Field(
        default=1024,
        env="MAX_IMAGE_DIMENSION",
        description="Maximum dimension for processed images"
    )
    page_image_resolution: int = Field(
        default=150,
        env="PAGE_IMAGE_RESOLUTION",
        description="DPI for page image generation"
    )
    
    # Processing parameters
    visual_batch_size: int = Field(
        default=2,
        env="VISUAL_BATCH_SIZE",
        description="Number of pages to process visually in parallel"
    )
    visual_confidence_threshold: float = Field(
        default=0.5,
        env="VISUAL_CONFIDENCE_THRESHOLD",
        description="Minimum confidence for visual boundaries"
    )
    
    # VLM specific settings
    vlm_model: str = Field(
        default="docling-vlm-v1",
        env="VLM_MODEL",
        description="Vision Language Model to use"
    )
    vlm_prompt_template: str = Field(
        default="Analyze if this page transition represents a document boundary. Consider layout changes, visual separators, and document structure.",
        env="VLM_PROMPT_TEMPLATE",
        description="Prompt template for VLM analysis"
    )
    
    class Config:
        env_prefix = "PDF_SPLITTER_"
        case_sensitive = False
```

### Integration with Main Boundary Detector

```python
# backend/src/core/hybrid_boundary_detector.py

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from .boundary_detector import BoundaryDetector, BoundaryCandidate
from .visual_boundary_detector import VisualBoundaryDetector, VisualBoundaryCandidate
from .enhanced_document_processor import EnhancedDocumentProcessor
from .models import (
    PageInfo, 
    PageVisualInfo, 
    Boundary, 
    Signal, 
    SignalType,
    VisualSignalType
)
from .config import VisualProcessingConfig

logger = logging.getLogger(__name__)


class HybridBoundaryDetector:
    """
    Combines text-based and visual boundary detection for improved accuracy.
    """
    
    def __init__(
        self,
        config: Optional[VisualProcessingConfig] = None,
        text_weight: float = 0.6,
        visual_weight: float = 0.4
    ):
        """
        Initialize hybrid boundary detector.
        
        Args:
            config: Visual processing configuration
            text_weight: Weight for text-based signals
            visual_weight: Weight for visual signals
        """
        self.config = config or VisualProcessingConfig()
        self.text_weight = text_weight
        self.visual_weight = visual_weight
        
        # Initialize detectors
        self.text_detector = BoundaryDetector()
        self.visual_detector = VisualBoundaryDetector(
            enable_vlm_analysis=self.config.enable_vlm
        ) if self.config.enable_visual_features else None
        
        # Initialize processor
        self.processor = EnhancedDocumentProcessor(
            enable_visual_features=self.config.enable_visual_features,
            enable_vlm=self.config.enable_vlm,
            visual_memory_limit_mb=self.config.visual_memory_limit_mb
        )
    
    def detect_boundaries(
        self,
        file_path: Path,
        use_visual: Optional[bool] = None
    ) -> List[Boundary]:
        """
        Detect boundaries using hybrid approach.
        
        Args:
            file_path: Path to PDF file
            use_visual: Override config to enable/disable visual processing
            
        Returns:
            List of detected boundaries with combined confidence
        """
        logger.info(f"Starting hybrid boundary detection for: {file_path}")
        
        # Process document
        if use_visual is None:
            use_visual = self.config.enable_visual_features
        
        if use_visual and self.visual_detector:
            # Process with visual features
            pages = list(self.processor.process_document_with_visual(file_path))
            return self._detect_hybrid_boundaries(pages)
        else:
            # Process text only
            pages = list(self.processor.process_document(file_path))
            page_infos = [PageInfo(**p.dict()) for p in pages]
            return self.text_detector.detect_boundaries(page_infos)
    
    def _detect_hybrid_boundaries(
        self,
        pages: List[PageVisualInfo]
    ) -> List[Boundary]:
        """Combine text and visual boundary detection."""
        # Get text-based boundaries
        page_infos = [PageInfo(**p.dict()) for p in pages]
        text_boundaries = self.text_detector.detect_boundaries(page_infos)
        
        # Get visual boundaries
        visual_candidates = self.visual_detector.detect_visual_boundaries(pages)
        
        # Combine results
        combined_boundaries = self._combine_boundaries(
            text_boundaries,
            visual_candidates,
            pages
        )
        
        return combined_boundaries
    
    def _combine_boundaries(
        self,
        text_boundaries: List[Boundary],
        visual_candidates: List[VisualBoundaryCandidate],
        pages: List[PageVisualInfo]
    ) -> List[Boundary]:
        """Combine text and visual boundaries with weighted confidence."""
        # Create page number to boundary mapping
        boundary_map = {}
        
        # Add text boundaries
        for boundary in text_boundaries:
            if boundary.start_page not in boundary_map:
                boundary_map[boundary.start_page] = {
                    'text_boundary': boundary,
                    'visual_candidate': None,
                    'page': pages[boundary.start_page - 1]
                }
            else:
                boundary_map[boundary.start_page]['text_boundary'] = boundary
        
        # Add visual candidates
        for candidate in visual_candidates:
            if candidate.page_number not in boundary_map:
                boundary_map[candidate.page_number] = {
                    'text_boundary': None,
                    'visual_candidate': candidate,
                    'page': pages[candidate.page_number - 1]
                }
            else:
                boundary_map[candidate.page_number]['visual_candidate'] = candidate
        
        # Combine and create final boundaries
        combined_boundaries = []
        
        for page_num in sorted(boundary_map.keys()):
            data = boundary_map[page_num]
            text_boundary = data['text_boundary']
            visual_candidate = data['visual_candidate']
            
            # Calculate combined confidence
            if text_boundary and visual_candidate:
                # Both detectors agree - boost confidence
                combined_confidence = (
                    self.text_weight * text_boundary.confidence +
                    self.visual_weight * visual_candidate.visual_confidence
                )
                # Boost for agreement
                combined_confidence = min(1.0, combined_confidence * 1.1)
                
                # Combine signals
                all_signals = text_boundary.signals + visual_candidate.visual_signals
                
                boundary = Boundary(
                    start_page=page_num,
                    end_page=page_num,
                    confidence=combined_confidence,
                    signals=all_signals,
                    document_type=text_boundary.document_type,
                    metadata={
                        'text_confidence': text_boundary.confidence,
                        'visual_confidence': visual_candidate.visual_confidence,
                        'detection_method': 'hybrid'
                    }
                )
            
            elif text_boundary:
                # Only text detection
                boundary = text_boundary
                boundary.metadata['detection_method'] = 'text_only'
            
            elif visual_candidate and visual_candidate.visual_confidence > 0.7:
                # Strong visual signal without text confirmation
                boundary = Boundary(
                    start_page=page_num,
                    end_page=page_num,
                    confidence=visual_candidate.visual_confidence * 0.8,  # Slight penalty
                    signals=visual_candidate.visual_signals,
                    document_type=None,  # Will be detected later
                    metadata={
                        'visual_confidence': visual_candidate.visual_confidence,
                        'detection_method': 'visual_only'
                    }
                )
            else:
                continue
            
            combined_boundaries.append(boundary)
        
        # Fix end pages
        for i in range(len(combined_boundaries) - 1):
            combined_boundaries[i].end_page = combined_boundaries[i + 1].start_page - 1
        
        if combined_boundaries and pages:
            combined_boundaries[-1].end_page = pages[-1].page_number
        
        logger.info(f"Detected {len(combined_boundaries)} boundaries using hybrid approach")
        return combined_boundaries
```

### Example Usage

```python
# Example usage of the enhanced visual boundary detection

from pathlib import Path
from backend.src.core.hybrid_boundary_detector import HybridBoundaryDetector
from backend.src.core.config import VisualProcessingConfig

# Configure visual processing
config = VisualProcessingConfig(
    enable_visual_features=True,
    enable_picture_classification=True,
    enable_vlm=False,  # Can be enabled if VLM model is available
    visual_memory_limit_mb=2048,
    visual_confidence_threshold=0.6
)

# Create hybrid detector
detector = HybridBoundaryDetector(
    config=config,
    text_weight=0.5,  # Equal weights for demo
    visual_weight=0.5
)

# Process a PDF
pdf_path = Path("test_files/Test_PDF_Set_1.pdf")
boundaries = detector.detect_boundaries(pdf_path)

# Print results
for boundary in boundaries:
    print(f"\nDocument {boundary.start_page}-{boundary.end_page}")
    print(f"  Confidence: {boundary.confidence:.2f}")
    print(f"  Type: {boundary.document_type}")
    print(f"  Detection: {boundary.metadata.get('detection_method', 'unknown')}")
    print("  Signals:")
    
    # Group signals by type
    text_signals = [s for s in boundary.signals if isinstance(s.type, SignalType)]
    visual_signals = [s for s in boundary.signals if isinstance(s.type, VisualSignalType)]
    
    if text_signals:
        print("    Text-based:")
        for signal in text_signals:
            print(f"      - {signal.type.value}: {signal.description} ({signal.confidence:.2f})")
    
    if visual_signals:
        print("    Visual:")
        for signal in visual_signals:
            print(f"      - {signal.type.value}: {signal.description} ({signal.confidence:.2f})")
```

## Memory Management Strategy

1. **Page-by-page processing**: Visual features are extracted one page at a time
2. **Batch processing**: Pages are processed in configurable batches (default 2)
3. **Image resolution limiting**: Page images are generated at configurable DPI
4. **Feature caching**: Extracted features are cached to avoid reprocessing
5. **Garbage collection**: Explicit cleanup after each batch

## Configuration Options

The system is highly configurable through environment variables:

- `ENABLE_VISUAL_FEATURES`: Toggle visual processing entirely
- `ENABLE_PICTURE_CLASSIFICATION`: Enable/disable picture classification
- `ENABLE_VLM`: Enable/disable Vision Language Model (resource intensive)
- `VISUAL_MEMORY_LIMIT_MB`: Set memory limit for visual processing
- `VISUAL_BATCH_SIZE`: Number of pages to process in parallel
- `VISUAL_CONFIDENCE_THRESHOLD`: Minimum confidence for visual boundaries

## Benefits of Visual Detection

1. **Improved accuracy**: Visual cues often indicate boundaries more reliably than text
2. **Layout-aware**: Detects structural changes that text analysis might miss
3. **Logo/letterhead detection**: Identifies new documents by visual branding
4. **Form detection**: Better at identifying form boundaries through visual structure
5. **Complementary**: Works alongside text detection for higher confidence

## Future Enhancements

1. **Deep learning models**: Train custom models for boundary detection
2. **Template matching**: Detect known document templates
3. **Clustering**: Group visually similar pages
4. **Performance optimization**: GPU acceleration for image processing
5. **Advanced VLM integration**: Use more sophisticated vision-language models