"""
VLM (Vision Language Model) integration for advanced boundary detection.

This module demonstrates how to leverage Docling's VLM capabilities for:
1. Visual layout analysis
2. Multi-modal boundary detection
3. Advanced document understanding
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    LayoutModelOptions,
    TableDetectionModelOptions
)
from docling.models.layout_analyzer import LayoutAnalyzer
from docling.models.table_detection import TableDetectionModel

from .models import PageInfo, Signal, SignalType, BoundingBox
from .boundary_detector_enhanced import ExtendedSignalType

logger = logging.getLogger(__name__)


@dataclass
class VisualFeatures:
    """Visual features extracted from a page."""
    layout_type: str  # 'single_column', 'multi_column', 'mixed'
    has_header: bool
    has_footer: bool
    column_count: int
    text_blocks: List[BoundingBox]
    image_blocks: List[BoundingBox]
    table_blocks: List[BoundingBox]
    whitespace_ratio: float
    dominant_font_size: Optional[float]
    color_complexity: float  # 0-1, higher means more colors
    visual_signature: np.ndarray  # Feature vector for similarity


class VLMDocumentAnalyzer:
    """
    Advanced document analyzer using Vision Language Models.
    
    Combines visual and textual analysis for superior boundary detection.
    """
    
    def __init__(
        self,
        layout_model: str = "dit-large",  # Document Image Transformer
        table_model: str = "table-transformer",
        enable_color_analysis: bool = True,
        enable_font_analysis: bool = True
    ):
        """
        Initialize VLM analyzer with model configurations.
        
        Args:
            layout_model: Model for layout analysis
            table_model: Model for table detection
            enable_color_analysis: Analyze color patterns
            enable_font_analysis: Analyze font patterns
        """
        self.layout_model = layout_model
        self.table_model = table_model
        self.enable_color = enable_color_analysis
        self.enable_font = enable_font_analysis
        
        self._setup_pipeline()
        
    def _setup_pipeline(self):
        """Configure Docling pipeline for VLM analysis."""
        # Configure layout model
        layout_options = LayoutModelOptions(
            model_name=self.layout_model,
            device='cuda' if self._cuda_available() else 'cpu',
            batch_size=4
        )
        
        # Configure table detection
        table_options = TableDetectionModelOptions(
            model_name=self.table_model,
            device='cuda' if self._cuda_available() else 'cpu'
        )
        
        # Create pipeline options
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_layout_analysis = True
        self.pipeline_options.layout_model_options = layout_options
        self.pipeline_options.do_table_detection = True
        self.pipeline_options.table_detection_options = table_options
        
        # Enable visual features
        self.pipeline_options.generate_page_images = True
        self.pipeline_options.page_image_resolution = 300  # DPI
        self.pipeline_options.extract_font_info = self.enable_font
        self.pipeline_options.extract_color_info = self.enable_color
        
        # Create converter
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options
                )
            }
        )
        
    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
            
    def analyze_page_visual_features(
        self,
        page_info: PageInfo,
        page_image: Optional[Any] = None
    ) -> VisualFeatures:
        """
        Extract visual features from a page.
        
        Args:
            page_info: Page information
            page_image: Optional page image for analysis
            
        Returns:
            Visual features of the page
        """
        # Initialize feature extraction
        text_blocks = []
        image_blocks = []
        table_blocks = []
        
        # Extract layout elements
        for element in page_info.layout_elements:
            elem_type = element.get('type', '').lower()
            bbox = element.get('bbox')
            
            if bbox:
                if 'text' in elem_type or 'paragraph' in elem_type:
                    text_blocks.append(bbox)
                elif 'image' in elem_type or 'figure' in elem_type:
                    image_blocks.append(bbox)
                elif 'table' in elem_type:
                    table_blocks.append(bbox)
                    
        # Determine layout type
        layout_type = self._determine_layout_type(text_blocks, page_info)
        
        # Check for headers/footers
        has_header = self._has_header(text_blocks, page_info.height)
        has_footer = self._has_footer(text_blocks, page_info.height)
        
        # Calculate whitespace ratio
        whitespace_ratio = self._calculate_whitespace_ratio(
            text_blocks + image_blocks + table_blocks,
            page_info.width,
            page_info.height
        )
        
        # Generate visual signature
        visual_signature = self._generate_visual_signature(
            text_blocks, image_blocks, table_blocks,
            page_info.width, page_info.height
        )
        
        return VisualFeatures(
            layout_type=layout_type,
            has_header=has_header,
            has_footer=has_footer,
            column_count=self._count_columns(text_blocks),
            text_blocks=text_blocks,
            image_blocks=image_blocks,
            table_blocks=table_blocks,
            whitespace_ratio=whitespace_ratio,
            dominant_font_size=None,  # Would need font analysis
            color_complexity=0.0,  # Would need color analysis
            visual_signature=visual_signature
        )
        
    def _determine_layout_type(
        self,
        text_blocks: List[BoundingBox],
        page_info: PageInfo
    ) -> str:
        """Determine the layout type of the page."""
        if not text_blocks:
            return 'empty'
            
        # Analyze x-coordinates to detect columns
        x_positions = [bbox.x for bbox in text_blocks]
        x_clusters = self._cluster_positions(x_positions, threshold=50)
        
        if len(x_clusters) == 1:
            return 'single_column'
        elif len(x_clusters) == 2:
            return 'two_column'
        elif len(x_clusters) > 2:
            return 'multi_column'
        else:
            return 'mixed'
            
    def _has_header(self, text_blocks: List[BoundingBox], page_height: float) -> bool:
        """Check if page has a header."""
        header_threshold = page_height * 0.15
        header_blocks = [b for b in text_blocks if b.y < header_threshold]
        return len(header_blocks) > 0
        
    def _has_footer(self, text_blocks: List[BoundingBox], page_height: float) -> bool:
        """Check if page has a footer."""
        footer_threshold = page_height * 0.85
        footer_blocks = [b for b in text_blocks if b.y > footer_threshold]
        return len(footer_blocks) > 0
        
    def _count_columns(self, text_blocks: List[BoundingBox]) -> int:
        """Count the number of text columns."""
        if not text_blocks:
            return 0
            
        x_positions = [bbox.x for bbox in text_blocks]
        clusters = self._cluster_positions(x_positions, threshold=50)
        return len(clusters)
        
    def _cluster_positions(self, positions: List[float], threshold: float) -> List[List[float]]:
        """Cluster positions based on threshold."""
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
        
    def _calculate_whitespace_ratio(
        self,
        blocks: List[BoundingBox],
        page_width: float,
        page_height: float
    ) -> float:
        """Calculate the ratio of whitespace on the page."""
        if not blocks:
            return 1.0
            
        # Calculate total area covered by blocks
        covered_area = 0.0
        for block in blocks:
            covered_area += block.width * block.height
            
        total_area = page_width * page_height
        return 1.0 - (covered_area / total_area) if total_area > 0 else 1.0
        
    def _generate_visual_signature(
        self,
        text_blocks: List[BoundingBox],
        image_blocks: List[BoundingBox],
        table_blocks: List[BoundingBox],
        page_width: float,
        page_height: float
    ) -> np.ndarray:
        """
        Generate a visual signature vector for the page.
        
        This can be used for similarity comparison between pages.
        """
        features = []
        
        # Basic counts (normalized)
        features.append(len(text_blocks) / 100.0)
        features.append(len(image_blocks) / 10.0)
        features.append(len(table_blocks) / 5.0)
        
        # Spatial distribution features
        if text_blocks:
            # Average position
            avg_x = np.mean([b.x for b in text_blocks]) / page_width
            avg_y = np.mean([b.y for b in text_blocks]) / page_height
            features.extend([avg_x, avg_y])
            
            # Spread
            std_x = np.std([b.x for b in text_blocks]) / page_width
            std_y = np.std([b.y for b in text_blocks]) / page_height
            features.extend([std_x, std_y])
        else:
            features.extend([0, 0, 0, 0])
            
        # Size features
        if text_blocks:
            avg_width = np.mean([b.width for b in text_blocks]) / page_width
            avg_height = np.mean([b.height for b in text_blocks]) / page_height
            features.extend([avg_width, avg_height])
        else:
            features.extend([0, 0])
            
        return np.array(features)
        
    def compare_visual_similarity(
        self,
        features1: VisualFeatures,
        features2: VisualFeatures
    ) -> float:
        """
        Compare visual similarity between two pages.
        
        Returns:
            Similarity score between 0 and 1
        """
        # Layout type similarity
        layout_score = 1.0 if features1.layout_type == features2.layout_type else 0.0
        
        # Structure similarity
        structure_score = 0.0
        if features1.has_header == features2.has_header:
            structure_score += 0.5
        if features1.has_footer == features2.has_footer:
            structure_score += 0.5
            
        # Column similarity
        column_diff = abs(features1.column_count - features2.column_count)
        column_score = 1.0 / (1.0 + column_diff)
        
        # Whitespace similarity
        ws_diff = abs(features1.whitespace_ratio - features2.whitespace_ratio)
        ws_score = 1.0 - ws_diff
        
        # Visual signature similarity (cosine similarity)
        sig_score = self._cosine_similarity(
            features1.visual_signature,
            features2.visual_signature
        )
        
        # Weighted combination
        weights = {
            'layout': 0.3,
            'structure': 0.2,
            'columns': 0.2,
            'whitespace': 0.1,
            'signature': 0.2
        }
        
        total_score = (
            weights['layout'] * layout_score +
            weights['structure'] * structure_score +
            weights['columns'] * column_score +
            weights['whitespace'] * ws_score +
            weights['signature'] * sig_score
        )
        
        return total_score
        
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
        
    def detect_visual_boundaries(
        self,
        pages: List[PageInfo],
        similarity_threshold: float = 0.3
    ) -> List[Tuple[int, float]]:
        """
        Detect boundaries based on visual dissimilarity.
        
        Args:
            pages: List of pages to analyze
            similarity_threshold: Threshold below which pages are considered different
            
        Returns:
            List of (page_number, confidence) tuples for boundaries
        """
        boundaries = []
        
        # Extract visual features for all pages
        page_features = []
        for page in pages:
            features = self.analyze_page_visual_features(page)
            page_features.append(features)
            
        # Compare consecutive pages
        for i in range(1, len(pages)):
            similarity = self.compare_visual_similarity(
                page_features[i-1],
                page_features[i]
            )
            
            # Low similarity indicates a boundary
            if similarity < similarity_threshold:
                confidence = 1.0 - similarity  # Higher confidence for lower similarity
                boundaries.append((pages[i].page_number, confidence))
                
        return boundaries


class MultiModalBoundaryDetector:
    """
    Combines visual and textual analysis for boundary detection.
    
    Uses both VLM analysis and LLM understanding for best results.
    """
    
    def __init__(
        self,
        vlm_analyzer: VLMDocumentAnalyzer,
        enable_cross_modal_validation: bool = True
    ):
        """
        Initialize multi-modal detector.
        
        Args:
            vlm_analyzer: VLM analyzer instance
            enable_cross_modal_validation: Validate across modalities
        """
        self.vlm_analyzer = vlm_analyzer
        self.cross_validate = enable_cross_modal_validation
        
    def analyze_boundary_multimodal(
        self,
        prev_page: PageInfo,
        curr_page: PageInfo,
        text_similarity: float,
        visual_similarity: float
    ) -> Tuple[bool, float, str]:
        """
        Analyze boundary using both text and visual signals.
        
        Args:
            prev_page: Previous page
            curr_page: Current page
            text_similarity: Text-based similarity score
            visual_similarity: Visual similarity score
            
        Returns:
            Tuple of (is_boundary, confidence, reason)
        """
        # Define thresholds
        text_threshold = 0.4
        visual_threshold = 0.3
        
        # Check individual modalities
        text_boundary = text_similarity < text_threshold
        visual_boundary = visual_similarity < visual_threshold
        
        # Multi-modal decision logic
        if text_boundary and visual_boundary:
            # Both modalities agree - high confidence
            confidence = 0.9
            reason = "Both text and visual analysis indicate boundary"
            return True, confidence, reason
            
        elif text_boundary or visual_boundary:
            # One modality indicates boundary
            if self.cross_validate:
                # Use cross-validation logic
                if text_boundary and visual_similarity > 0.7:
                    # Text says boundary but visually very similar
                    confidence = 0.5
                    reason = "Text indicates boundary but visually similar"
                elif visual_boundary and text_similarity > 0.7:
                    # Visual says boundary but textually similar
                    confidence = 0.4
                    reason = "Visual boundary but textually similar"
                else:
                    # Moderate disagreement
                    confidence = 0.7
                    reason = f"{'Text' if text_boundary else 'Visual'} analysis indicates boundary"
            else:
                # Simple OR logic
                confidence = 0.7
                reason = f"{'Text' if text_boundary else 'Visual'} analysis indicates boundary"
                
            return True, confidence, reason
            
        else:
            # Neither modality indicates boundary
            return False, 0.0, "No boundary detected"
            
    def create_boundary_signal(
        self,
        page_number: int,
        confidence: float,
        reason: str,
        visual_features: Optional[VisualFeatures] = None
    ) -> Signal:
        """Create a multi-modal boundary signal."""
        metadata = {
            'detection_method': 'multimodal',
            'reason': reason
        }
        
        if visual_features:
            metadata.update({
                'layout_type': visual_features.layout_type,
                'column_count': visual_features.column_count,
                'whitespace_ratio': visual_features.whitespace_ratio
            })
            
        return Signal(
            type=ExtendedSignalType.VLM_VISUAL,
            confidence=confidence,
            page_number=page_number,
            description=f"Multi-modal boundary: {reason}",
            metadata=metadata
        )


# Example integration
def create_vlm_enhanced_pipeline():
    """
    Create a complete VLM-enhanced boundary detection pipeline.
    
    This demonstrates how to integrate VLM capabilities into the
    existing boundary detection system.
    """
    from .boundary_detector_enhanced import (
        EnhancedBoundaryDetector,
        DetectorConfig,
        VLMBoundaryDetectorPlugin
    )
    
    # Configure VLM analyzer
    vlm_analyzer = VLMDocumentAnalyzer(
        layout_model="dit-large",
        table_model="table-transformer",
        enable_color_analysis=True,
        enable_font_analysis=True
    )
    
    # Create multi-modal detector
    multimodal = MultiModalBoundaryDetector(
        vlm_analyzer=vlm_analyzer,
        enable_cross_modal_validation=True
    )
    
    # Configure enhanced boundary detector with VLM
    detector_configs = [
        # Rule-based detector
        DetectorConfig(
            name="rule_based",
            enabled=True,
            weight=1.0,
            config={'min_confidence': 0.6}
        ),
        # LLM detector
        DetectorConfig(
            name="llm",
            enabled=True,
            weight=0.8,
            config={
                'ollama_url': 'http://localhost:11434',
                'model_name': 'llama3.2'
            }
        ),
        # VLM detector with enhanced configuration
        DetectorConfig(
            name="vlm",
            enabled=True,
            weight=0.9,
            config={
                'vlm_model': 'dit-large',
                'vlm_analyzer': vlm_analyzer,
                'multimodal_detector': multimodal
            }
        ),
        # Construction-specific detector
        DetectorConfig(
            name="construction",
            enabled=True,
            weight=1.2,
            config={}
        )
    ]
    
    # Create enhanced detector
    detector = EnhancedBoundaryDetector(
        min_confidence=0.65,
        min_signals=1,
        detector_configs=detector_configs
    )
    
    return detector, vlm_analyzer, multimodal