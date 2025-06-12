"""
Configuration examples and usage patterns for the enhanced boundary detection system.

This module provides practical examples of how to configure and use the
enhanced boundary detector for different scenarios.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from .boundary_detector_enhanced import (
    EnhancedBoundaryDetector,
    DetectorConfig,
    RuleBasedDetectorPlugin,
    LLMBoundaryDetectorPlugin,
    VLMBoundaryDetectorPlugin,
    ConstructionSpecificDetectorPlugin
)
from .document_processor import DocumentProcessor
from .models import PageInfo, Boundary

logger = logging.getLogger(__name__)


class BoundaryDetectionPresets:
    """
    Preset configurations for different use cases.
    
    These presets demonstrate how to configure the detector
    for specific scenarios while maintaining flexibility.
    """
    
    @staticmethod
    def get_construction_focused_config() -> List[DetectorConfig]:
        """
        Configuration optimized for construction documents.
        
        Prioritizes construction-specific patterns while using
        LLM for complex cases.
        """
        return [
            DetectorConfig(
                name="construction",
                enabled=True,
                weight=1.5,  # Highest weight for domain expertise
                config={}
            ),
            DetectorConfig(
                name="rule_based",
                enabled=True,
                weight=1.0,
                config={
                    'min_confidence': 0.6,
                    'min_signals': 1
                }
            ),
            DetectorConfig(
                name="llm",
                enabled=True,
                weight=0.7,  # Lower weight, used for validation
                config={
                    'ollama_url': 'http://localhost:11434',
                    'model_name': 'llama3.2',
                    'temperature': 0.1,
                    'max_context_chars': 1500
                }
            )
        ]
    
    @staticmethod
    def get_general_purpose_config() -> List[DetectorConfig]:
        """
        Balanced configuration for general document types.
        
        Equal weighting across different detection methods.
        """
        return [
            DetectorConfig(
                name="rule_based",
                enabled=True,
                weight=1.0,
                config={
                    'min_confidence': 0.6,
                    'min_signals': 1
                }
            ),
            DetectorConfig(
                name="llm",
                enabled=True,
                weight=1.0,
                config={
                    'ollama_url': 'http://localhost:11434',
                    'model_name': 'llama3.2',
                    'temperature': 0.15
                }
            ),
            DetectorConfig(
                name="construction",
                enabled=True,
                weight=0.8,  # Still useful but not primary
                config={}
            )
        ]
    
    @staticmethod
    def get_high_accuracy_config() -> List[DetectorConfig]:
        """
        Configuration for maximum accuracy using all methods.
        
        Includes VLM for visual analysis at the cost of performance.
        """
        return [
            DetectorConfig(
                name="rule_based",
                enabled=True,
                weight=1.0,
                config={
                    'min_confidence': 0.7,
                    'min_signals': 2  # Require more signals
                }
            ),
            DetectorConfig(
                name="llm",
                enabled=True,
                weight=1.1,
                config={
                    'ollama_url': 'http://localhost:11434',
                    'model_name': 'llama3.2',
                    'temperature': 0.05,  # More deterministic
                    'max_context_chars': 3000  # More context
                }
            ),
            DetectorConfig(
                name="vlm",
                enabled=True,
                weight=0.9,
                config={
                    'vlm_model': 'dit-large',
                    'enable_vlm': True
                }
            ),
            DetectorConfig(
                name="construction",
                enabled=True,
                weight=1.2,
                config={}
            )
        ]
    
    @staticmethod
    def get_fast_config() -> List[DetectorConfig]:
        """
        Fast configuration using only rule-based detection.
        
        Suitable for quick processing or resource-constrained environments.
        """
        return [
            DetectorConfig(
                name="rule_based",
                enabled=True,
                weight=1.0,
                config={
                    'min_confidence': 0.5,  # Lower threshold for speed
                    'min_signals': 1
                }
            ),
            DetectorConfig(
                name="construction",
                enabled=True,
                weight=1.0,
                config={}
            )
        ]
    
    @staticmethod
    def get_llm_only_config(model_name: str = "llama3.2") -> List[DetectorConfig]:
        """
        Configuration using only LLM-based detection.
        
        Useful for testing or when traditional patterns don't apply.
        """
        return [
            DetectorConfig(
                name="llm",
                enabled=True,
                weight=1.0,
                config={
                    'ollama_url': 'http://localhost:11434',
                    'model_name': model_name,
                    'temperature': 0.1,
                    'max_context_chars': 2500
                }
            )
        ]


class AdaptiveBoundaryDetector:
    """
    Adaptive detector that can switch configurations based on document characteristics.
    
    This demonstrates how to dynamically adjust detection strategies.
    """
    
    def __init__(self):
        self.presets = BoundaryDetectionPresets()
        self.current_config = None
        self.detector = None
        
    async def detect_with_adaptation(
        self,
        file_path: Path,
        initial_analysis: Optional[Dict[str, Any]] = None
    ) -> List[Boundary]:
        """
        Detect boundaries with adaptive configuration selection.
        
        Args:
            file_path: Path to PDF file
            initial_analysis: Optional pre-analysis of document
            
        Returns:
            List of detected boundaries
        """
        # Analyze document characteristics if not provided
        if initial_analysis is None:
            initial_analysis = await self._analyze_document(file_path)
            
        # Select appropriate configuration
        config = self._select_configuration(initial_analysis)
        
        # Create detector with selected config
        self.detector = EnhancedBoundaryDetector(
            min_confidence=0.6,
            min_signals=1,
            detector_configs=config
        )
        
        # Process document
        processor = DocumentProcessor()
        pages = list(processor.process_document(file_path))
        
        # Detect boundaries
        boundaries = await self.detector.detect_boundaries(pages)
        
        # Post-process if needed
        boundaries = self._post_process_boundaries(boundaries, initial_analysis)
        
        return boundaries
    
    async def _analyze_document(self, file_path: Path) -> Dict[str, Any]:
        """Analyze document to determine characteristics."""
        processor = DocumentProcessor(enable_ocr=False)  # Fast scan
        
        analysis = {
            'total_pages': 0,
            'avg_words_per_page': 0,
            'has_images': False,
            'has_tables': False,
            'detected_types': set(),
            'is_scanned': False
        }
        
        word_counts = []
        sample_pages = []
        
        # Sample first 10 pages
        for i, page in enumerate(processor.process_document(file_path)):
            if i >= 10:
                break
                
            analysis['total_pages'] += 1
            word_counts.append(page.word_count)
            
            if page.has_images:
                analysis['has_images'] = True
            if page.has_tables:
                analysis['has_tables'] = True
                
            # Check if scanned (very low word count despite having content)
            if page.word_count < 50 and len(page.layout_elements) > 0:
                analysis['is_scanned'] = True
                
            sample_pages.append(page)
            
        analysis['avg_words_per_page'] = sum(word_counts) / len(word_counts) if word_counts else 0
        
        # Quick type detection on sample
        for page in sample_pages[:3]:
            if page.text_content:
                text_lower = page.text_content[:500].lower()
                if 'rfi' in text_lower or 'submittal' in text_lower:
                    analysis['detected_types'].add('construction')
                if 'from:' in text_lower and 'to:' in text_lower:
                    analysis['detected_types'].add('email')
                if 'invoice' in text_lower or 'purchase order' in text_lower:
                    analysis['detected_types'].add('business')
                    
        return analysis
    
    def _select_configuration(self, analysis: Dict[str, Any]) -> List[DetectorConfig]:
        """Select configuration based on document analysis."""
        # Construction documents
        if 'construction' in analysis['detected_types']:
            logger.info("Selected construction-focused configuration")
            return self.presets.get_construction_focused_config()
            
        # Scanned documents - need OCR and possibly VLM
        if analysis['is_scanned']:
            logger.info("Selected high-accuracy configuration for scanned document")
            return self.presets.get_high_accuracy_config()
            
        # Large documents - use faster config
        if analysis['total_pages'] > 100:
            logger.info("Selected fast configuration for large document")
            return self.presets.get_fast_config()
            
        # Complex documents with mixed content
        if analysis['has_tables'] and analysis['has_images']:
            logger.info("Selected high-accuracy configuration for complex document")
            return self.presets.get_high_accuracy_config()
            
        # Default to general purpose
        logger.info("Selected general-purpose configuration")
        return self.presets.get_general_purpose_config()
    
    def _post_process_boundaries(
        self,
        boundaries: List[Boundary],
        analysis: Dict[str, Any]
    ) -> List[Boundary]:
        """Post-process boundaries based on document characteristics."""
        # For construction documents, merge closely spaced boundaries
        if 'construction' in analysis['detected_types']:
            return self._merge_close_boundaries(boundaries, threshold=2)
            
        return boundaries
    
    def _merge_close_boundaries(
        self,
        boundaries: List[Boundary],
        threshold: int = 2
    ) -> List[Boundary]:
        """Merge boundaries that are very close together."""
        if len(boundaries) <= 1:
            return boundaries
            
        merged = [boundaries[0]]
        
        for boundary in boundaries[1:]:
            prev = merged[-1]
            
            # Check if boundaries are close
            if boundary.start_page - prev.end_page <= threshold:
                # Merge by extending previous boundary
                prev.end_page = boundary.end_page
                # Combine signals
                prev.signals.extend(boundary.signals)
                # Recalculate confidence
                prev.confidence = max(prev.confidence, boundary.confidence)
            else:
                merged.append(boundary)
                
        return merged


class BatchProcessingExample:
    """
    Example of batch processing multiple PDFs with progress tracking.
    """
    
    def __init__(self, config: List[DetectorConfig]):
        self.detector = EnhancedBoundaryDetector(
            min_confidence=0.6,
            min_signals=1,
            detector_configs=config
        )
        self.processor = DocumentProcessor()
        
    async def process_batch(
        self,
        pdf_files: List[Path],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, List[Boundary]]:
        """
        Process multiple PDFs in batch.
        
        Args:
            pdf_files: List of PDF file paths
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping file paths to boundaries
        """
        results = {}
        
        for i, pdf_file in enumerate(pdf_files):
            try:
                # Update progress
                if progress_callback:
                    progress_callback(i + 1, len(pdf_files), str(pdf_file))
                    
                # Process file
                pages = list(self.processor.process_document(pdf_file))
                boundaries = await self.detector.detect_boundaries(pages)
                
                results[str(pdf_file)] = boundaries
                
                logger.info(f"Processed {pdf_file}: {len(boundaries)} documents detected")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                results[str(pdf_file)] = []
                
        return results


# Example usage functions
async def example_construction_processing():
    """Example: Process construction documents with specialized config."""
    # Get construction-focused configuration
    config = BoundaryDetectionPresets.get_construction_focused_config()
    
    # Create detector
    detector = EnhancedBoundaryDetector(
        min_confidence=0.7,
        min_signals=1,
        detector_configs=config
    )
    
    # Process a construction PDF
    processor = DocumentProcessor()
    pages = list(processor.process_document(Path("construction_docs.pdf")))
    
    # Detect boundaries
    boundaries = await detector.detect_boundaries(pages)
    
    # Print results
    print(f"Found {len(boundaries)} documents:")
    for i, boundary in enumerate(boundaries):
        print(f"\nDocument {i+1}:")
        print(f"  Pages: {boundary.page_range}")
        print(f"  Type: {boundary.document_type}")
        print(f"  Confidence: {boundary.confidence:.2f}")
        
        # Show construction-specific signals
        construction_signals = [
            s for s in boundary.signals 
            if 'construction_type' in s.metadata
        ]
        if construction_signals:
            print("  Construction types detected:")
            for signal in construction_signals:
                print(f"    - {signal.metadata['construction_type']}")


async def example_adaptive_processing():
    """Example: Use adaptive configuration selection."""
    # Create adaptive detector
    adaptive = AdaptiveBoundaryDetector()
    
    # Process different types of documents
    test_files = [
        Path("mixed_construction_docs.pdf"),
        Path("scanned_invoices.pdf"),
        Path("email_thread.pdf"),
        Path("large_report.pdf")
    ]
    
    for pdf_file in test_files:
        if pdf_file.exists():
            print(f"\nProcessing: {pdf_file}")
            
            # Detect with automatic configuration selection
            boundaries = await adaptive.detect_with_adaptation(pdf_file)
            
            print(f"Configuration used: {adaptive.current_config}")
            print(f"Documents found: {len(boundaries)}")


async def example_custom_llm_model():
    """Example: Use a custom LLM model for detection."""
    # Configure with specific LLM model
    config = [
        DetectorConfig(
            name="llm",
            enabled=True,
            weight=1.0,
            config={
                'ollama_url': 'http://localhost:11434',
                'model_name': 'mixtral:8x7b',  # More powerful model
                'temperature': 0.05,
                'max_context_chars': 4000
            }
        ),
        DetectorConfig(
            name="rule_based",
            enabled=True,
            weight=0.8,
            config={'min_confidence': 0.7}
        )
    ]
    
    detector = EnhancedBoundaryDetector(
        min_confidence=0.65,
        min_signals=1,
        detector_configs=config
    )
    
    # Process document
    processor = DocumentProcessor()
    pages = list(processor.process_document(Path("complex_document.pdf")))
    boundaries = await detector.detect_boundaries(pages)
    
    # Show LLM-specific detections
    for boundary in boundaries:
        llm_signals = [s for s in boundary.signals if s.type == "llm_semantic"]
        if llm_signals:
            print(f"\nLLM detected boundary at page {boundary.start_page}")
            for signal in llm_signals:
                print(f"  Reasoning: {signal.metadata.get('detected_type')}")
                print(f"  Confidence: {signal.confidence:.2f}")


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_construction_processing())