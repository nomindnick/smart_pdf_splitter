"""
Example usage of the enhanced visual boundary detection system.

This script demonstrates how to use the hybrid boundary detector with
various configuration options.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.hybrid_boundary_detector import HybridBoundaryDetector, VisualProcessingConfig
from src.core.models import SignalType, VisualSignalType
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """Run visual boundary detection example."""
    
    # Configure visual processing
    config = VisualProcessingConfig(
        enable_visual_features=True,
        enable_picture_classification=True,
        enable_vlm=False,  # Can be enabled if VLM model is available
        visual_memory_limit_mb=2048,
        visual_confidence_threshold=0.6,
        visual_batch_size=2
    )
    
    # Create hybrid detector with balanced weights
    detector = HybridBoundaryDetector(
        config=config,
        text_weight=0.5,  # Equal weights for demo
        visual_weight=0.5,
        min_combined_confidence=0.5
    )
    
    # Example 1: Process a test PDF
    print("\n=== Example 1: Processing Test PDF ===")
    pdf_path = Path("../../tests/test_files/Test_PDF_Set_1.pdf")
    
    if pdf_path.exists():
        boundaries = detector.detect_boundaries(pdf_path)
        print_boundaries(boundaries)
        
        # Get detection summary
        summary = detector.get_detection_summary(boundaries)
        print_summary(summary)
    else:
        print(f"Test PDF not found at {pdf_path}")
    
    # Example 2: Process with visual features disabled
    print("\n=== Example 2: Text-Only Detection ===")
    if pdf_path.exists():
        boundaries_text_only = detector.detect_boundaries(pdf_path, use_visual=False)
        print(f"Detected {len(boundaries_text_only)} boundaries using text-only detection")
    
    # Example 3: Process specific page range
    print("\n=== Example 3: Processing Page Range ===")
    if pdf_path.exists():
        boundaries_partial = detector.detect_boundaries(pdf_path, page_range=(1, 10))
        print(f"Detected {len(boundaries_partial)} boundaries in pages 1-10")
    
    # Example 4: Process from bytes (simulating upload)
    print("\n=== Example 4: Processing from Bytes Stream ===")
    if pdf_path.exists():
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        boundaries_stream = detector.detect_boundaries_from_stream(
            pdf_bytes,
            filename="uploaded_document.pdf"
        )
        print(f"Detected {len(boundaries_stream)} boundaries from stream")
    
    # Example 5: Custom configuration
    print("\n=== Example 5: Custom Configuration ===")
    custom_config = VisualProcessingConfig(
        enable_visual_features=True,
        enable_picture_classification=False,  # Disable picture classification
        visual_confidence_threshold=0.7,  # Higher threshold
        visual_memory_limit_mb=1024  # Lower memory limit
    )
    
    custom_detector = HybridBoundaryDetector(
        config=custom_config,
        text_weight=0.7,  # Favor text signals
        visual_weight=0.3
    )
    
    if pdf_path.exists():
        boundaries_custom = custom_detector.detect_boundaries(pdf_path)
        print(f"Detected {len(boundaries_custom)} boundaries with custom config")


def print_boundaries(boundaries):
    """Print detailed boundary information."""
    for i, boundary in enumerate(boundaries, 1):
        print(f"\nDocument {i}: Pages {boundary.start_page}-{boundary.end_page}")
        print(f"  Confidence: {boundary.confidence:.2f}")
        print(f"  Type: {boundary.document_type.value if boundary.document_type else 'Unknown'}")
        print(f"  Detection: {boundary.metadata.get('detection_method', 'unknown')}")
        
        # Group signals by type
        text_signals = [s for s in boundary.signals if isinstance(s.type, SignalType)]
        visual_signals = [s for s in boundary.signals if isinstance(s.type, VisualSignalType)]
        
        if text_signals:
            print("  Text Signals:")
            for signal in text_signals[:3]:  # Show first 3
                print(f"    - {signal.type.value}: {signal.description} ({signal.confidence:.2f})")
            if len(text_signals) > 3:
                print(f"    ... and {len(text_signals) - 3} more")
        
        if visual_signals:
            print("  Visual Signals:")
            for signal in visual_signals[:3]:  # Show first 3
                print(f"    - {signal.type.value}: {signal.description} ({signal.confidence:.2f})")
            if len(visual_signals) > 3:
                print(f"    ... and {len(visual_signals) - 3} more")
        
        # Show additional metadata if hybrid detection
        if boundary.metadata.get('detection_method') == 'hybrid':
            print(f"  Text Confidence: {boundary.metadata.get('text_confidence', 0):.2f}")
            print(f"  Visual Confidence: {boundary.metadata.get('visual_confidence', 0):.2f}")
            print(f"  Layout Change Score: {boundary.metadata.get('layout_change_score', 0):.2f}")


def print_summary(summary):
    """Print detection summary."""
    print("\n=== Detection Summary ===")
    print(f"Total Documents: {summary['total_documents']}")
    print(f"Average Confidence: {summary['average_confidence']:.2f}")
    print(f"High Confidence (>=0.8): {summary['high_confidence_boundaries']}")
    print(f"Low Confidence (<0.6): {summary['low_confidence_boundaries']}")
    
    print("\nDetection Methods:")
    for method, count in summary['detection_methods'].items():
        print(f"  {method}: {count}")
    
    print("\nDocument Types:")
    for doc_type, count in summary['document_types'].items():
        print(f"  {doc_type}: {count}")
    
    if summary['visual_signals_used']:
        print("\nVisual Signals Used:")
        for signal, count in sorted(summary['visual_signals_used'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {signal}: {count}")
    
    if summary['text_signals_used']:
        print("\nText Signals Used:")
        for signal, count in sorted(summary['text_signals_used'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {signal}: {count}")


if __name__ == "__main__":
    main()