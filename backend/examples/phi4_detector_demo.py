"""
Demo script showing how to use the Phi-4 Mini boundary detector.

This example demonstrates:
1. Basic usage with pattern-based detection
2. Enhanced detection with LLM analysis
3. Explaining boundary decisions
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import DocumentProcessor, Phi4MiniBoundaryDetector
from src.core.models import ProcessingStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def process_with_phi4(pdf_path: str, use_llm: bool = True):
    """Process a PDF using Phi-4 Mini boundary detection."""
    
    logger.info(f"Processing {pdf_path}")
    logger.info(f"LLM analysis: {'Enabled' if use_llm else 'Disabled'}")
    
    # Initialize document processor
    processor = DocumentProcessor()
    
    # Initialize Phi-4 detector
    try:
        detector = Phi4MiniBoundaryDetector(
            model_name="phi4-mini:3.8b",
            use_llm_for_ambiguous=use_llm,
            min_confidence=0.6,
            llm_batch_size=3,
            llm_timeout=30.0
        )
        logger.info(f"Phi-4 detector initialized (LLM: {detector.use_llm_for_ambiguous})")
    except Exception as e:
        logger.error(f"Failed to initialize Phi-4 detector: {e}")
        logger.info("Falling back to pattern-based detection only")
        detector = Phi4MiniBoundaryDetector(use_llm_for_ambiguous=False)
    
    # Process the document
    try:
        # Extract pages using Docling
        logger.info("Extracting document content...")
        pages = await processor.process_document(pdf_path)
        logger.info(f"Extracted {len(pages)} pages")
        
        # Detect boundaries
        logger.info("Detecting document boundaries...")
        boundaries = detector.detect_boundaries(pages)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"BOUNDARY DETECTION RESULTS")
        print(f"{'='*60}")
        print(f"Total pages: {len(pages)}")
        print(f"Documents found: {len(boundaries)}")
        print(f"LLM analysis: {'Enabled' if detector.use_llm_for_ambiguous else 'Disabled'}")
        print(f"{'='*60}\n")
        
        for i, boundary in enumerate(boundaries, 1):
            print(f"Document {i}:")
            print(f"  Pages: {boundary.page_range}")
            print(f"  Type: {boundary.document_type.value if boundary.document_type else 'Unknown'}")
            print(f"  Confidence: {boundary.confidence:.0%}")
            print(f"  Signals detected:")
            
            for signal in boundary.signals:
                print(f"    - {signal.type.value}: {signal.description} "
                      f"(confidence: {signal.confidence:.0%})")
            print()
        
        # Demonstrate boundary explanation feature
        if detector.client and len(boundaries) > 1:
            print(f"\n{'='*60}")
            print("BOUNDARY EXPLANATIONS")
            print(f"{'='*60}\n")
            
            # Explain a few boundaries
            for boundary in boundaries[:3]:
                explanation = detector.explain_boundary(pages, boundary.start_page)
                if explanation:
                    print(explanation)
                    print("-" * 40)
        
        return boundaries
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise


async def compare_detection_methods(pdf_path: str):
    """Compare detection with and without LLM enhancement."""
    
    print("\n" + "="*80)
    print("COMPARING DETECTION METHODS")
    print("="*80)
    
    # Process without LLM
    print("\n1. Pattern-based detection only:")
    print("-" * 40)
    boundaries_no_llm = await process_with_phi4(pdf_path, use_llm=False)
    
    # Process with LLM
    print("\n2. Pattern + LLM detection:")
    print("-" * 40)
    boundaries_with_llm = await process_with_phi4(pdf_path, use_llm=True)
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Pattern-based: {len(boundaries_no_llm)} documents found")
    print(f"Pattern + LLM: {len(boundaries_with_llm)} documents found")
    
    if len(boundaries_with_llm) != len(boundaries_no_llm):
        print("\nDifferences detected! The LLM analysis found different boundaries.")


def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demo of Phi-4 Mini boundary detection"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to PDF file to process"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM analysis and use only pattern matching"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare results with and without LLM"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.pdf_path).exists():
        print(f"Error: File not found: {args.pdf_path}")
        sys.exit(1)
    
    # Run the appropriate demo
    if args.compare:
        asyncio.run(compare_detection_methods(args.pdf_path))
    else:
        asyncio.run(process_with_phi4(args.pdf_path, use_llm=not args.no_llm))


if __name__ == "__main__":
    main()