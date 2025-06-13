#!/usr/bin/env python3
"""
Example usage of the pipeline architecture.

This demonstrates how the standalone splitter uses the pipeline
and how it can be extended for RAG in the future.
"""

import asyncio
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.processing_pipeline import SmartPDFPipeline, ProcessingResult, PostProcessor
from src.core.pipeline_config import PipelineProfiles, ProcessingStrategies
from src.core.parallel_processor import AdaptiveParallelProcessor
from src.api.services.document_service_v2 import DocumentServiceV2
from src.core.models import Document, ProcessingStatus


# Example 1: Standalone Splitter Usage
async def standalone_splitter_example():
    """Example of using the pipeline for the standalone splitter."""
    print("=== Standalone PDF Splitter Example ===\n")
    
    # Get optimized config for splitting
    detection_config = PipelineProfiles.get_splitter_detection_config()
    strategy = ProcessingStrategies.get_splitter_strategy()
    
    # Create pipeline
    pipeline = SmartPDFPipeline(detection_config=detection_config)
    
    # Process a PDF
    test_pdf = Path("../../tests/test_files/Test_PDF_Set_1.pdf")
    if test_pdf.exists():
        print(f"Processing: {test_pdf.name}")
        print(f"Strategy: {strategy['quality_target']} mode, max {strategy['max_ocr_pages']} OCR pages")
        
        def progress(pct, msg):
            print(f"  {pct:.0f}% - {msg}")
        
        result = pipeline.process_for_splitting(test_pdf, progress)
        
        print(f"\nResults:")
        print(f"  Processing time: {result.processing_time:.2f}s")
        print(f"  Total pages: {result.document.total_pages}")
        print(f"  Detection stats: {result.metadata['detection_stats']}")
    else:
        print("Test PDF not found!")


# Example 2: RAG Pipeline Extension (Future)
class RAGPostProcessor(PostProcessor):
    """Example post-processor for RAG pipeline."""
    
    async def process(self, result: ProcessingResult) -> None:
        """This would queue full OCR and embedding generation."""
        print(f"\n[RAG Post-Processor]")
        print(f"  Would queue document {result.document.id} for:")
        print(f"  - Full OCR extraction")
        print(f"  - Embedding generation")
        print(f"  - Vector DB storage")
        
        # In real implementation:
        # - Queue to Celery/Redis
        # - Generate embeddings with sentence-transformers
        # - Store in Pinecone/Weaviate/etc


async def future_rag_example():
    """Example of how the pipeline would be used in the RAG app."""
    print("\n\n=== Future RAG Pipeline Example ===\n")
    
    # Get high-quality config for RAG
    extraction_config = PipelineProfiles.get_rag_extraction_config()
    strategy = ProcessingStrategies.get_rag_strategy()
    
    # Create pipeline with RAG post-processors
    pipeline = SmartPDFPipeline(
        detection_config=PipelineProfiles.get_splitter_detection_config(),
        extraction_config=extraction_config
    )
    
    # Add RAG-specific processing
    pipeline.add_post_processor(RAGPostProcessor())
    
    print("This configuration would:")
    print(f"- Use {extraction_config.ocr_engine} at {extraction_config.target_dpi} DPI")
    print(f"- Process {strategy['page_selection']['sample_rate']*100}% of pages")
    print(f"- Use {strategy['parallel_workers']} parallel workers")
    print(f"- Apply preprocessing: {extraction_config.preprocessing_steps}")
    print(f"- Target quality: {strategy['quality_target']}")


# Example 3: Using DocumentServiceV2
async def document_service_example():
    """Example of using the enhanced document service."""
    print("\n\n=== Document Service V2 Example ===\n")
    
    # Create service
    service = DocumentServiceV2(
        enable_parallel_processing=True,
        queue_type="memory",  # Would be "celery" for RAG
        processing_mode="smart"
    )
    
    # Create a test document
    doc = Document(
        id="test-123",
        filename="test.pdf",
        total_pages=36,
        file_size=10_000_000,
        status=ProcessingStatus.PENDING,
        original_path="../../tests/test_files/Test_PDF_Set_1.pdf"
    )
    
    print("Document service features:")
    print("- Parallel OCR processing")
    print("- Task queue for async processing")
    print("- Hybrid boundary detection")
    print("- RAG export preparation")
    
    # Example: Prepare for RAG export
    # (In real usage, this would be after processing)
    doc.detected_boundaries = []  # Would be populated by processing
    rag_data = service.prepare_for_rag_export(doc)
    
    print(f"\nRAG export data structure:")
    print(f"- Document ID: {rag_data['document_id']}")
    print(f"- Boundaries: {len(rag_data['boundaries'])}")
    print(f"- OCR cache: {rag_data['ocr_cache'].keys()}")


# Example 4: Performance Comparison
async def performance_comparison():
    """Compare different processing strategies."""
    print("\n\n=== Performance Comparison ===\n")
    
    strategies = {
        "Splitter (Fast)": ProcessingStrategies.get_splitter_strategy(),
        "RAG (Quality)": ProcessingStrategies.get_rag_strategy()
    }
    
    print("Strategy Comparison:")
    print("-" * 60)
    print(f"{'Setting':<25} {'Splitter':<17} {'RAG':<17}")
    print("-" * 60)
    
    for key in ["max_ocr_pages", "parallel_workers", "quality_target"]:
        splitter_val = strategies["Splitter (Fast)"].get(key, "N/A")
        rag_val = strategies["RAG (Quality)"].get(key, "N/A")
        print(f"{key:<25} {str(splitter_val):<17} {str(rag_val):<17}")
    
    print("\nPage Selection:")
    for key in ["sample_rate", "max_consecutive_skip"]:
        splitter_val = strategies["Splitter (Fast)"]["page_selection"][key]
        rag_val = strategies["RAG (Quality)"]["page_selection"][key]
        print(f"  {key:<23} {str(splitter_val):<17} {str(rag_val):<17}")


async def main():
    """Run all examples."""
    await standalone_splitter_example()
    await future_rag_example()
    await document_service_example()
    await performance_comparison()
    
    print("\n\n=== Summary ===")
    print("The architecture provides:")
    print("1. Fast boundary detection for immediate splitting")
    print("2. Placeholder hooks for future RAG processing")
    print("3. Parallel processing capabilities")
    print("4. Flexible configuration profiles")
    print("5. Clear separation between splitter and RAG concerns")


if __name__ == "__main__":
    asyncio.run(main())