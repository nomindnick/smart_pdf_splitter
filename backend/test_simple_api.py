#!/usr/bin/env python3
"""Simple test to verify API functionality."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api.services.document_service import DocumentService
from src.core.models import Document, ProcessingStatus
from pathlib import Path
import asyncio

async def test_document_service():
    """Test the document service directly."""
    
    # Find test file
    test_file = Path(__file__).parent.parent / "tests" / "test_files" / "Test_PDF_Set_1.pdf"
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return
    
    print(f"Using test file: {test_file}")
    
    # Test configurations
    configs = [
        {
            "name": "Basic (text only)",
            "visual": False,
            "llm": False,
            "intelligent_ocr": False
        },
        {
            "name": "Visual detection",
            "visual": True,
            "llm": False,
            "intelligent_ocr": False
        },
        {
            "name": "Intelligent OCR",
            "visual": True,
            "llm": False,
            "intelligent_ocr": True
        }
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"{'='*60}")
        
        try:
            # Create service
            service = DocumentService(
                enable_visual_detection=config['visual'],
                enable_llm_detection=config['llm'],
                enable_intelligent_ocr=config['intelligent_ocr']
            )
            
            # Create test document
            document = Document(
                id="test-123",
                filename="test.pdf",
                status=ProcessingStatus.PENDING,
                total_pages=1,
                file_size=1000,
                original_path=str(test_file)
            )
            
            # Process
            result = await service.process_document(document)
            
            print(f"Status: {result.status}")
            print(f"Total pages: {result.total_pages}")
            print(f"Boundaries detected: {len(result.detected_boundaries) if result.detected_boundaries else 0}")
            
            if result.detected_boundaries:
                for i, boundary in enumerate(result.detected_boundaries[:5]):
                    print(f"\nBoundary {i+1}:")
                    print(f"  Pages: {boundary.start_page}-{boundary.end_page}")
                    print(f"  Confidence: {boundary.confidence:.2%}")
                    print(f"  Type: {boundary.document_type.value if boundary.document_type else 'unknown'}")
                    print(f"  Signals: {len(boundary.signals)}")
            
            print(f"\nProcessing time: {result.processing_time:.2f}s")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("Direct Document Service Test")
    print("===========================\n")
    
    asyncio.run(test_document_service())