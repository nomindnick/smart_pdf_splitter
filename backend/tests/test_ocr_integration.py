"""Integration test for OCR improvements on Test_PDF_Set_1.pdf."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
import pytest
from datetime import datetime

from src.core.unified_document_processor import UnifiedDocumentProcessor, ProcessingMode
from src.core.ocr_config import OCRConfig


class TestOCRIntegration:
    """Test OCR improvements with real PDFs."""
    
    @pytest.fixture
    def test_pdf_path(self):
        """Get path to test PDF."""
        return Path(__file__).parent.parent.parent / "tests/test_files/Test_PDF_Set_1.pdf"
    
    @pytest.fixture
    def ground_truth_path(self):
        """Get path to ground truth JSON."""
        return Path(__file__).parent.parent.parent / "tests/test_files/Test_PDF_Set_Ground_Truth.json"
    
    def test_enhanced_processor_performance(self, test_pdf_path, ground_truth_path):
        """Test enhanced processor performance on Test_PDF_Set_1.pdf."""
        assert test_pdf_path.exists(), f"Test PDF not found: {test_pdf_path}"
        
        # Load ground truth
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)
        
        # Initialize unified processor in smart mode with page limit
        processor = UnifiedDocumentProcessor(
            mode=ProcessingMode.SMART,
            enable_adaptive=True,
            language="en",
            max_ocr_pages=5  # Limit OCR to 5 pages for speed
        )
        
        # Process document
        print("\n=== Processing with Unified Document Processor (Enhanced Mode) ===")
        start_time = time.time()
        
        def progress_callback(progress, message):
            print(f"Progress: {progress:.1f}% - {message}")
        
        document = processor.process_document(
            test_pdf_path,
            progress_callback=progress_callback,
            return_quality_report=True
        )
        
        processing_time = time.time() - start_time
        
        # Print results
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        print(f"Total pages: {document.total_pages}")
        
        # Get processing stats
        stats = processor.get_processing_stats()
        print("\n=== Processing Statistics ===")
        print(f"Pages processed: {stats['pages_processed']}")
        print(f"Pages preprocessed: {stats['pages_preprocessed']}")
        print(f"OCR performed: {stats['ocr_performed']}")
        print(f"Average confidence: {stats['average_confidence']:.2%}")
        print(f"Total corrections: {stats['total_corrections']}")
        
        # Check metadata
        if document.metadata and document.metadata.custom_fields:
            print("\n=== Document Metadata ===")
            print(f"Overall confidence: {document.metadata.custom_fields.get('overall_confidence', 0):.2%}")
            print(f"Quality assessment: {document.metadata.custom_fields.get('quality_assessment', 'N/A')}")
            
            # Check page quality reports
            if "page_quality_reports" in document.metadata.custom_fields:
                reports = document.metadata.custom_fields["page_quality_reports"]
                
                # Find pages with lowest confidence
                low_confidence_pages = []
                for report in reports:
                    confidence = report.get("confidence", {}).get("overall_confidence", 1.0)
                    if confidence < 0.7:
                        low_confidence_pages.append({
                            "page": report["page_number"],
                            "confidence": confidence,
                            "issues": report.get("confidence", {}).get("issues", [])
                        })
                
                if low_confidence_pages:
                    print(f"\n=== Low Confidence Pages ({len(low_confidence_pages)}) ===")
                    for page_info in sorted(low_confidence_pages, key=lambda x: x["confidence"])[:5]:
                        print(f"Page {page_info['page']}: {page_info['confidence']:.2%} - Issues: {', '.join(page_info['issues'])}")
        
        # Check individual page info
        print("\n=== Sample Page Analysis ===")
        for i in [0, 5, 10, 15, 20]:  # Sample pages
            if i < len(document.page_info):
                page = document.page_info[i]
                print(f"\nPage {page.page_number}:")
                print(f"  Word count: {page.word_count}")
                print(f"  Has images: {page.has_images}")
                print(f"  OCR confidence: {page.ocr_confidence:.2%}" if page.ocr_confidence else "  OCR confidence: N/A")
                print(f"  OCR quality: {page.ocr_quality_assessment}" if page.ocr_quality_assessment else "  OCR quality: N/A")
                print(f"  Preprocessing: {', '.join(page.preprocessing_applied)}" if page.preprocessing_applied else "  Preprocessing: None")
                print(f"  Corrections: {page.corrections_made}")
                if page.needs_review:
                    print(f"  ⚠️  Needs manual review")
        
        # Basic assertions
        assert document.total_pages == 36  # Ground truth says 36 pages
        assert stats['pages_processed'] == 36
        assert processing_time < 120  # Should complete in under 2 minutes
        
        # Save results for comparison
        results = {
            "processing_time": processing_time,
            "stats": stats,
            "overall_confidence": document.metadata.custom_fields.get('overall_confidence', 0) if document.metadata else 0,
            "quality_assessment": document.metadata.custom_fields.get('quality_assessment', 'N/A') if document.metadata else 'N/A',
            "timestamp": datetime.now().isoformat()
        }
        
        results_path = Path(__file__).parent / "ocr_integration_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {results_path}")
    
    def test_compare_with_base_processor(self, test_pdf_path):
        """Compare enhanced processor with base processor."""
        print("\n=== Comparing Base vs Enhanced Processor ===")
        
        # Process with basic mode
        print("\n1. Processing with Basic Mode...")
        basic_processor = UnifiedDocumentProcessor(
            mode=ProcessingMode.BASIC,
            enable_adaptive=False,
            language="en"
        )
        
        base_start = time.time()
        base_doc = basic_processor.process_document(test_pdf_path)
        base_time = time.time() - base_start
        
        # Process with smart mode (limited OCR for speed)
        print("\n2. Processing with Smart Mode...")
        smart_processor = UnifiedDocumentProcessor(
            mode=ProcessingMode.SMART,
            enable_adaptive=False,
            language="en",
            max_ocr_pages=3  # Only OCR 3 pages for speed
        )
        
        smart_start = time.time()
        smart_doc = smart_processor.process_document(test_pdf_path, return_quality_report=False)
        smart_time = time.time() - smart_start
        
        # Compare results
        print("\n=== Comparison Results ===")
        print(f"Base processor time: {base_time:.2f}s")
        print(f"Smart processor time: {smart_time:.2f}s")
        print(f"Time difference: {smart_time - base_time:.2f}s ({((smart_time/base_time - 1) * 100):.1f}%)")
        
        # Compare text extraction
        base_text_length = sum(len(p.text_content or "") for p in base_doc.page_info)
        smart_text_length = sum(len(p.text_content or "") for p in smart_doc.page_info)
        
        print(f"\nBase text extracted: {base_text_length} characters")
        print(f"Smart text extracted: {smart_text_length} characters")
        print(f"Text difference: {smart_text_length - base_text_length} characters")
        
        # Check quality metrics (only available in smart)
        smart_stats = smart_processor.get_processing_stats()
        print(f"\nSmart processor stats:")
        print(f"  OCR performed: {smart_stats['ocr_performed']} pages")
        print(f"  Average confidence: {smart_stats['average_confidence']:.2%}")
        print(f"  Total corrections: {smart_stats['total_corrections']}")