"""
Simple demonstration of enhanced detection methods.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any

from src.core.models import PageInfo, Boundary
from src.core.boundary_detector import BoundaryDetector
from src.core.phi4_mini_detector import Phi4MiniBoundaryDetector
from src.core.hybrid_boundary_detector import HybridBoundaryDetector


def create_test_pages() -> List[PageInfo]:
    """Create test pages based on ground truth data."""
    ground_truth_path = Path(__file__).parent.parent.parent / "tests" / "test_files" / "Test_PDF_Set_Ground_Truth.json"
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    pages = []
    for doc in ground_truth['documents']:
        # Parse page range
        pages_str = doc['pages']
        if '-' in pages_str:
            start, end = map(int, pages_str.split('-'))
        else:
            start = end = int(pages_str)
        
        doc_type = doc['type'].lower()
        
        for page_num in range(start, end + 1):
            # Create realistic content based on document type
            if page_num == start:  # First page
                if 'email' in doc_type:
                    text = f"From: sender@example.com\nTo: recipient@example.com\nSubject: {doc['summary'][:50]}\nDate: March 1, 2024\n\n{doc['summary']}"
                elif 'invoice' in doc_type:
                    text = f"INVOICE #12345\nDate: March 1, 2024\n\nBill To:\nCustomer Name\n\n{doc['summary']}"
                elif 'submittal' in doc_type:
                    text = f"SUBMITTAL #0007\nProject: Construction\nDate: 2/29/2024\n\n{doc['summary']}"
                elif 'schedule of values' in doc_type:
                    text = f"SCHEDULE OF VALUES\nContract: $1,459,395.00\n\n{doc['summary']}"
                elif 'application for payment' in doc_type:
                    text = f"APPLICATION FOR PAYMENT\nPeriod: 1/31/2024\n\n{doc['summary']}"
                elif 'request for information' in doc_type:
                    text = f"REQUEST FOR INFORMATION (RFI) #007\nProject: Construction\n\n{doc['summary']}"
                elif 'cost proposal' in doc_type:
                    text = f"COST PROPOSAL #2\nDate: 4/5/2024\n\n{doc['summary']}"
                elif 'plans' in doc_type:
                    text = f"STRUCTURAL ENGINEERING DRAWINGS\nSheet SI-1.1\n\n{doc['summary']}"
                else:
                    text = f"{doc_type.upper()}\n\n{doc['summary']}"
            else:  # Continuation pages
                text = f"Page {page_num - start + 1} of {end - start + 1}\n\nContinuation of {doc_type}.\n\n" + "Content continues... " * 50
            
            pages.append(PageInfo(
                page_number=page_num,
                width=612,
                height=792,
                text_content=text,
                word_count=len(text.split()),
                has_tables='schedule' in doc_type or 'invoice' in doc_type,
                has_images='plans' in doc_type
            ))
    
    return pages


def evaluate_results(ground_truth: Dict, boundaries: List[Boundary]) -> Dict[str, Any]:
    """Evaluate detection results."""
    # Extract expected boundaries
    expected = []
    for i, doc in enumerate(ground_truth['documents']):
        if i > 0:
            pages = doc['pages']
            start = int(pages.split('-')[0] if '-' in pages else pages)
            expected.append(start)
    
    # Extract detected boundaries
    detected = [b.start_page for b in boundaries[1:]]
    
    # Calculate metrics
    tp = len([d for d in detected if d in expected])
    fp = len([d for d in detected if d not in expected])
    fn = len([e for e in expected if e not in detected])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "doc_count": len(boundaries),
        "tp": tp, "fp": fp, "fn": fn,
        "missed": [e for e in expected if e not in detected],
        "extra": [d for d in detected if d not in expected]
    }


def main():
    """Run detection demonstration."""
    print("="*80)
    print("ENHANCED BOUNDARY DETECTION DEMONSTRATION")
    print("="*80)
    
    # Load ground truth
    ground_truth_path = Path(__file__).parent.parent.parent / "tests" / "test_files" / "Test_PDF_Set_Ground_Truth.json"
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Create test pages
    pages = create_test_pages()
    print(f"\nTest Data:")
    print(f"- Total pages: {len(pages)}")
    print(f"- Expected documents: {len(ground_truth['documents'])}")
    
    print("\n" + "-"*80)
    
    # Test different detection methods
    results = []
    
    # 1. Baseline text-based detection
    print("\n1. BASELINE TEXT-BASED DETECTION")
    print("-"*40)
    start = time.time()
    text_detector = BoundaryDetector(min_confidence=0.5, min_signals=1)
    text_boundaries = text_detector.detect_boundaries(pages)
    text_time = time.time() - start
    text_eval = evaluate_results(ground_truth, text_boundaries)
    
    print(f"Time: {text_time:.2f}s")
    print(f"Documents detected: {text_eval['doc_count']}/14")
    print(f"Precision: {text_eval['precision']:.2%}")
    print(f"Recall: {text_eval['recall']:.2%}")
    print(f"F1 Score: {text_eval['f1_score']:.2%}")
    if text_eval['missed']:
        print(f"Missed boundaries at pages: {text_eval['missed']}")
    if text_eval['extra']:
        print(f"Extra boundaries at pages: {text_eval['extra']}")
    
    # 2. Optimized text detection
    print("\n2. OPTIMIZED TEXT DETECTION")
    print("-"*40)
    start = time.time()
    opt_detector = BoundaryDetector(min_confidence=0.6, min_signals=2)
    # Adjust weights for construction documents
    from src.core.models import SignalType
    opt_detector.SIGNAL_WEIGHTS[SignalType.WHITE_SPACE] = 0.3
    opt_detector.SIGNAL_WEIGHTS[SignalType.DOCUMENT_HEADER] = 0.9
    
    opt_boundaries = opt_detector.detect_boundaries(pages)
    opt_time = time.time() - start
    opt_eval = evaluate_results(ground_truth, opt_boundaries)
    
    print(f"Time: {opt_time:.2f}s")
    print(f"Documents detected: {opt_eval['doc_count']}/14")
    print(f"Precision: {opt_eval['precision']:.2%}")
    print(f"Recall: {opt_eval['recall']:.2%}")
    print(f"F1 Score: {opt_eval['f1_score']:.2%}")
    if opt_eval['missed']:
        print(f"Missed boundaries at pages: {opt_eval['missed']}")
    if opt_eval['extra']:
        print(f"Extra boundaries at pages: {opt_eval['extra']}")
    
    # 3. LLM-enhanced detection (if available)
    try:
        print("\n3. LLM-ENHANCED DETECTION (Phi4-Mini)")
        print("-"*40)
        llm_detector = Phi4MiniBoundaryDetector(
            ollama_base_url="http://localhost:11434",
            model_name="phi4-mini:3.8b",
            min_confidence=0.5
        )
        
        if llm_detector._check_ollama_available():
            start = time.time()
            llm_boundaries = llm_detector.detect_boundaries(pages)
            llm_time = time.time() - start
            llm_eval = evaluate_results(ground_truth, llm_boundaries)
            
            print(f"Time: {llm_time:.2f}s")
            print(f"Documents detected: {llm_eval['doc_count']}/14")
            print(f"Precision: {llm_eval['precision']:.2%}")
            print(f"Recall: {llm_eval['recall']:.2%}")
            print(f"F1 Score: {llm_eval['f1_score']:.2%}")
            
            # Show LLM analysis for ambiguous cases
            ambiguous_count = sum(1 for b in llm_boundaries 
                                if any('llm_enhanced' in str(s.details) for s in b.signals))
            print(f"LLM analyzed {ambiguous_count} ambiguous boundaries")
            
            if llm_eval['missed']:
                print(f"Missed boundaries at pages: {llm_eval['missed']}")
            if llm_eval['extra']:
                print(f"Extra boundaries at pages: {llm_eval['extra']}")
        else:
            print("Skipped - Ollama not available")
            print("To enable LLM detection:")
            print("1. Install Ollama: https://ollama.ai")
            print("2. Pull model: ollama pull phi4-mini:3.8b")
            print("3. Ensure Ollama is running")
    except Exception as e:
        print(f"Skipped - Error: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nKey Findings:")
    print("- Text-based detection works well for emails and standard documents")
    print("- Construction documents (RFIs, submittals, cost proposals) need special handling")
    print("- Optimized weights improve precision but may reduce recall")
    print("- LLM enhancement can help with ambiguous boundaries")
    
    print("\nRecommendations:")
    print("1. Use hybrid approach combining multiple detection methods")
    print("2. Enable OCR for scanned PDFs to improve text-based detection")
    print("3. Fine-tune detection weights for specific document types")
    print("4. Consider LLM enhancement for critical applications")
    
    # Test with actual PDF data
    print("\n" + "="*80)
    print("ACTUAL PDF ANALYSIS")
    print("="*80)
    
    from src.core.document_processor import DocumentProcessor
    test_pdf_path = Path(__file__).parent.parent.parent / "tests" / "test_files" / "Test_PDF_Set_1.pdf"
    
    print(f"\nAnalyzing: {test_pdf_path.name}")
    print("Processing without OCR (faster but less accurate)...")
    
    processor = DocumentProcessor(enable_ocr=False)
    try:
        pdf_pages = list(processor.process_document(str(test_pdf_path)))
        print(f"Extracted {len(pdf_pages)} pages")
        
        total_text = sum(len(p.text_content or "") for p in pdf_pages)
        print(f"Total text extracted: {total_text} characters")
        
        if total_text == 0:
            print("\n⚠️  WARNING: No text extracted from PDF")
            print("This PDF appears to be scanned without embedded text.")
            print("For accurate detection, enable OCR:")
            print("  processor = DocumentProcessor(enable_ocr=True, ocr_engine='easyocr')")
        else:
            # Run detection on actual PDF
            detector = BoundaryDetector(min_confidence=0.5)
            boundaries = detector.detect_boundaries(pdf_pages)
            print(f"\nDetected {len(boundaries)} documents")
            
    except Exception as e:
        print(f"Error processing PDF: {e}")


if __name__ == "__main__":
    main()