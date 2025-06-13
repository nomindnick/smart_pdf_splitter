"""
Test all enhanced detection methods against ground truth.
Compares text-based, visual, LLM-based, and hybrid detection.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

import pytest

from src.core.models import PageInfo, Boundary
from src.core.boundary_detector import BoundaryDetector
from src.core.visual_boundary_detector import VisualBoundaryDetector
from src.core.phi4_mini_detector import Phi4MiniBoundaryDetector
from src.core.hybrid_boundary_detector import HybridBoundaryDetector
from src.core.document_processor import DocumentProcessor


class EnhancedDetectionEvaluator:
    """Evaluates different detection methods against ground truth."""
    
    def __init__(self, ground_truth_path: str):
        with open(ground_truth_path, 'r') as f:
            self.ground_truth = json.load(f)
        self.expected_documents = self.ground_truth["documents"]
        self._parse_page_ranges()
        self.expected_boundaries = self._extract_boundaries()
    
    def _parse_page_ranges(self):
        """Parse page ranges from 'pages' field to start_page and end_page."""
        for doc in self.expected_documents:
            pages = doc["pages"]
            if "-" in pages:
                start, end = pages.split("-")
                doc["start_page"] = int(start)
                doc["end_page"] = int(end)
            else:
                doc["start_page"] = int(pages)
                doc["end_page"] = int(pages)
            doc["document_type"] = doc.get("type", "OTHER")
    
    def _extract_boundaries(self) -> List[int]:
        """Extract boundary positions from ground truth."""
        boundaries = []
        for i, doc in enumerate(self.expected_documents):
            if i > 0:  # Add boundary before each document except first
                boundaries.append(doc["start_page"])
        return sorted(boundaries)
    
    def evaluate_method(self, method_name: str, boundaries: List[Boundary]) -> Dict[str, Any]:
        """Evaluate a detection method's results."""
        start_time = time.time()
        
        # Extract start pages from boundaries
        detected_positions = []
        for i, boundary in enumerate(boundaries):
            if i > 0:  # Skip the first boundary which is always page 1
                detected_positions.append(boundary.start_page)
        
        # Calculate metrics
        true_positives = len([p for p in detected_positions if p in self.expected_boundaries])
        false_positives = len([p for p in detected_positions if p not in self.expected_boundaries])
        false_negatives = len([p for p in self.expected_boundaries if p not in detected_positions])
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Document-level metrics
        expected_doc_count = len(self.expected_documents)
        detected_doc_count = len(boundaries)
        doc_count_accuracy = 1 - abs(detected_doc_count - expected_doc_count) / expected_doc_count
        
        # Per-document type accuracy
        doc_type_accuracy = self._calculate_doc_type_accuracy(boundaries)
        
        return {
            "method": method_name,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "expected_boundaries": self.expected_boundaries,
            "detected_boundaries": detected_positions,
            "expected_doc_count": expected_doc_count,
            "detected_doc_count": detected_doc_count,
            "doc_count_accuracy": doc_count_accuracy,
            "doc_type_accuracy": doc_type_accuracy,
            "missed_boundaries": [p for p in self.expected_boundaries if p not in detected_positions],
            "extra_boundaries": [p for p in detected_positions if p not in self.expected_boundaries],
            "processing_time": time.time() - start_time
        }
    
    def _calculate_doc_type_accuracy(self, boundaries: List[Boundary]) -> Dict[str, float]:
        """Calculate accuracy by document type."""
        doc_type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for i, doc in enumerate(self.expected_documents):
            doc_type = doc["document_type"]
            doc_start = doc["start_page"]
            
            if i > 0:  # Skip first document (no boundary before it)
                expected_boundary = doc_start
                detected = any(b.start_page == expected_boundary for b in boundaries[1:])
                
                doc_type_stats[doc_type]["total"] += 1
                if detected:
                    doc_type_stats[doc_type]["correct"] += 1
        
        # Convert to accuracy percentages
        return {
            doc_type: stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            for doc_type, stats in doc_type_stats.items()
        }


class TestEnhancedDetectionMethods:
    """Test all detection methods against ground truth."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator with ground truth data."""
        ground_truth_path = Path(__file__).parent.parent.parent / "tests" / "test_files" / "Test_PDF_Set_Ground_Truth.json"
        return EnhancedDetectionEvaluator(str(ground_truth_path))
    
    @pytest.fixture
    def test_pdf_path(self):
        """Get test PDF path."""
        return Path(__file__).parent.parent.parent / "tests" / "test_files" / "Test_PDF_Set_1.pdf"
    
    @pytest.fixture
    def processed_pages(self, test_pdf_path) -> Optional[List[PageInfo]]:
        """Process test PDF with OCR enabled (cached)."""
        print("\n=== Processing PDF with OCR (this may take several minutes) ===")
        
        try:
            # Try EasyOCR first (no additional installation required)
            processor = DocumentProcessor(enable_ocr=True, ocr_engine="easyocr")
            pages = list(processor.process_document(str(test_pdf_path)))
            
            if pages and any(p.text_content for p in pages):
                print(f"Successfully extracted {len(pages)} pages with EasyOCR")
                return pages
                
        except Exception as e:
            print(f"EasyOCR failed: {e}")
        
        # Fall back to no OCR for testing
        print("Falling back to no OCR - visual detection only")
        processor = DocumentProcessor(enable_ocr=False)
        pages = list(processor.process_document(str(test_pdf_path)))
        return pages if pages else None
    
    def test_all_detection_methods(self, evaluator, processed_pages):
        """Test all detection methods and compare results."""
        if not processed_pages:
            pytest.skip("Could not process PDF")
        
        print(f"\n=== Testing Detection Methods on {len(processed_pages)} Pages ===")
        
        results = []
        
        # 1. Text-based detection (baseline)
        print("\n1. Testing text-based detection...")
        text_detector = BoundaryDetector(
            min_confidence=0.5,
            min_signals=1,
            enable_visual_analysis=False
        )
        text_boundaries = text_detector.detect_boundaries(processed_pages)
        results.append(evaluator.evaluate_method("Text-Based", text_boundaries))
        
        # 2. Visual detection only
        print("\n2. Testing visual detection...")
        visual_detector = VisualBoundaryDetector(
            min_confidence=0.4,
            min_signals=1
        )
        visual_boundaries = visual_detector.detect_boundaries(processed_pages)
        results.append(evaluator.evaluate_method("Visual-Only", visual_boundaries))
        
        # 3. LLM-based detection (if Ollama available)
        print("\n3. Testing LLM-based detection...")
        try:
            llm_detector = Phi4MiniBoundaryDetector(
                ollama_base_url="http://localhost:11434",
                model_name="phi4-mini:3.8b",
                min_confidence=0.5,
                min_signals=1
            )
            # Check if Ollama is available
            if llm_detector._check_ollama_available():
                llm_boundaries = llm_detector.detect_boundaries(processed_pages)
                results.append(evaluator.evaluate_method("LLM-Enhanced", llm_boundaries))
            else:
                print("   Skipping - Ollama not available")
        except Exception as e:
            print(f"   Skipping - LLM detection failed: {e}")
        
        # 4. Hybrid detection (text + visual)
        print("\n4. Testing hybrid detection...")
        hybrid_detector = HybridBoundaryDetector(
            text_weight=0.6,
            visual_weight=0.4,
            min_confidence=0.5
        )
        hybrid_boundaries = hybrid_detector.detect_boundaries(processed_pages)
        results.append(evaluator.evaluate_method("Hybrid (Text+Visual)", hybrid_boundaries))
        
        # Print comprehensive results
        self._print_results_table(results)
        self._print_document_type_accuracy(results)
        self._print_detailed_analysis(results, processed_pages)
    
    def _print_results_table(self, results: List[Dict[str, Any]]):
        """Print results in a formatted table."""
        print("\n" + "="*100)
        print("DETECTION METHOD COMPARISON")
        print("="*100)
        print(f"{'Method':<20} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Docs Found':<12} {'Time (s)':<10}")
        print("-"*100)
        
        for r in results:
            print(f"{r['method']:<20} {r['precision']:<10.2f} {r['recall']:<10.2f} "
                  f"{r['f1_score']:<10.2f} {r['detected_doc_count']:>3}/{r['expected_doc_count']:<8} "
                  f"{r['processing_time']:<10.2f}")
        
        print("-"*100)
    
    def _print_document_type_accuracy(self, results: List[Dict[str, Any]]):
        """Print accuracy by document type."""
        print("\n" + "="*100)
        print("ACCURACY BY DOCUMENT TYPE")
        print("="*100)
        
        # Collect all document types
        all_types = set()
        for r in results:
            all_types.update(r['doc_type_accuracy'].keys())
        
        # Header
        print(f"{'Document Type':<25}", end="")
        for r in results:
            print(f"{r['method']:<20}", end="")
        print()
        print("-"*100)
        
        # Data rows
        for doc_type in sorted(all_types):
            print(f"{doc_type:<25}", end="")
            for r in results:
                accuracy = r['doc_type_accuracy'].get(doc_type, 0)
                print(f"{accuracy:<20.2f}", end="")
            print()
        
        print("-"*100)
    
    def _print_detailed_analysis(self, results: List[Dict[str, Any]], pages: List[PageInfo]):
        """Print detailed analysis of detection results."""
        print("\n" + "="*100)
        print("DETAILED ANALYSIS")
        print("="*100)
        
        for r in results:
            print(f"\n{r['method']} Method:")
            print(f"  - Missed boundaries: {r['missed_boundaries']}")
            print(f"  - Extra boundaries: {r['extra_boundaries']}")
            
            # Analyze why boundaries were missed/added
            if r['missed_boundaries']:
                print(f"  - Analysis: May have missed subtle document transitions")
            if r['extra_boundaries']:
                print(f"  - Analysis: May be over-sensitive to formatting changes")
        
        # Text content analysis
        total_text = sum(len(p.text_content or "") for p in pages)
        print(f"\n\nPDF Analysis:")
        print(f"  - Total pages: {len(pages)}")
        print(f"  - Total text extracted: {total_text} characters")
        print(f"  - Average text per page: {total_text/len(pages):.0f} characters")
        
        if total_text == 0:
            print("  - WARNING: No text extracted - PDF appears to be scanned without OCR")
            print("  - Only visual detection methods will be effective")


if __name__ == "__main__":
    # Run with extended timeout
    pytest.main([__file__, "-v", "-s", "--timeout=600"])