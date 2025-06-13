"""
Quick comparison test of enhanced detection methods using mock data.
This avoids slow OCR processing while demonstrating the different detection approaches.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import pytest

from src.core.models import PageInfo, Boundary, DocumentType
from src.core.boundary_detector import BoundaryDetector
from src.core.visual_boundary_detector import VisualBoundaryDetector
from src.core.phi4_mini_detector import Phi4MiniBoundaryDetector
from src.core.hybrid_boundary_detector import HybridBoundaryDetector


class MockDataGenerator:
    """Generate mock page data based on ground truth."""
    
    def __init__(self, ground_truth_path: str):
        with open(ground_truth_path, 'r') as f:
            self.ground_truth = json.load(f)
    
    def generate_pages(self) -> List[PageInfo]:
        """Generate mock pages with text and visual features."""
        pages = []
        
        for doc in self.ground_truth['documents']:
            # Parse page range
            pages_str = doc['pages']
            if '-' in pages_str:
                start, end = map(int, pages_str.split('-'))
            else:
                start = end = int(pages_str)
            
            doc_type = doc['type'].lower()
            
            for page_num in range(start, end + 1):
                # Create appropriate content based on document type and position
                if page_num == start:  # First page of document
                    text, layout = self._create_first_page_content(doc_type, doc['summary'])
                else:
                    text, layout = self._create_continuation_content(doc_type, page_num - start + 1, end - start + 1)
                
                # Add visual features
                visual_features = self._create_visual_features(doc_type, page_num == start)
                
                pages.append(PageInfo(
                    page_number=page_num,
                    width=612,  # 8.5 inches in points
                    height=792,  # 11 inches in points
                    text_content=text,
                    word_count=len(text.split()),
                    layout_elements=layout,
                    has_images='invoice' in doc_type or 'plans' in doc_type,
                    has_tables='schedule' in doc_type or 'invoice' in doc_type or 'application' in doc_type,
                    **visual_features
                ))
        
        return pages
    
    def _create_first_page_content(self, doc_type: str, summary: str) -> tuple:
        """Create first page content and layout."""
        layouts = []
        
        if 'email' in doc_type:
            text = f"From: sender@example.com\nTo: recipient@example.com\nSubject: Important Document\nDate: March 1, 2024\n\n{summary[:200]}"
            layouts = [
                {"type": "text", "bbox": [50, 50, 550, 100], "style": "header"},
                {"type": "text", "bbox": [50, 120, 550, 700], "style": "body"}
            ]
        elif 'invoice' in doc_type:
            text = f"INVOICE #12345\nDate: March 1, 2024\n\nBill To:\nCustomer Name\n123 Main Street\n\n{summary[:150]}"
            layouts = [
                {"type": "text", "bbox": [50, 50, 300, 100], "style": "title", "font_size": 18},
                {"type": "table", "bbox": [50, 200, 550, 400]},
                {"type": "logo", "bbox": [450, 50, 550, 150]}
            ]
        elif 'submittal' in doc_type:
            text = f"SUBMITTAL #0007\nProject: Construction Project\nDate: 2/29/2024\n\n{summary[:150]}"
            layouts = [
                {"type": "text", "bbox": [50, 50, 550, 100], "style": "header", "font_size": 16},
                {"type": "text", "bbox": [50, 150, 550, 700], "style": "body"}
            ]
        elif 'request for information' in doc_type:
            text = f"REQUEST FOR INFORMATION (RFI) #007\nProject: Construction\nDate: March 28, 2024\n\n{summary[:150]}"
            layouts = [
                {"type": "text", "bbox": [50, 50, 550, 100], "style": "header", "font_size": 14},
                {"type": "form", "bbox": [50, 150, 550, 700]}
            ]
        elif 'cost proposal' in doc_type:
            text = f"COST PROPOSAL #2\nDate: 4/5/2024\n\nProposed Costs:\n{summary[:150]}"
            layouts = [
                {"type": "text", "bbox": [50, 50, 550, 100], "style": "title"},
                {"type": "table", "bbox": [50, 150, 550, 400]}
            ]
        elif 'plans' in doc_type:
            text = f"STRUCTURAL ENGINEERING DRAWINGS\nSheet SI-1.1\n\n{summary[:100]}"
            layouts = [
                {"type": "text", "bbox": [50, 50, 550, 100], "style": "title"},
                {"type": "drawing", "bbox": [50, 150, 550, 700]}
            ]
        else:
            text = f"{doc_type.upper()}\n\n{summary[:200]}"
            layouts = [{"type": "text", "bbox": [50, 50, 550, 700], "style": "body"}]
        
        # Ensure sufficient text
        text += "\n\n" + "Additional content. " * 30
        
        return text, layouts
    
    def _create_continuation_content(self, doc_type: str, page_num: int, total_pages: int) -> tuple:
        """Create continuation page content."""
        text = f"Page {page_num} of {total_pages}\n\nContinuation of {doc_type}.\n\n"
        text += "Content continues from previous page. " * 40
        
        layouts = [
            {"type": "text", "bbox": [50, 50, 550, 100], "style": "header", "font_size": 10},
            {"type": "text", "bbox": [50, 100, 550, 700], "style": "body"}
        ]
        
        return text, layouts
    
    def _create_visual_features(self, doc_type: str, is_first_page: bool) -> Dict[str, Any]:
        """Create visual features for a page."""
        features = {}
        
        if 'email' in doc_type:
            features['dominant_font_size'] = 11
            features['primary_color'] = '#000000'
            features['background_color'] = '#FFFFFF'
            features['column_count'] = 1
        elif 'invoice' in doc_type:
            features['dominant_font_size'] = 10
            features['has_logo'] = is_first_page
            features['column_count'] = 2 if is_first_page else 1
        elif 'plans' in doc_type:
            features['orientation'] = 'landscape'
            features['has_drawings'] = True
        
        return features


class QuickDetectionComparison:
    """Quick comparison of detection methods."""
    
    def evaluate_boundaries(self, ground_truth: Dict, boundaries: List[Boundary]) -> Dict[str, Any]:
        """Evaluate detected boundaries against ground truth."""
        # Extract expected boundaries
        expected = []
        for i, doc in enumerate(ground_truth['documents']):
            if i > 0:  # Boundary before each doc except first
                pages = doc['pages']
                start = int(pages.split('-')[0] if '-' in pages else pages)
                expected.append(start)
        
        # Extract detected boundaries
        detected = [b.start_page for b in boundaries[1:]]  # Skip first
        
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
            "expected_count": len(ground_truth['documents']),
            "tp": tp, "fp": fp, "fn": fn,
            "missed": [e for e in expected if e not in detected],
            "extra": [d for d in detected if d not in expected]
        }


def test_detection_method_comparison():
    """Compare all detection methods using mock data."""
    # Load ground truth
    ground_truth_path = Path(__file__).parent.parent.parent / "tests" / "test_files" / "Test_PDF_Set_Ground_Truth.json"
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Generate mock pages
    generator = MockDataGenerator(str(ground_truth_path))
    pages = generator.generate_pages()
    
    print(f"\n=== Detection Method Comparison (Mock Data) ===")
    print(f"Total pages: {len(pages)}")
    print(f"Expected documents: {len(ground_truth['documents'])}")
    
    evaluator = QuickDetectionComparison()
    results = []
    
    # 1. Text-based detection
    print("\n1. Text-based detection...")
    start = time.time()
    text_detector = BoundaryDetector(min_confidence=0.5, min_signals=1)
    text_boundaries = text_detector.detect_boundaries(pages)
    text_time = time.time() - start
    text_eval = evaluator.evaluate_boundaries(ground_truth, text_boundaries)
    text_eval['method'] = 'Text-Based'
    text_eval['time'] = text_time
    results.append(text_eval)
    
    # 2. Visual detection
    print("2. Visual detection...")
    start = time.time()
    visual_detector = VisualBoundaryDetector()
    # Visual detector returns candidates, we need to convert them
    visual_candidates = visual_detector.detect_visual_boundaries(pages)
    # Convert candidates to boundaries format expected by evaluator
    from src.core.models import Boundary, Signal, SignalType
    visual_boundaries = []
    for i, candidate in enumerate(visual_candidates):
        # Convert visual signals to regular signals
        signals = []
        for vs in candidate.visual_signals:
            signals.append(Signal(
                type=SignalType.VISUAL_CHANGE,  # Generic visual signal type
                confidence=vs.confidence,
                details={"visual_type": vs.type.value, "description": vs.description}
            ))
        
        # Determine end page (assume single page documents for visual-only detection)
        end_page = candidate.page_number
        if i < len(visual_candidates) - 1:
            end_page = visual_candidates[i + 1].page_number - 1
        else:
            end_page = len(pages)
        
        visual_boundaries.append(Boundary(
            start_page=candidate.page_number,
            end_page=end_page,
            confidence=candidate.confidence,
            signals=signals,
            document_type=None  # Visual detector doesn't determine doc type
        ))
    
    # Add initial boundary if missing
    if not visual_boundaries or visual_boundaries[0].start_page > 1:
        visual_boundaries.insert(0, Boundary(
            start_page=1,
            end_page=visual_boundaries[0].start_page - 1 if visual_boundaries else len(pages),
            confidence=1.0,
            signals=[],
            document_type=None
        ))
    
    visual_time = time.time() - start
    visual_eval = evaluator.evaluate_boundaries(ground_truth, visual_boundaries)
    visual_eval['method'] = 'Visual-Only'
    visual_eval['time'] = visual_time
    results.append(visual_eval)
    
    # 3. Hybrid detection
    print("3. Hybrid detection...")
    start = time.time()
    hybrid_detector = HybridBoundaryDetector(text_weight=0.6, visual_weight=0.4)
    hybrid_boundaries = hybrid_detector.detect_boundaries(pages)
    hybrid_time = time.time() - start
    hybrid_eval = evaluator.evaluate_boundaries(ground_truth, hybrid_boundaries)
    hybrid_eval['method'] = 'Hybrid'
    hybrid_eval['time'] = hybrid_time
    results.append(hybrid_eval)
    
    # 4. LLM detection (if available)
    try:
        print("4. LLM-enhanced detection...")
        llm_detector = Phi4MiniBoundaryDetector(
            ollama_base_url="http://localhost:11434",
            model_name="phi4-mini:3.8b"
        )
        if llm_detector._check_ollama_available():
            start = time.time()
            llm_boundaries = llm_detector.detect_boundaries(pages)
            llm_time = time.time() - start
            llm_eval = evaluator.evaluate_boundaries(ground_truth, llm_boundaries)
            llm_eval['method'] = 'LLM-Enhanced'
            llm_eval['time'] = llm_time
            results.append(llm_eval)
        else:
            print("   Skipped - Ollama not available")
    except Exception as e:
        print(f"   Skipped - {e}")
    
    # Print results table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Method':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Docs':<10} {'Time(s)':<10}")
    print("-"*80)
    
    for r in results:
        print(f"{r['method']:<15} {r['precision']:<10.2f} {r['recall']:<10.2f} "
              f"{r['f1_score']:<10.2f} {r['doc_count']:>2}/{r['expected_count']:<7} "
              f"{r['time']:<10.2f}")
    
    # Detailed analysis
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    for r in results:
        print(f"\n{r['method']}:")
        print(f"  - True Positives: {r['tp']}")
        print(f"  - False Positives: {r['fp']} (pages: {r['extra']})")
        print(f"  - False Negatives: {r['fn']} (pages: {r['missed']})")
    
    # Best method
    best = max(results, key=lambda x: x['f1_score'])
    print(f"\n🏆 Best Method: {best['method']} with F1 score of {best['f1_score']:.2f}")
    
    # Verify at least one method works well
    assert any(r['f1_score'] > 0.7 for r in results), "No method achieved F1 > 0.7"
    assert any(r['precision'] > 0.8 for r in results), "No method achieved precision > 0.8"


if __name__ == "__main__":
    test_detection_method_comparison()