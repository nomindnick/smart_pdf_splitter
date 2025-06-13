"""
Comprehensive test for boundary detection against ground truth.
Tests multiple detection methods and compares accuracy.
"""

import json
import pytest
from pathlib import Path
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, patch

from backend.src.core.models import (
    PageInfo, Boundary, Document, DocumentType,
    ProcessingStatus, Signal, SignalType
)
from backend.src.core.boundary_detector import BoundaryDetector
from backend.src.core.document_processor import DocumentProcessor


class GroundTruthEvaluator:
    """Evaluates boundary detection accuracy against ground truth."""
    
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
            # Also normalize the type field
            doc["document_type"] = doc.get("type", "OTHER")
    
    def _extract_boundaries(self) -> List[int]:
        """Extract boundary positions from ground truth."""
        boundaries = []
        for i, doc in enumerate(self.expected_documents):
            if i > 0:  # Add boundary before each document except first
                boundaries.append(doc["start_page"])
        return sorted(boundaries)
    
    def evaluate_boundaries(self, detected_boundaries: List[Boundary]) -> Dict[str, Any]:
        """Evaluate detected boundaries against ground truth."""
        # Extract start pages from boundaries
        detected_positions = []
        for i, boundary in enumerate(detected_boundaries):
            if i > 0:  # Skip the first boundary which is always page 1
                detected_positions.append(boundary.start_page)
        
        # Calculate metrics
        true_positives = len([p for p in detected_positions if p in self.expected_boundaries])
        false_positives = len([p for p in detected_positions if p not in self.expected_boundaries])
        false_negatives = len([p for p in self.expected_boundaries if p not in detected_positions])
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Document count accuracy
        expected_doc_count = len(self.expected_documents)
        detected_doc_count = len(detected_boundaries)
        doc_count_accuracy = 1 - abs(detected_doc_count - expected_doc_count) / expected_doc_count
        
        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "expected_boundaries": self.expected_boundaries,
            "detected_boundaries": detected_positions,
            "expected_doc_count": expected_doc_count,
            "detected_doc_count": detected_doc_count,
            "doc_count_accuracy": doc_count_accuracy,
            "missed_boundaries": [p for p in self.expected_boundaries if p not in detected_positions],
            "extra_boundaries": [p for p in detected_positions if p not in self.expected_boundaries]
        }
    
    def create_mock_pages(self) -> List[PageInfo]:
        """Create mock pages based on ground truth for testing."""
        pages = []
        
        for doc in self.expected_documents:
            doc_type = doc["document_type"].lower()
            start_page = doc["start_page"]
            end_page = doc["end_page"]
            
            for page_num in range(start_page, end_page + 1):
                # Create appropriate content based on document type
                if page_num == start_page:
                    content = self._generate_first_page_content(doc_type, doc)
                else:
                    content = self._generate_continuation_content(doc_type, page_num - start_page + 1)
                
                pages.append(PageInfo(
                    page_number=page_num,
                    width=8.5 * 72,  # 8.5 inches in points
                    height=11 * 72,  # 11 inches in points
                    text_content=content,
                    word_count=len(content.split()),
                    has_tables="table" in content.lower() or "schedule" in content.lower()
                ))
        
        return pages
    
    def _generate_first_page_content(self, doc_type: str, doc_info: Dict) -> str:
        """Generate realistic first page content based on document type."""
        content_templates = {
            "email": """From: sender@example.com
To: recipient@example.com
Subject: Important Document
Date: January 1, 2024

Dear Recipient,

This is the beginning of an email document. It contains important information
that spans multiple lines and paragraphs.

The email discusses various topics and includes relevant details about the
subject matter at hand.""",
            
            "email chain": """From: person1@example.com
To: person2@example.com
Subject: RE: Project Discussion
Date: January 2, 2024

Thanks for your response. Please see below.

On January 1, 2024, person2@example.com wrote:
> Original message content here
> Multiple lines of discussion""",
            
            "invoice": """INVOICE #12345
Date: January 3, 2024
Due Date: February 3, 2024

Bill To:
Customer Name
123 Main Street
City, State 12345

Ship To:
Same as billing

Description of services provided with multiple line items
Service 1 - $100.00
Service 2 - $200.00

Total Amount Due: $300.00""",
            
            "submittal": """SUBMITTAL #001
Project: Construction Project
Date: January 4, 2024

Submitted for approval:
- Item 1: Technical specifications
- Item 2: Product data sheets
- Item 3: Installation instructions

Please review and provide approval or comments.""",
            
            "cost proposal": """COST PROPOSAL
Project: New Development
Date: January 5, 2024

Proposed costs for the following scope of work:
1. Phase 1: $50,000
2. Phase 2: $75,000
3. Phase 3: $25,000

Total Proposed Cost: $150,000""",
            
            "schedule of values": """SCHEDULE OF VALUES
Project: Building Renovation
Date: January 6, 2024

Line Item | Description | Value
1 | Demolition | $10,000
2 | Foundation | $20,000
3 | Framing | $30,000

Total Contract Value: $60,000""",
            
            "application for payment": """APPLICATION FOR PAYMENT #1
Project: Office Building
Period: January 2024

Work Completed:
1. Site Preparation - 100% - $15,000
2. Foundation - 50% - $10,000

Total Due This Period: $25,000""",
            
            "request for information": """REQUEST FOR INFORMATION (RFI) #001
Project: Mall Expansion
Date: January 7, 2024

Question: Please clarify the specifications for the exterior cladding
material mentioned in drawing A-101.

Response Required By: January 14, 2024""",
            
            "plans and specifications": """PLANS AND SPECIFICATIONS
Project: Residential Complex
Date: January 8, 2024

Document Set Includes:
- Architectural Drawings (A-001 through A-050)
- Structural Drawings (S-001 through S-030)
- MEP Drawings (M-001 through M-040)

General Notes and Requirements..."""
        }
        
        # Return template content or generate generic content
        template = content_templates.get(doc_type, f"Document Type: {doc_type}\n\nPage 1 content...")
        return template + "\n\n" + "Additional content " * 20  # Ensure enough words
    
    def _generate_continuation_content(self, doc_type: str, page_num: int) -> str:
        """Generate continuation page content."""
        return f"Page {page_num} continuation of {doc_type}.\n\n" + "Content continues " * 30


class TestComprehensiveBoundaryDetection:
    """Test boundary detection with multiple methods against ground truth."""
    
    @pytest.fixture
    def ground_truth_evaluator(self):
        """Create ground truth evaluator."""
        ground_truth_path = Path(__file__).parent.parent.parent / "tests" / "test_files" / "Test_PDF_Set_Ground_Truth.json"
        return GroundTruthEvaluator(str(ground_truth_path))
    
    @pytest.fixture
    def test_pdf_path(self):
        """Get test PDF path."""
        return Path(__file__).parent.parent.parent / "tests" / "test_files" / "Test_PDF_Set_1.pdf"
    
    def test_text_detector_with_default_config(self, ground_truth_evaluator):
        """Test text-based detector with default configuration."""
        # Create mock pages based on ground truth
        pages = ground_truth_evaluator.create_mock_pages()
        
        # Create detector with default config
        detector = BoundaryDetector()
        boundaries = detector.detect_boundaries(pages)
        
        # Evaluate results
        results = ground_truth_evaluator.evaluate_boundaries(boundaries)
        
        print("\n=== Text Detector Results (Default) ===")
        print(f"Precision: {results['precision']:.2f}")
        print(f"Recall: {results['recall']:.2f}")
        print(f"F1 Score: {results['f1_score']:.2f}")
        print(f"Document Count: Expected {results['expected_doc_count']}, Detected {results['detected_doc_count']}")
        print(f"Document Count Accuracy: {results['doc_count_accuracy']:.2f}")
        
        if results['missed_boundaries']:
            print(f"Missed boundaries at pages: {results['missed_boundaries']}")
        if results['extra_boundaries']:
            print(f"Extra boundaries at pages: {results['extra_boundaries']}")
    
    def test_text_detector_with_optimized_config(self, ground_truth_evaluator):
        """Test text-based detector with optimized configuration."""
        # Create mock pages based on ground truth
        pages = ground_truth_evaluator.create_mock_pages()
        
        # Create detector with optimized config
        detector = BoundaryDetector(
            min_confidence=0.7,  # Increased from 0.6
            min_signals=2,       # Increased from 1
            enable_visual_analysis=True
        )
        
        # Manually adjust signal weights for better construction document detection
        detector.SIGNAL_WEIGHTS[SignalType.WHITE_SPACE] = 0.3  # Reduced weight
        detector.SIGNAL_WEIGHTS[SignalType.DOCUMENT_HEADER] = 0.9  # Increased weight
        
        boundaries = detector.detect_boundaries(pages)
        
        # Evaluate results
        results = ground_truth_evaluator.evaluate_boundaries(boundaries)
        
        print("\n=== Text Detector Results (Optimized) ===")
        print(f"Precision: {results['precision']:.2f}")
        print(f"Recall: {results['recall']:.2f}")
        print(f"F1 Score: {results['f1_score']:.2f}")
        print(f"Document Count: Expected {results['expected_doc_count']}, Detected {results['detected_doc_count']}")
        print(f"Document Count Accuracy: {results['doc_count_accuracy']:.2f}")
        
        if results['missed_boundaries']:
            print(f"Missed boundaries at pages: {results['missed_boundaries']}")
        if results['extra_boundaries']:
            print(f"Extra boundaries at pages: {results['extra_boundaries']}")
        
        # Assert reasonable performance
        assert results['f1_score'] > 0.5, f"F1 score too low: {results['f1_score']}"
        assert results['doc_count_accuracy'] >= 0.5, f"Document count accuracy too low: {results['doc_count_accuracy']}"
    
    def test_boundary_accuracy_by_document_type(self, ground_truth_evaluator):
        """Test accuracy for different document types."""
        pages = ground_truth_evaluator.create_mock_pages()
        
        # Create optimized detector
        detector = BoundaryDetector(
            min_confidence=0.7,
            min_signals=2
        )
        
        boundaries = detector.detect_boundaries(pages)
        
        # Analyze by document type
        doc_type_accuracy = {}
        
        for i, doc in enumerate(ground_truth_evaluator.expected_documents):
            doc_type = doc["document_type"]
            doc_start = doc["start_page"]
            
            # Check if boundary was correctly detected for this document
            if i > 0:  # Skip first document (no boundary before it)
                expected_boundary = doc_start
                detected = any(b.start_page == expected_boundary for b in boundaries[1:])  # Skip first boundary
                
                if doc_type not in doc_type_accuracy:
                    doc_type_accuracy[doc_type] = {"correct": 0, "total": 0}
                
                doc_type_accuracy[doc_type]["total"] += 1
                if detected:
                    doc_type_accuracy[doc_type]["correct"] += 1
        
        print("\n=== Accuracy by Document Type ===")
        for doc_type, stats in doc_type_accuracy.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"{doc_type}: {accuracy:.2f} ({stats['correct']}/{stats['total']})")
    
    def test_signal_distribution(self, ground_truth_evaluator):
        """Analyze which signals are most effective."""
        pages = ground_truth_evaluator.create_mock_pages()
        
        detector = BoundaryDetector()
        boundaries = detector.detect_boundaries(pages)
        
        # Analyze signal distribution
        signal_counts = {}
        signal_effectiveness = {}
        
        for i, boundary in enumerate(boundaries[1:], 1):  # Skip first boundary
            is_correct = boundary.start_page in ground_truth_evaluator.expected_boundaries
            
            for signal in boundary.signals:
                signal_type = signal.type.value
                
                if signal_type not in signal_counts:
                    signal_counts[signal_type] = 0
                    signal_effectiveness[signal_type] = {"correct": 0, "total": 0}
                
                signal_counts[signal_type] += 1
                signal_effectiveness[signal_type]["total"] += 1
                if is_correct:
                    signal_effectiveness[signal_type]["correct"] += 1
        
        print("\n=== Signal Effectiveness ===")
        for signal_type, stats in signal_effectiveness.items():
            effectiveness = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"{signal_type}: {effectiveness:.2f} effectiveness ({stats['correct']}/{stats['total']} correct)")
    
    def test_with_actual_pdf_ocr_disabled(self, ground_truth_evaluator, test_pdf_path):
        """Test with actual PDF but without OCR to see structure-based detection."""
        print("\n=== Testing with Actual PDF (No OCR) ===")
        
        # Create document processor without OCR
        processor = DocumentProcessor(enable_ocr=False)
        
        # Process the PDF
        pages = list(processor.process_document(str(test_pdf_path)))
        
        if not pages:
            print("No pages extracted from PDF")
            return
        
        print(f"Processed {len(pages)} pages")
        
        # Check if we have any text content
        total_text_chars = sum(len(page.text_content or "") for page in pages)
        print(f"Total text characters extracted: {total_text_chars}")
        
        if total_text_chars == 0:
            print("WARNING: No text extracted - PDF appears to be scanned without embedded text")
            print("Boundary detection will be based on visual/structural signals only")
        
        # Create detector
        detector = BoundaryDetector(
            min_confidence=0.5,  # Lower threshold for structure-only detection
            min_signals=1
        )
        
        boundaries = detector.detect_boundaries(pages)
        
        # Evaluate results
        results = ground_truth_evaluator.evaluate_boundaries(boundaries)
        
        print(f"\nPrecision: {results['precision']:.2f}")
        print(f"Recall: {results['recall']:.2f}")
        print(f"F1 Score: {results['f1_score']:.2f}")
        print(f"Document Count: Expected {results['expected_doc_count']}, Detected {results['detected_doc_count']}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])