#!/usr/bin/env python3
"""
Comprehensive test script comparing all three boundary detection methods:
1. Text-based detection (pattern matching)
2. Visual detection (layout analysis)
3. LLM-based detection (Phi-4 Mini)

This script runs all methods on the test PDF and compares their performance
against the ground truth data.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import time

# Add the backend src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.boundary_detector import BoundaryDetector
from core.visual_boundary_detector import VisualBoundaryDetector
from core.phi4_mini_detector import Phi4MiniBoundaryDetector
from core.hybrid_boundary_detector import HybridBoundaryDetector, VisualProcessingConfig
from core.unified_document_processor import UnifiedDocumentProcessor, ProcessingMode
from core.models import Boundary, PageInfo, PageVisualInfo, DocumentType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose Docling logs
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("docling.backend").setLevel(logging.WARNING)


@dataclass
class DetectionResult:
    """Results from a single detection method."""
    method_name: str
    boundaries: List[Boundary]
    processing_time: float
    memory_usage_mb: float = 0.0
    error: Optional[str] = None


@dataclass
class GroundTruthDocument:
    """Ground truth document information."""
    start_page: int
    end_page: int
    doc_type: str
    summary: str


class BoundaryDetectionEvaluator:
    """Evaluates boundary detection methods against ground truth."""
    
    def __init__(self, test_pdf_path: Path, ground_truth_path: Path):
        self.test_pdf_path = test_pdf_path
        self.ground_truth_path = ground_truth_path
        self.ground_truth = self._load_ground_truth()
        
    def _load_ground_truth(self) -> List[GroundTruthDocument]:
        """Load ground truth data from JSON file."""
        with open(self.ground_truth_path, 'r') as f:
            data = json.load(f)
        
        documents = []
        for doc in data['documents']:
            # Parse page range
            pages = doc['pages']
            if '-' in pages:
                start, end = pages.split('-')
                start_page = int(start)
                end_page = int(end)
            else:
                start_page = end_page = int(pages)
            
            documents.append(GroundTruthDocument(
                start_page=start_page,
                end_page=end_page,
                doc_type=doc['type'],
                summary=doc['summary']
            ))
        
        return documents
    
    def run_all_detections(self) -> Dict[str, DetectionResult]:
        """Run all detection methods and collect results."""
        results = {}
        
        # 1. Text-based detection
        logger.info("\n" + "="*60)
        logger.info("Running TEXT-BASED detection...")
        results['text'] = self._run_text_detection()
        
        # 2. Visual detection
        logger.info("\n" + "="*60)
        logger.info("Running VISUAL detection...")
        results['visual'] = self._run_visual_detection()
        
        # 3. LLM-based detection (Phi-4 Mini)
        logger.info("\n" + "="*60)
        logger.info("Running LLM-BASED detection (Phi-4 Mini)...")
        results['llm'] = self._run_llm_detection()
        
        # 4. Hybrid detection (combines all methods)
        logger.info("\n" + "="*60)
        logger.info("Running HYBRID detection (all methods combined)...")
        results['hybrid'] = self._run_hybrid_detection()
        
        return results
    
    def _run_text_detection(self) -> DetectionResult:
        """Run text-based boundary detection."""
        try:
            start_time = time.time()
            
            # Initialize detector and processor
            detector = BoundaryDetector(
                min_confidence=0.6,
                min_signals=1,
                enable_visual_analysis=False
            )
            processor = UnifiedDocumentProcessor(mode=ProcessingMode.BASIC)
            
            # Process document
            doc = processor.process_document(self.test_pdf_path)
            pages = doc.page_info
            
            # Detect boundaries
            boundaries = detector.detect_boundaries(pages)
            
            processing_time = time.time() - start_time
            
            return DetectionResult(
                method_name="Text-Based Detection",
                boundaries=boundaries,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            return DetectionResult(
                method_name="Text-Based Detection",
                boundaries=[],
                processing_time=0.0,
                error=str(e)
            )
    
    def _run_visual_detection(self) -> DetectionResult:
        """Run visual boundary detection."""
        try:
            start_time = time.time()
            
            # Initialize detector and processor
            detector = VisualBoundaryDetector(
                min_visual_confidence=0.5,
                enable_vlm_analysis=False
            )
            processor = EnhancedDocumentProcessor(
                enable_visual_features=True,
                enable_vlm=False
            )
            
            # Process document with visual features
            pages = list(processor.process_document_with_visual(self.test_pdf_path))
            
            # Detect boundaries using visual signals
            visual_candidates = detector.detect_visual_boundaries(pages)
            
            # Convert candidates to boundaries
            boundaries = []
            for candidate in visual_candidates:
                if candidate.visual_confidence >= 0.5:
                    boundary = Boundary(
                        start_page=candidate.page_number,
                        end_page=candidate.page_number,
                        confidence=candidate.visual_confidence,
                        signals=[],  # Visual signals are separate
                        document_type=DocumentType.OTHER
                    )
                    boundaries.append(boundary)
            
            # Fix end pages
            for i in range(len(boundaries) - 1):
                boundaries[i].end_page = boundaries[i + 1].start_page - 1
            if boundaries:
                boundaries[-1].end_page = 36  # Last page of test PDF
            
            processing_time = time.time() - start_time
            
            return DetectionResult(
                method_name="Visual Detection",
                boundaries=boundaries,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Visual detection failed: {e}")
            return DetectionResult(
                method_name="Visual Detection",
                boundaries=[],
                processing_time=0.0,
                error=str(e)
            )
    
    def _run_llm_detection(self) -> DetectionResult:
        """Run LLM-based boundary detection."""
        try:
            start_time = time.time()
            
            # Initialize Phi-4 Mini detector
            detector = Phi4MiniBoundaryDetector(
                model_name="phi4-mini:3.8b",
                min_confidence=0.6,
                use_llm_for_ambiguous=True,
                llm_batch_size=5
            )
            processor = EnhancedDocumentProcessor(enable_visual_features=False)
            
            # Process document
            pages = list(processor.process_document(self.test_pdf_path))
            
            # Detect boundaries with LLM assistance
            boundaries = detector.detect_boundaries(pages)
            
            processing_time = time.time() - start_time
            
            return DetectionResult(
                method_name="LLM-Based Detection (Phi-4 Mini)",
                boundaries=boundaries,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"LLM detection failed: {e}")
            return DetectionResult(
                method_name="LLM-Based Detection (Phi-4 Mini)",
                boundaries=[],
                processing_time=0.0,
                error=str(e)
            )
    
    def _run_hybrid_detection(self) -> DetectionResult:
        """Run hybrid detection combining all methods."""
        try:
            start_time = time.time()
            
            # Configure hybrid detector
            config = VisualProcessingConfig(
                enable_visual_features=True,
                enable_vlm=False,  # VLM is separate from Phi-4
                visual_confidence_threshold=0.5
            )
            
            detector = HybridBoundaryDetector(
                config=config,
                text_weight=0.6,
                visual_weight=0.4,
                min_combined_confidence=0.6
            )
            
            # Detect boundaries
            boundaries = detector.detect_boundaries(self.test_pdf_path)
            
            processing_time = time.time() - start_time
            
            return DetectionResult(
                method_name="Hybrid Detection (Text + Visual)",
                boundaries=boundaries,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Hybrid detection failed: {e}")
            return DetectionResult(
                method_name="Hybrid Detection",
                boundaries=[],
                processing_time=0.0,
                error=str(e)
            )
    
    def evaluate_results(self, results: Dict[str, DetectionResult]) -> Dict[str, Any]:
        """Evaluate detection results against ground truth."""
        evaluation = {}
        
        for method, result in results.items():
            if result.error:
                evaluation[method] = {
                    'error': result.error,
                    'processing_time': result.processing_time
                }
                continue
            
            # Calculate metrics
            metrics = self._calculate_metrics(result.boundaries)
            
            # Document type accuracy
            doc_type_accuracy = self._calculate_doc_type_accuracy(result.boundaries)
            
            evaluation[method] = {
                'method_name': result.method_name,
                'processing_time': result.processing_time,
                'boundaries_detected': len(result.boundaries),
                'boundaries_expected': len(self.ground_truth),
                'metrics': metrics,
                'document_type_accuracy': doc_type_accuracy,
                'detailed_results': self._get_detailed_results(result.boundaries)
            }
        
        return evaluation
    
    def _calculate_metrics(self, boundaries: List[Boundary]) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score."""
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Ground truth boundary pages
        gt_boundaries = {doc.start_page for doc in self.ground_truth}
        
        # Detected boundary pages
        detected_boundaries = {b.start_page for b in boundaries}
        
        # Calculate TP, FP
        for detected in detected_boundaries:
            if detected in gt_boundaries:
                true_positives += 1
            else:
                false_positives += 1
        
        # Calculate FN
        for gt in gt_boundaries:
            if gt not in detected_boundaries:
                false_negatives += 1
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def _calculate_doc_type_accuracy(self, boundaries: List[Boundary]) -> Dict[str, Any]:
        """Calculate document type classification accuracy."""
        correct_types = 0
        total_matched = 0
        type_confusion = {}
        
        for boundary in boundaries:
            # Find matching ground truth
            for gt_doc in self.ground_truth:
                if boundary.start_page == gt_doc.start_page:
                    total_matched += 1
                    
                    # Map boundary document type to ground truth type
                    detected_type = boundary.document_type.value if boundary.document_type else "unknown"
                    gt_type = gt_doc.doc_type.lower()
                    
                    # Check if types match (with some flexibility)
                    if self._types_match(detected_type, gt_type):
                        correct_types += 1
                    else:
                        key = f"{gt_type} -> {detected_type}"
                        type_confusion[key] = type_confusion.get(key, 0) + 1
                    break
        
        accuracy = correct_types / total_matched if total_matched > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct_types,
            'total': total_matched,
            'confusion': type_confusion
        }
    
    def _types_match(self, detected: str, ground_truth: str) -> bool:
        """Check if detected type matches ground truth with some flexibility."""
        detected = detected.lower()
        ground_truth = ground_truth.lower()
        
        # Direct match
        if detected == ground_truth:
            return True
        
        # Flexible matching rules
        type_mappings = {
            'email chain': ['email'],
            'submittal': ['report', 'other'],
            'schedule of values': ['report', 'spreadsheet', 'other'],
            'application for payment': ['form', 'invoice'],
            'invoice': ['invoice'],
            'request for information': ['report', 'letter'],
            'plans and specifications': ['report', 'other'],
            'cost proposal': ['quote', 'invoice', 'report']
        }
        
        if ground_truth in type_mappings:
            return detected in type_mappings[ground_truth]
        
        return False
    
    def _get_detailed_results(self, boundaries: List[Boundary]) -> List[Dict[str, Any]]:
        """Get detailed results for each boundary."""
        results = []
        
        for boundary in boundaries:
            # Find matching ground truth
            matched_gt = None
            for gt_doc in self.ground_truth:
                if boundary.start_page == gt_doc.start_page:
                    matched_gt = gt_doc
                    break
            
            result = {
                'page': boundary.start_page,
                'page_range': f"{boundary.start_page}-{boundary.end_page}",
                'confidence': boundary.confidence,
                'detected_type': boundary.document_type.value if boundary.document_type else "unknown",
                'signals': [s.type.value for s in boundary.signals],
                'matched': matched_gt is not None
            }
            
            if matched_gt:
                result['ground_truth_type'] = matched_gt.doc_type
                result['ground_truth_range'] = f"{matched_gt.start_page}-{matched_gt.end_page}"
                result['type_correct'] = self._types_match(
                    result['detected_type'], 
                    matched_gt.doc_type
                )
            
            results.append(result)
        
        return results
    
    def print_results(self, evaluation: Dict[str, Any]):
        """Print formatted evaluation results."""
        print("\n" + "="*80)
        print("BOUNDARY DETECTION EVALUATION RESULTS")
        print("="*80)
        print(f"Test PDF: {self.test_pdf_path.name}")
        print(f"Ground Truth Documents: {len(self.ground_truth)}")
        print("="*80)
        
        # Sort methods by F1 score
        methods_by_f1 = sorted(
            [(method, data) for method, data in evaluation.items() if 'metrics' in data],
            key=lambda x: x[1]['metrics']['f1_score'],
            reverse=True
        )
        
        for method, data in methods_by_f1:
            print(f"\n{data['method_name']}")
            print("-" * len(data['method_name']))
            
            if 'error' in data:
                print(f"ERROR: {data['error']}")
                continue
            
            metrics = data['metrics']
            print(f"Processing Time: {data['processing_time']:.2f} seconds")
            print(f"Boundaries Detected: {data['boundaries_detected']} (Expected: {data['boundaries_expected']})")
            print(f"Precision: {metrics['precision']:.2%}")
            print(f"Recall: {metrics['recall']:.2%}")
            print(f"F1 Score: {metrics['f1_score']:.2%}")
            print(f"True Positives: {metrics['true_positives']}")
            print(f"False Positives: {metrics['false_positives']}")
            print(f"False Negatives: {metrics['false_negatives']}")
            
            doc_acc = data['document_type_accuracy']
            print(f"\nDocument Type Accuracy: {doc_acc['accuracy']:.2%} ({doc_acc['correct']}/{doc_acc['total']})")
            
            if doc_acc['confusion']:
                print("Type Confusion:")
                for mistake, count in doc_acc['confusion'].items():
                    print(f"  {mistake}: {count}")
        
        # Best method summary
        print("\n" + "="*80)
        print("SUMMARY: Best Detection Method")
        print("="*80)
        
        if methods_by_f1:
            best_method, best_data = methods_by_f1[0]
            print(f"Winner: {best_data['method_name']}")
            print(f"F1 Score: {best_data['metrics']['f1_score']:.2%}")
            print(f"Processing Time: {best_data['processing_time']:.2f} seconds")
            
            # Performance comparison
            print("\nPerformance Comparison:")
            for method, data in methods_by_f1:
                if 'metrics' in data:
                    print(f"  {data['method_name']}: F1={data['metrics']['f1_score']:.2%}, Time={data['processing_time']:.1f}s")
        
        # Document type performance
        print("\n" + "="*80)
        print("DOCUMENT TYPE PERFORMANCE")
        print("="*80)
        
        # Analyze which methods work best for different document types
        self._analyze_document_type_performance(evaluation)
    
    def _analyze_document_type_performance(self, evaluation: Dict[str, Any]):
        """Analyze which methods work best for different document types."""
        # Group ground truth by type
        doc_types = {}
        for gt_doc in self.ground_truth:
            doc_type = gt_doc.doc_type
            if doc_type not in doc_types:
                doc_types[doc_type] = []
            doc_types[doc_type].append(gt_doc.start_page)
        
        # Check each method's performance per document type
        for doc_type, pages in doc_types.items():
            print(f"\n{doc_type} (Pages: {', '.join(map(str, pages))})")
            print("-" * len(doc_type))
            
            for method, data in evaluation.items():
                if 'detailed_results' not in data:
                    continue
                
                detected = 0
                for result in data['detailed_results']:
                    if result['page'] in pages and result['matched']:
                        detected += 1
                
                detection_rate = detected / len(pages) if pages else 0
                print(f"  {data['method_name']}: {detection_rate:.0%} ({detected}/{len(pages)})")
    
    def save_detailed_report(self, evaluation: Dict[str, Any], output_path: Path):
        """Save detailed evaluation report to JSON file."""
        report = {
            'test_pdf': str(self.test_pdf_path),
            'ground_truth_file': str(self.ground_truth_path),
            'evaluation_results': evaluation,
            'ground_truth': [
                {
                    'start_page': doc.start_page,
                    'end_page': doc.end_page,
                    'type': doc.doc_type,
                    'summary': doc.summary
                }
                for doc in self.ground_truth
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Detailed report saved to: {output_path}")


def main():
    """Run the comprehensive boundary detection evaluation."""
    # Paths
    test_pdf = Path(__file__).parent.parent / "tests" / "test_files" / "Test_PDF_Set_1.pdf"
    ground_truth = Path(__file__).parent.parent / "tests" / "test_files" / "Test_PDF_Set_Ground_Truth.json"
    
    # Verify files exist
    if not test_pdf.exists():
        logger.error(f"Test PDF not found: {test_pdf}")
        sys.exit(1)
    
    if not ground_truth.exists():
        logger.error(f"Ground truth file not found: {ground_truth}")
        sys.exit(1)
    
    # Create evaluator
    evaluator = BoundaryDetectionEvaluator(test_pdf, ground_truth)
    
    # Run all detection methods
    logger.info("Starting comprehensive boundary detection evaluation...")
    results = evaluator.run_all_detections()
    
    # Evaluate results
    evaluation = evaluator.evaluate_results(results)
    
    # Print results
    evaluator.print_results(evaluation)
    
    # Save detailed report
    report_path = Path(__file__).parent / "boundary_detection_evaluation_report.json"
    evaluator.save_detailed_report(evaluation, report_path)
    
    print(f"\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    main()