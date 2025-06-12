"""
Tests for visual boundary detection functionality.

This module tests the visual boundary detection features against the ground truth
data for the test PDF set.
"""

import pytest
from pathlib import Path
import json
from typing import List, Dict, Any

from src.core.hybrid_boundary_detector import HybridBoundaryDetector, VisualProcessingConfig
from src.core.models import Boundary, DocumentType


class TestVisualBoundaryDetection:
    """Test visual boundary detection features."""
    
    @pytest.fixture
    def test_pdf_path(self):
        """Path to test PDF file."""
        return Path(__file__).parent.parent / "tests" / "test_files" / "Test_PDF_Set_1.pdf"
    
    @pytest.fixture
    def ground_truth(self):
        """Load ground truth data."""
        ground_truth_path = Path(__file__).parent.parent / "tests" / "test_files" / "Test_PDF_Set_Ground_Truth.json"
        with open(ground_truth_path, 'r') as f:
            return json.load(f)
    
    @pytest.fixture
    def visual_config(self):
        """Visual processing configuration for tests."""
        return VisualProcessingConfig(
            enable_visual_features=True,
            enable_picture_classification=True,
            enable_vlm=False,  # Disabled for tests
            visual_memory_limit_mb=1024,  # Lower for tests
            visual_confidence_threshold=0.5,
            visual_batch_size=2
        )
    
    def test_visual_feature_extraction(self, test_pdf_path, visual_config):
        """Test that visual features are extracted correctly."""
        detector = HybridBoundaryDetector(config=visual_config)
        
        # Process first few pages to test visual extraction
        from src.core.enhanced_document_processor import EnhancedDocumentProcessor
        processor = EnhancedDocumentProcessor(
            enable_visual_features=True,
            page_batch_size=2
        )
        
        pages = list(processor.process_document_with_visual(
            test_pdf_path,
            page_range=(1, 5)
        ))
        
        # Check that visual features were extracted
        assert len(pages) >= 5
        
        pages_with_visual = sum(1 for p in pages if p.visual_features is not None)
        assert pages_with_visual > 0, "No visual features extracted"
        
        # Check specific visual features
        for page in pages:
            if page.visual_features:
                assert page.visual_features.num_columns >= 1
                assert page.visual_features.orientation in ["portrait", "landscape"]
    
    def test_hybrid_detection_accuracy(self, test_pdf_path, ground_truth, visual_config):
        """Test hybrid detection against ground truth."""
        detector = HybridBoundaryDetector(
            config=visual_config,
            text_weight=0.5,
            visual_weight=0.5
        )
        
        boundaries = detector.detect_boundaries(test_pdf_path)
        
        # Compare with ground truth
        gt_documents = ground_truth['documents']
        
        # Should detect approximately the same number of documents
        assert len(boundaries) >= len(gt_documents) - 2, \
            f"Expected ~{len(gt_documents)} documents, got {len(boundaries)}"
        assert len(boundaries) <= len(gt_documents) + 2, \
            f"Expected ~{len(gt_documents)} documents, got {len(boundaries)}"
        
        # Check that major boundaries are detected
        # Ground truth has documents starting at pages: 1, 5, 7, 9, 13, 14, 18, 20, 23, 26, 32, 34, 35, 36
        major_boundaries = [1, 5, 7, 9, 13, 14, 18, 20, 23, 26, 32, 34, 35, 36]
        detected_starts = [b.start_page for b in boundaries]
        
        # Count how many major boundaries were detected
        detected_major = sum(1 for mb in major_boundaries if mb in detected_starts)
        assert detected_major >= len(major_boundaries) * 0.7, \
            f"Only detected {detected_major}/{len(major_boundaries)} major boundaries"
    
    def test_visual_signals_present(self, test_pdf_path, visual_config):
        """Test that visual signals are being generated."""
        detector = HybridBoundaryDetector(config=visual_config)
        boundaries = detector.detect_boundaries(test_pdf_path)
        
        # Count boundaries with visual signals
        visual_signal_count = 0
        for boundary in boundaries:
            for signal in boundary.signals:
                if hasattr(signal.type, 'value') and 'visual' in str(signal.type.value):
                    visual_signal_count += 1
                    break
        
        assert visual_signal_count > 0, "No visual signals detected"
    
    def test_detection_methods(self, test_pdf_path, visual_config):
        """Test different detection methods are being used."""
        detector = HybridBoundaryDetector(config=visual_config)
        boundaries = detector.detect_boundaries(test_pdf_path)
        
        # Get detection summary
        summary = detector.get_detection_summary(boundaries)
        
        # Should have multiple detection methods
        assert summary['detection_methods']['hybrid'] > 0, "No hybrid detections"
        assert summary['total_documents'] == len(boundaries)
        assert summary['average_confidence'] > 0.5
    
    def test_visual_only_detection(self, test_pdf_path, visual_config):
        """Test visual-only detection by disabling text detection."""
        # Create detector with very low text weight
        detector = HybridBoundaryDetector(
            config=visual_config,
            text_weight=0.1,
            visual_weight=0.9
        )
        
        boundaries = detector.detect_boundaries(test_pdf_path)
        
        # Should still detect boundaries
        assert len(boundaries) > 5, "Visual detection should find boundaries"
        
        # Check that visual signals are dominant
        for boundary in boundaries[:5]:  # Check first 5
            visual_signals = sum(1 for s in boundary.signals 
                               if hasattr(s.type, 'value') and 'visual' in str(s.type.value))
            assert visual_signals > 0, "Visual-weighted detection should have visual signals"
    
    def test_memory_efficiency(self, test_pdf_path, visual_config):
        """Test that memory limits are respected."""
        import psutil
        import os
        
        # Configure with lower memory limit
        low_memory_config = VisualProcessingConfig(
            enable_visual_features=True,
            visual_memory_limit_mb=512,
            visual_batch_size=1
        )
        
        detector = HybridBoundaryDetector(config=low_memory_config)
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process document
        boundaries = detector.detect_boundaries(test_pdf_path)
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 1024, f"Memory increased by {memory_increase}MB"
        assert len(boundaries) > 0, "Should still detect boundaries with memory limit"
    
    def test_document_type_inference(self, test_pdf_path, visual_config):
        """Test document type inference from visual features."""
        detector = HybridBoundaryDetector(config=visual_config)
        boundaries = detector.detect_boundaries(test_pdf_path)
        
        # Count document types
        doc_types = {}
        for boundary in boundaries:
            if boundary.document_type:
                doc_type = boundary.document_type.value
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # Should detect multiple document types
        assert len(doc_types) > 1, "Should detect multiple document types"
        
        # Based on ground truth, should have emails, invoices, etc.
        expected_types = ['email', 'invoice', 'form', 'report']
        detected_types = list(doc_types.keys())
        
        matches = sum(1 for et in expected_types if et in detected_types)
        assert matches >= 2, f"Only detected {matches} expected document types"
    
    @pytest.mark.parametrize("page_range,expected_min", [
        ((1, 10), 2),   # First 10 pages should have at least 2 documents
        ((20, 30), 2),  # Pages 20-30 should have at least 2 documents
        ((30, 36), 3),  # Last pages should have at least 3 documents
    ])
    def test_page_range_processing(self, test_pdf_path, visual_config, page_range, expected_min):
        """Test processing specific page ranges."""
        detector = HybridBoundaryDetector(config=visual_config)
        boundaries = detector.detect_boundaries(test_pdf_path, page_range=page_range)
        
        assert len(boundaries) >= expected_min, \
            f"Expected at least {expected_min} boundaries in pages {page_range}"
        
        # All boundaries should be within the page range
        for boundary in boundaries:
            assert boundary.start_page >= page_range[0]
            assert boundary.end_page <= page_range[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])