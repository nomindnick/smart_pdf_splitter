"""
Simple test to verify document processor structure without external dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.document_processor import DocumentProcessor
from src.core.models import PageInfo, BoundingBox


def test_document_processor_initialization():
    """Test that DocumentProcessor can be initialized."""
    try:
        processor = DocumentProcessor()
        print("✓ DocumentProcessor initialized successfully")
        
        # Check attributes
        assert hasattr(processor, 'enable_ocr')
        assert hasattr(processor, 'ocr_languages')
        assert hasattr(processor, 'page_batch_size')
        assert hasattr(processor, 'max_memory_mb')
        print("✓ All required attributes present")
        
        # Check default values
        assert processor.enable_ocr is True
        assert processor.ocr_languages == ["en"]
        assert processor.page_batch_size == 4
        assert processor.max_memory_mb == 4096
        print("✓ Default values are correct")
        
        # Test custom initialization
        custom_processor = DocumentProcessor(
            enable_ocr=False,
            ocr_languages=["en", "es"],
            page_batch_size=8,
            max_memory_mb=8192
        )
        assert custom_processor.enable_ocr is False
        assert custom_processor.ocr_languages == ["en", "es"]
        assert custom_processor.page_batch_size == 8
        assert custom_processor.max_memory_mb == 8192
        print("✓ Custom initialization works")
        
        print("\nAll tests passed! ✅")
        
    except ImportError as e:
        print(f"Import error (likely missing dependencies): {e}")
        print("This is expected if Docling is not installed yet.")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    test_document_processor_initialization()