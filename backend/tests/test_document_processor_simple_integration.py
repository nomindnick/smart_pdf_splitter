"""
Simple integration test to verify document processor works.
"""

from src.core.document_processor import DocumentProcessor

def test_basic_functionality():
    """Test that we can create and use a DocumentProcessor."""
    print("Creating DocumentProcessor...")
    processor = DocumentProcessor(
        enable_ocr=False,
        page_batch_size=1
    )
    
    print("✓ DocumentProcessor created successfully")
    
    # Test pipeline options
    options = processor._create_pipeline_options()
    print(f"✓ Pipeline options created: OCR={options.do_ocr}")
    
    # Check converter is initialized
    assert processor.converter is not None
    print("✓ Converter initialized")
    
    print("\nAll basic functionality tests passed!")

if __name__ == "__main__":
    test_basic_functionality()