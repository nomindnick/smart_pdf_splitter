"""
Integration test for document processor with a real PDF.
"""

import pytest
from pathlib import Path
from src.core.document_processor import DocumentProcessor


def test_document_processor_with_test_pdf():
    """Test document processor with the test PDF file."""
    # Check if test PDF exists
    test_pdf_path = Path("/home/nick/Projects/smart_pdf_splitter/tests/test_files/Test_PDF_Set_1.pdf")
    
    if not test_pdf_path.exists():
        pytest.skip(f"Test PDF not found at {test_pdf_path}")
    
    # Create processor
    processor = DocumentProcessor(
        enable_ocr=False,  # Disable OCR for faster testing
        page_batch_size=4
    )
    
    # Process the test PDF (first few pages only)
    pages = list(processor.process_document(test_pdf_path))
    
    # Verify we got pages
    assert len(pages) > 0
    
    # Check first page
    first_page = pages[0]
    assert first_page.page_number == 1
    assert first_page.width > 0
    assert first_page.height > 0
    assert first_page.text_content is not None
    assert first_page.word_count >= 0
    
    # Print some info for debugging
    print(f"\nProcessed {len(pages)} pages")
    for page in pages:
        print(f"Page {page.page_number}: {page.word_count} words, "
              f"has_images={page.has_images}, has_tables={page.has_tables}")
        print(f"  Text preview: {page.text_content[:100]}..." if page.text_content else "  No text")


def test_document_metadata():
    """Test getting document metadata."""
    test_pdf_path = Path("/home/nick/Projects/smart_pdf_splitter/tests/test_files/Test_PDF_Set_1.pdf")
    
    if not test_pdf_path.exists():
        pytest.skip(f"Test PDF not found at {test_pdf_path}")
    
    processor = DocumentProcessor(enable_ocr=False)
    metadata = processor.get_document_metadata(test_pdf_path)
    
    # Check metadata
    assert 'total_pages' in metadata
    assert metadata['total_pages'] > 0
    print(f"\nDocument metadata: {metadata}")


if __name__ == "__main__":
    # Run tests directly
    test_document_processor_with_test_pdf()
    print("\n" + "="*50 + "\n")
    test_document_metadata()