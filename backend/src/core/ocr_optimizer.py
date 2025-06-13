"""OCR optimization utilities for smart PDF processing."""

import logging
from typing import Optional, Tuple
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def check_page_needs_ocr(
    page: fitz.Page,
    text_threshold: int = 50,
    image_area_threshold: float = 0.8
) -> Tuple[bool, str]:
    """
    Check if a PDF page needs OCR.
    
    Args:
        page: PyMuPDF page object
        text_threshold: Minimum character count to consider page has text
        image_area_threshold: Minimum image area ratio to consider page as image-based
        
    Returns:
        Tuple of (needs_ocr, reason)
    """
    # Extract text
    text = page.get_text()
    text_length = len(text.strip())
    
    # If page has sufficient text, no OCR needed
    if text_length > text_threshold:
        return False, f"Page has {text_length} characters of embedded text"
    
    # Check if page is mostly images
    page_area = page.rect.width * page.rect.height
    image_area = 0
    
    for img in page.get_images():
        try:
            # Get image dimensions
            xref = img[0]
            pix = fitz.Pixmap(page.parent, xref)
            image_area += pix.width * pix.height
            pix = None  # Free memory
        except:
            continue
    
    image_ratio = image_area / page_area if page_area > 0 else 0
    
    if image_ratio > image_area_threshold:
        return True, f"Page is {image_ratio:.1%} images with only {text_length} text chars"
    
    # Page has neither text nor significant images
    return False, f"Page appears empty (text: {text_length}, image ratio: {image_ratio:.1%})"


def check_pdf_needs_ocr(
    pdf_path: str,
    sample_size: int = 5,
    ocr_threshold: float = 0.5
) -> Tuple[bool, str]:
    """
    Check if a PDF needs OCR by sampling pages.
    
    Args:
        pdf_path: Path to PDF file
        sample_size: Number of pages to sample
        ocr_threshold: Fraction of sampled pages that need OCR to enable it
        
    Returns:
        Tuple of (needs_ocr, reason)
    """
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # Sample evenly distributed pages
        sample_indices = []
        if total_pages <= sample_size:
            sample_indices = list(range(total_pages))
        else:
            step = total_pages / sample_size
            sample_indices = [int(i * step) for i in range(sample_size)]
        
        pages_needing_ocr = 0
        reasons = []
        
        for idx in sample_indices:
            page = doc[idx]
            needs_ocr, reason = check_page_needs_ocr(page)
            if needs_ocr:
                pages_needing_ocr += 1
                reasons.append(f"Page {idx + 1}: {reason}")
        
        doc.close()
        
        ocr_ratio = pages_needing_ocr / len(sample_indices)
        
        if ocr_ratio >= ocr_threshold:
            return True, f"{pages_needing_ocr}/{len(sample_indices)} sampled pages need OCR"
        else:
            return False, f"Only {pages_needing_ocr}/{len(sample_indices)} sampled pages need OCR"
            
    except Exception as e:
        logger.error(f"Error checking PDF for OCR needs: {e}")
        return True, f"Error checking PDF, enabling OCR as fallback: {e}"


def get_optimal_ocr_engine(
    page_count: int,
    has_gpu: bool = False,
    memory_limit_mb: int = 4096
) -> str:
    """
    Determine optimal OCR engine based on document characteristics.
    
    Args:
        page_count: Number of pages in document
        has_gpu: Whether GPU is available
        memory_limit_mb: Memory limit in MB
        
    Returns:
        OCR engine name: "easyocr", "tesserocr", or "tesseract-cli"
    """
    # EasyOCR is best with GPU, but memory hungry
    if has_gpu and memory_limit_mb > 8192:
        return "easyocr"
    
    # For large documents without GPU, limit to first few pages or disable OCR
    if page_count > 50 and not has_gpu:
        return "easyocr"  # Will process with limited pages
    
    # Default to EasyOCR which is included with Docling
    return "easyocr"