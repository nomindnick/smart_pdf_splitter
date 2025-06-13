"""Parallel processing utilities for OCR optimization."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Tuple
from pathlib import Path
import logging
import fitz
import numpy as np
from dataclasses import dataclass
import time

from .ocr_config import OCRConfig

logger = logging.getLogger(__name__)


@dataclass
class PageTask:
    """Represents a single page OCR task."""
    pdf_path: Path
    page_number: int
    config: OCRConfig
    purpose: str = "detection"


@dataclass 
class PageResult:
    """Result from processing a single page."""
    page_number: int
    text: str
    confidence: float
    processing_time: float
    word_count: int
    error: Optional[str] = None


class ParallelOCRProcessor:
    """
    Handles parallel OCR processing with configurable strategies.
    
    Supports both CPU-bound parallelism (multiprocessing) and
    I/O-bound parallelism (threading) depending on the use case.
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = True,
        batch_size: int = 4
    ):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Number of parallel workers (None = CPU count)
            use_processes: Use processes (True) or threads (False)
            batch_size: Pages to process per batch
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.use_processes = use_processes
        self.batch_size = batch_size
        
        logger.info(
            f"Initialized ParallelOCRProcessor with {self.max_workers} workers "
            f"({'processes' if use_processes else 'threads'})"
        )
    
    def process_document(
        self,
        pdf_path: Path,
        config: OCRConfig,
        page_numbers: Optional[List[int]] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[int, PageResult]:
        """
        Process document pages in parallel.
        
        Args:
            pdf_path: Path to PDF document
            config: OCR configuration
            page_numbers: Specific pages to process (None = all)
            progress_callback: Progress callback function
            
        Returns:
            Dictionary mapping page numbers to results
        """
        # Determine pages to process
        if page_numbers is None:
            doc = fitz.open(str(pdf_path))
            page_numbers = list(range(len(doc)))
            doc.close()
        
        total_pages = len(page_numbers)
        logger.info(f"Processing {total_pages} pages in parallel")
        
        # Create tasks
        tasks = [
            PageTask(pdf_path, page_num, config)
            for page_num in page_numbers
        ]
        
        # Process in parallel
        results = {}
        completed = 0
        
        # Choose executor based on configuration
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit tasks in batches
            future_to_task = {}
            
            for i in range(0, len(tasks), self.batch_size):
                batch = tasks[i:i + self.batch_size]
                for task in batch:
                    future = executor.submit(self._process_single_page, task)
                    future_to_task[future] = task
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results[result.page_number] = result
                    completed += 1
                    
                    if progress_callback:
                        progress = (completed / total_pages) * 100
                        progress_callback(
                            progress,
                            f"Processed page {result.page_number}"
                        )
                        
                except Exception as e:
                    logger.error(f"Failed to process page {task.page_number}: {e}")
                    results[task.page_number] = PageResult(
                        page_number=task.page_number,
                        text="",
                        confidence=0.0,
                        processing_time=0.0,
                        word_count=0,
                        error=str(e)
                    )
        
        return results
    
    @staticmethod
    def _process_single_page(task: PageTask) -> PageResult:
        """
        Process a single page (runs in separate process/thread).
        
        This method must be static to work with multiprocessing.
        """
        start_time = time.time()
        
        try:
            # Open document in this process
            doc = fitz.open(str(task.pdf_path))
            page = doc[task.page_number]
            
            # Check if page needs OCR
            existing_text = page.get_text()
            if len(existing_text.strip()) > 50 and task.purpose == "detection":
                # Page has text, skip OCR for detection
                doc.close()
                return PageResult(
                    page_number=task.page_number,
                    text=existing_text,
                    confidence=1.0,
                    processing_time=time.time() - start_time,
                    word_count=len(existing_text.split())
                )
            
            # Extract page as image
            pix = page.get_pixmap(dpi=task.config.target_dpi)
            img = np.frombuffer(
                pix.samples,
                dtype=np.uint8
            ).reshape(pix.height, pix.width, pix.n)
            
            # Convert RGBA to RGB if needed
            if pix.n == 4:
                img = img[:, :, :3]
            
            # Perform OCR based on engine
            if task.config.ocr_engine == "tesseract-cli":
                result = ParallelOCRProcessor._ocr_with_tesseract(img, task.config)
            else:
                result = ParallelOCRProcessor._ocr_with_easyocr(img, task.config)
            
            doc.close()
            
            return PageResult(
                page_number=task.page_number,
                text=result["text"],
                confidence=result["confidence"],
                processing_time=time.time() - start_time,
                word_count=len(result["text"].split())
            )
            
        except Exception as e:
            logger.error(f"Error processing page {task.page_number}: {e}")
            return PageResult(
                page_number=task.page_number,
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                word_count=0,
                error=str(e)
            )
    
    @staticmethod
    def _ocr_with_tesseract(image: np.ndarray, config: OCRConfig) -> Dict[str, Any]:
        """Perform OCR using Tesseract."""
        try:
            import pytesseract
            
            # Configure Tesseract
            custom_config = r'--oem 3 --psm 6'
            if config.ocr_languages:
                lang = '+'.join(config.ocr_languages)
                custom_config += f' -l {lang}'
            
            # Run OCR
            data = pytesseract.image_to_data(
                image,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and confidence
            text = " ".join(
                word for word, conf in zip(data['text'], data['conf'])
                if conf > 0 and word.strip()
            )
            
            confidences = [c for c in data['conf'] if c > 0]
            avg_confidence = np.mean(confidences) / 100 if confidences else 0
            
            return {
                "text": text,
                "confidence": avg_confidence
            }
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return {"text": "", "confidence": 0.0}
    
    @staticmethod
    def _ocr_with_easyocr(image: np.ndarray, config: OCRConfig) -> Dict[str, Any]:
        """Perform OCR using EasyOCR."""
        try:
            import easyocr
            
            # Initialize reader (cached in process)
            if not hasattr(ParallelOCRProcessor._ocr_with_easyocr, 'reader'):
                ParallelOCRProcessor._ocr_with_easyocr.reader = easyocr.Reader(
                    config.ocr_languages,
                    gpu=False  # No GPU
                )
            
            reader = ParallelOCRProcessor._ocr_with_easyocr.reader
            
            # Run OCR
            results = reader.readtext(image)
            
            # Extract text and confidence
            if results:
                texts = [item[1] for item in results]
                confidences = [item[2] for item in results]
                
                text = " ".join(texts)
                avg_confidence = np.mean(confidences) if confidences else 0
            else:
                text = ""
                avg_confidence = 0.0
            
            return {
                "text": text,
                "confidence": avg_confidence
            }
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return {"text": "", "confidence": 0.0}


class AdaptiveParallelProcessor:
    """
    Adaptive processor that chooses optimal strategy based on document.
    
    This is designed to work well for both standalone splitting
    and future RAG processing.
    """
    
    def __init__(self):
        self.strategies = {
            "small": ParallelOCRProcessor(max_workers=2, batch_size=2),
            "medium": ParallelOCRProcessor(max_workers=4, batch_size=4),
            "large": ParallelOCRProcessor(max_workers=mp.cpu_count(), batch_size=8)
        }
    
    def process_document(
        self,
        pdf_path: Path,
        config: OCRConfig,
        purpose: str = "detection"
    ) -> Dict[int, PageResult]:
        """
        Process document with adaptive strategy.
        
        Args:
            pdf_path: Path to PDF
            config: OCR configuration
            purpose: "detection" or "extraction"
        """
        # Determine document size
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        doc.close()
        
        # Choose strategy
        if purpose == "detection":
            # For boundary detection, limit parallelism
            if page_count < 10:
                strategy = "small"
            elif page_count < 50:
                strategy = "medium"
            else:
                # Sample pages for large documents
                strategy = "medium"
                # TODO: Implement page sampling
        else:
            # For full extraction (RAG), use maximum parallelism
            strategy = "large"
        
        logger.info(
            f"Using {strategy} strategy for {page_count} pages ({purpose})"
        )
        
        processor = self.strategies[strategy]
        return processor.process_document(pdf_path, config)