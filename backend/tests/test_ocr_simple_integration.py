"""Simple integration test for OCR configuration and preprocessing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
import fitz
import numpy as np
import cv2

from src.core.ocr_config import AdaptiveOCRConfigurator, DocumentCharacteristics
from src.core.ocr_preprocessor import OCRPreprocessor
from src.core.ocr_optimizer import check_pdf_needs_ocr


class TestOCRSimpleIntegration:
    """Test OCR components with Test_PDF_Set_1.pdf."""
    
    def test_pdf_ocr_detection(self):
        """Test if we correctly detect that Test_PDF_Set_1.pdf needs OCR."""
        pdf_path = Path(__file__).parent.parent.parent / "tests/test_files/Test_PDF_Set_1.pdf"
        assert pdf_path.exists(), f"Test PDF not found: {pdf_path}"
        
        # Check if PDF needs OCR
        needs_ocr, reason = check_pdf_needs_ocr(str(pdf_path))
        
        print(f"\n=== OCR Detection Results ===")
        print(f"PDF needs OCR: {needs_ocr}")
        print(f"Reason: {reason}")
        
        # We know Test_PDF_Set_1.pdf is image-based
        assert needs_ocr is True
        assert "image-based" in reason.lower()
    
    def test_document_analysis(self):
        """Test document characteristic analysis."""
        pdf_path = Path(__file__).parent.parent.parent / "tests/test_files/Test_PDF_Set_1.pdf"
        
        # Open document and get sample pages
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        
        print(f"\n=== Document Analysis ===")
        print(f"Total pages: {total_pages}")
        
        # Sample first 3 pages
        sample_pages = []
        for i in range(min(3, total_pages)):
            page = doc[i]
            pix = page.get_pixmap(dpi=150)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif pix.n == 3:  # RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            sample_pages.append(img)
            pix = None
        
        doc.close()
        
        # Analyze document
        configurator = AdaptiveOCRConfigurator()
        characteristics = configurator.analyze_document(sample_pages, total_pages)
        
        print(f"\nDocument characteristics:")
        print(f"  Quality: {characteristics.quality}")
        print(f"  Is scanned: {characteristics.is_scanned}")
        print(f"  Estimated DPI: {characteristics.estimated_dpi}")
        print(f"  Noise level: {characteristics.noise_level:.2f}")
        print(f"  Skew angle: {characteristics.skew_angle:.2f}")
        print(f"  Has tables: {characteristics.has_tables}")
        print(f"  Has columns: {characteristics.has_multiple_columns}")
        
        # Generate optimal config
        config = configurator.generate_config(characteristics)
        
        print(f"\nOptimal OCR configuration:")
        print(f"  OCR engine: {config.ocr_engine}")
        print(f"  Target DPI: {config.target_dpi}")
        print(f"  Preprocessing: {', '.join(config.preprocessing_steps)}")
        print(f"  Aggressive corrections: {config.apply_aggressive_corrections}")
        print(f"  Batch size: {config.page_batch_size}")
        
        # Basic assertions
        assert characteristics.is_scanned  # Image-based PDF
        assert total_pages == 36
    
    def test_preprocessing_sample_page(self):
        """Test preprocessing on a sample page."""
        pdf_path = Path(__file__).parent.parent.parent / "tests/test_files/Test_PDF_Set_1.pdf"
        
        # Get first page as image
        doc = fitz.open(str(pdf_path))
        page = doc[0]
        pix = page.get_pixmap(dpi=150)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif pix.n == 3:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        original_shape = img.shape
        doc.close()
        pix = None
        
        print(f"\n=== Preprocessing Sample Page ===")
        print(f"Original image shape: {original_shape}")
        
        # Initialize preprocessor
        preprocessor = OCRPreprocessor(target_dpi=300)
        
        # Process image
        start_time = time.time()
        processed_img, info = preprocessor.preprocess_image(
            img,
            current_dpi=150,
            auto_enhance=True
        )
        processing_time = time.time() - start_time
        
        print(f"\nPreprocessing completed in {processing_time:.3f} seconds")
        print(f"Processed image shape: {processed_img.shape}")
        print(f"Steps applied: {', '.join(info['steps_applied'])}")
        print(f"Quality before: {info['quality_score_before']:.3f}")
        print(f"Quality after: {info['quality_score_after']:.3f}")
        
        if 'skew_angle' in info:
            print(f"Skew angle detected: {info['skew_angle']:.2f} degrees")
        
        # Save sample images for visual inspection
        output_dir = Path(__file__).parent / "ocr_test_output"
        output_dir.mkdir(exist_ok=True)
        
        # Convert back to RGB for saving
        if len(img.shape) == 3:
            cv2.imwrite(str(output_dir / "original_page1.png"), img)
        if len(processed_img.shape) == 2:
            cv2.imwrite(str(output_dir / "processed_page1.png"), processed_img)
        
        print(f"\nSample images saved to: {output_dir}")
        
        # Assertions
        assert processed_img is not None
        assert info['quality_score_after'] >= info['quality_score_before']
        assert processing_time < 5.0  # Should process single page quickly
    
    def test_page_quality_distribution(self):
        """Analyze quality distribution across all pages."""
        pdf_path = Path(__file__).parent.parent.parent / "tests/test_files/Test_PDF_Set_1.pdf"
        
        print(f"\n=== Page Quality Distribution ===")
        
        # Open document
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        
        # Analyze each page (sample every 3rd page for speed)
        configurator = AdaptiveOCRConfigurator()
        quality_scores = []
        
        for i in range(0, total_pages, 3):
            page = doc[i]
            pix = page.get_pixmap(dpi=100)  # Lower DPI for faster analysis
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            
            # Convert to grayscale for quality assessment
            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif pix.n == 3:  # RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif pix.n == 1:  # Already grayscale
                pass
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Assess quality
            quality = configurator._assess_page_quality(img)
            quality_scores.append({
                'page': i + 1,
                'overall_score': quality['overall_score'],
                'sharpness': quality['sharpness'],
                'contrast_ratio': quality['contrast_ratio']
            })
            
            pix = None
        
        doc.close()
        
        # Print results
        print(f"Analyzed {len(quality_scores)} pages")
        
        avg_score = np.mean([q['overall_score'] for q in quality_scores])
        print(f"\nAverage quality score: {avg_score:.3f}")
        
        # Find best and worst pages
        sorted_pages = sorted(quality_scores, key=lambda x: x['overall_score'])
        
        print(f"\nWorst quality pages:")
        for page_info in sorted_pages[:3]:
            print(f"  Page {page_info['page']}: {page_info['overall_score']:.3f} (sharpness: {page_info['sharpness']:.1f}, contrast: {page_info['contrast_ratio']:.2f})")
        
        print(f"\nBest quality pages:")
        for page_info in sorted_pages[-3:]:
            print(f"  Page {page_info['page']}: {page_info['overall_score']:.3f} (sharpness: {page_info['sharpness']:.1f}, contrast: {page_info['contrast_ratio']:.2f})")
        
        # Save quality distribution
        results = {
            'total_pages': total_pages,
            'pages_analyzed': len(quality_scores),
            'average_quality': float(avg_score),
            'quality_scores': quality_scores
        }
        
        results_path = Path(__file__).parent / "page_quality_distribution.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Quality distribution saved to: {results_path}")