#!/usr/bin/env python3
"""Test script for enhanced OCR API endpoints."""

import requests
import json
import time
import sys
from pathlib import Path


def test_enhanced_api():
    """Test the enhanced OCR API endpoints."""
    base_url = "http://localhost:8000"
    
    # Check if server is running
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Server is running: {response.json()}")
    except requests.ConnectionError:
        print("❌ Server is not running. Please start it with: uvicorn src.api.main:app --reload")
        return
    
    # Test PDF path
    test_pdf = Path(__file__).parent.parent / "tests/test_files/Test_PDF_Set_1.pdf"
    if not test_pdf.exists():
        print(f"❌ Test PDF not found: {test_pdf}")
        return
    
    print(f"\n📄 Testing with: {test_pdf.name}")
    
    # Upload document
    print("\n1. Uploading document...")
    with open(test_pdf, "rb") as f:
        files = {"file": (test_pdf.name, f, "application/pdf")}
        response = requests.post(f"{base_url}/api/documents/upload", files=files)
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        return
    
    upload_data = response.json()
    document_id = upload_data["document_id"]
    print(f"✅ Document uploaded: {document_id}")
    
    # Poll for processing completion
    print("\n2. Processing document (this may take a while)...")
    start_time = time.time()
    max_wait = 300  # 5 minutes
    
    while True:
        response = requests.get(f"{base_url}/api/documents/{document_id}/status")
        if response.status_code != 200:
            print(f"❌ Status check failed: {response.text}")
            return
        
        status_data = response.json()
        status = status_data["status"]
        
        elapsed = time.time() - start_time
        print(f"\r   Status: {status} ({elapsed:.1f}s elapsed)", end="", flush=True)
        
        if status == "completed":
            print(f"\n✅ Processing completed in {elapsed:.1f} seconds")
            break
        elif status == "failed":
            print(f"\n❌ Processing failed: {status_data.get('message', 'Unknown error')}")
            return
        elif elapsed > max_wait:
            print(f"\n❌ Processing timeout after {elapsed:.1f} seconds")
            return
        
        time.sleep(2)
    
    # Get quality summary
    print("\n3. Getting OCR quality summary...")
    response = requests.get(f"{base_url}/api/documents/{document_id}/quality")
    
    if response.status_code != 200:
        print(f"❌ Quality check failed: {response.text}")
        return
    
    quality_data = response.json()
    print("\n📊 OCR Quality Summary:")
    print(f"   Overall confidence: {quality_data.get('overall_confidence', 0):.2%}")
    print(f"   Quality assessment: {quality_data.get('quality_assessment', 'N/A')}")
    
    if "processing_stats" in quality_data:
        stats = quality_data["processing_stats"]
        print(f"\n📈 Processing Statistics:")
        print(f"   Pages processed: {stats.get('pages_processed', 0)}")
        print(f"   Pages preprocessed: {stats.get('pages_preprocessed', 0)}")
        print(f"   OCR performed: {stats.get('ocr_performed', 0)}")
        print(f"   Average confidence: {stats.get('average_confidence', 0):.2%}")
        print(f"   Total corrections: {stats.get('total_corrections', 0)}")
    
    if "low_confidence_pages" in quality_data and quality_data["low_confidence_pages"]:
        print(f"\n⚠️  Low Confidence Pages ({len(quality_data['low_confidence_pages'])}):")
        for page_info in quality_data["low_confidence_pages"][:5]:
            print(f"   Page {page_info['page']}: {page_info['confidence']:.2%}")
            if page_info.get('issues'):
                print(f"      Issues: {', '.join(page_info['issues'])}")
    
    # Get sample page details
    print("\n4. Getting page-level OCR details...")
    for page_num in [1, 10, 20]:
        response = requests.get(f"{base_url}/api/documents/{document_id}/pages/{page_num}/ocr")
        
        if response.status_code == 200:
            page_data = response.json()
            print(f"\n📄 Page {page_num}:")
            print(f"   Word count: {page_data.get('word_count', 0)}")
            if page_data.get('ocr_confidence') is not None:
                print(f"   OCR confidence: {page_data['ocr_confidence']:.2%}")
            print(f"   Quality: {page_data.get('ocr_quality_assessment', 'N/A')}")
            if page_data.get('preprocessing_applied'):
                print(f"   Preprocessing: {', '.join(page_data['preprocessing_applied'])}")
            if page_data.get('corrections_made', 0) > 0:
                print(f"   Corrections made: {page_data['corrections_made']}")
            if page_data.get('needs_review'):
                print(f"   ⚠️  Needs manual review")
    
    # Get boundaries
    print("\n5. Getting detected boundaries...")
    response = requests.get(f"{base_url}/api/documents/{document_id}/boundaries")
    
    if response.status_code == 200:
        boundaries = response.json()
        print(f"\n📋 Detected {len(boundaries)} document boundaries:")
        for i, boundary in enumerate(boundaries[:5]):
            print(f"   Document {i+1}: Pages {boundary['start_page']}-{boundary['end_page']} "
                  f"(confidence: {boundary['confidence']:.2%})")
            if boundary.get('document_type'):
                print(f"      Type: {boundary['document_type']}")
    
    # Clean up
    print("\n6. Cleaning up...")
    response = requests.delete(f"{base_url}/api/documents/{document_id}")
    if response.status_code == 200:
        print("✅ Document deleted")
    else:
        print(f"⚠️  Failed to delete document: {response.text}")
    
    print("\n✨ Test completed successfully!")


if __name__ == "__main__":
    test_enhanced_api()