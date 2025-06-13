#!/usr/bin/env python3
"""Test script for the updated API endpoints with LLM and visual detection."""

import requests
import json
import time
from pathlib import Path


def test_upload_with_detection_options():
    """Test the upload endpoint with different detection options."""
    
    base_url = "http://localhost:8000"
    
    # Find a test PDF file
    test_file_path = Path(__file__).parent / "tests" / "test_files" / "Test_PDF_Set_1.pdf"
    
    if not test_file_path.exists():
        print(f"Test file not found at {test_file_path}")
        print("Please ensure test files are available")
        return
    
    print(f"Using test file: {test_file_path}")
    
    # Test configurations
    test_configs = [
        {
            "name": "Basic (text only)",
            "params": {
                "enable_visual_detection": False,
                "enable_llm_detection": False,
                "enable_intelligent_ocr": False
            }
        },
        {
            "name": "Visual detection enabled",
            "params": {
                "enable_visual_detection": True,
                "enable_llm_detection": False,
                "enable_intelligent_ocr": False
            }
        },
        {
            "name": "Intelligent OCR enabled",
            "params": {
                "enable_visual_detection": True,
                "enable_llm_detection": False,
                "enable_intelligent_ocr": True
            }
        },
        {
            "name": "Full detection (visual + LLM + intelligent OCR)",
            "params": {
                "enable_visual_detection": True,
                "enable_llm_detection": True,
                "enable_intelligent_ocr": True
            }
        }
    ]
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"Parameters: {json.dumps(config['params'], indent=2)}")
        print(f"{'='*60}\n")
        
        # Upload file
        with open(test_file_path, 'rb') as f:
            files = {'file': ('test.pdf', f, 'application/pdf')}
            
            # Build URL with query parameters
            params_str = "&".join([f"{k}={v}" for k, v in config['params'].items()])
            upload_url = f"{base_url}/api/documents/upload?{params_str}"
            
            print(f"Uploading to: {upload_url}")
            response = requests.post(upload_url, files=files)
        
        if response.status_code != 200:
            print(f"Upload failed: {response.status_code}")
            print(response.text)
            continue
        
        upload_result = response.json()
        document_id = upload_result['document_id']
        print(f"Document uploaded successfully. ID: {document_id}")
        
        # Poll for processing completion
        print("Processing document...")
        max_attempts = 60  # 5 minutes with 5-second intervals
        attempts = 0
        
        while attempts < max_attempts:
            status_response = requests.get(f"{base_url}/api/documents/{document_id}/status")
            
            if status_response.status_code != 200:
                print(f"Status check failed: {status_response.status_code}")
                break
            
            status_data = status_response.json()
            status = status_data['status']
            
            if status == 'completed':
                print(f"Processing completed in ~{attempts * 5} seconds")
                
                # Get boundaries
                boundaries = status_data.get('detected_boundaries', [])
                print(f"\nDetected {len(boundaries)} boundaries:")
                
                for i, boundary in enumerate(boundaries):
                    doc_type = boundary.get('document_type', 'unknown')
                    confidence = boundary.get('confidence', 0)
                    start_page = boundary.get('start_page', 0)
                    end_page = boundary.get('end_page', 0)
                    signals = boundary.get('signals', [])
                    
                    print(f"\nBoundary {i+1}:")
                    print(f"  Type: {doc_type}")
                    print(f"  Pages: {start_page}-{end_page}")
                    print(f"  Confidence: {confidence:.2%}")
                    print(f"  Signals: {len(signals)}")
                    
                    # Show first few signals
                    for j, signal in enumerate(signals[:3]):
                        print(f"    - {signal.get('type', 'unknown')}: {signal.get('description', '')}")
                    
                    if len(signals) > 3:
                        print(f"    ... and {len(signals) - 3} more signals")
                
                # Get quality summary if available
                try:
                    quality_response = requests.get(f"{base_url}/api/documents/{document_id}/quality")
                    if quality_response.status_code == 200:
                        quality_data = quality_response.json()
                        print(f"\nOCR Quality Summary:")
                        print(f"  Overall confidence: {quality_data.get('overall_confidence', 0):.2%}")
                        print(f"  Quality assessment: {quality_data.get('quality_assessment', 'unknown')}")
                except Exception as e:
                    print(f"Could not get quality summary: {e}")
                
                break
                
            elif status == 'failed':
                print(f"Processing failed: {status_data.get('message', 'Unknown error')}")
                break
            
            else:
                print(f"Status: {status}... (attempt {attempts + 1}/{max_attempts})")
                time.sleep(5)
                attempts += 1
        
        if attempts >= max_attempts:
            print("Processing timed out")
        
        # Clean up
        print("\nCleaning up...")
        delete_response = requests.delete(f"{base_url}/api/documents/{document_id}")
        if delete_response.status_code == 200:
            print("Document deleted successfully")
        else:
            print(f"Failed to delete document: {delete_response.status_code}")


def test_enhanced_routes():
    """Test the enhanced API routes."""
    print("\n\n" + "="*80)
    print("Testing Enhanced Routes")
    print("="*80)
    
    base_url = "http://localhost:8000"
    
    # Test detection presets endpoint
    print("\nTesting detection presets endpoint...")
    response = requests.get(f"{base_url}/api/v2/detection-presets")
    
    if response.status_code == 200:
        presets = response.json()
        print("Available presets:")
        for preset in presets.get('presets', []):
            print(f"  - {preset['name']}: {preset['description']}")
            print(f"    Detectors: {', '.join(preset['detectors'])}")
    else:
        print(f"Failed to get presets: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    print("Smart PDF Splitter API Integration Test")
    print("======================================")
    print("\nMake sure the API server is running on http://localhost:8000")
    print("Run with: cd backend && uvicorn src.api.main:app --reload")
    
    print("\nStarting tests...")
    
    try:
        # Check if server is running
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            print("Server is not responding. Please start the server first.")
            exit(1)
        
        test_upload_with_detection_options()
        test_enhanced_routes()
        
    except requests.ConnectionError:
        print("\nError: Could not connect to the API server.")
        print("Please ensure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()