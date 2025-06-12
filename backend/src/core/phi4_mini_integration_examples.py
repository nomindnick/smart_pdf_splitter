"""
Integration examples and usage patterns for phi4-mini boundary detection.

This module demonstrates how to integrate the phi4-mini detector into
the existing Smart PDF Splitter system with various configurations.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from .models import PageInfo, DocumentType
from .boundary_detector_enhanced import (
    EnhancedBoundaryDetector,
    DetectorConfig,
    ExtendedSignalType
)
from .phi4_mini_detector_design import (
    Phi4MiniBoundaryDetector,
    get_phi4_mini_config,
    get_enhanced_detector_with_phi4
)

logger = logging.getLogger(__name__)


# Example 1: Basic phi4-mini integration
async def basic_phi4_mini_example():
    """Basic example of using phi4-mini for boundary detection."""
    
    # Simple configuration
    detector_configs = [
        DetectorConfig(
            name="phi4_mini",
            enabled=True,
            weight=1.0,
            config={
                'phi4_settings': {
                    'model_name': 'phi4-mini:3.8b',
                    'ollama_url': 'http://localhost:11434',
                    'temperature': 0.1,
                    'max_chars_per_page': 800
                }
            }
        )
    ]
    
    # Create detector
    detector = EnhancedBoundaryDetector(
        min_confidence=0.7,
        detector_configs=detector_configs
    )
    
    # Example pages (would come from DocumentProcessor)
    pages = [
        PageInfo(
            page_number=1,
            width=612,
            height=792,
            text_content="Dear Mr. Smith,\n\nThank you for your inquiry...",
            word_count=150
        ),
        PageInfo(
            page_number=2,
            width=612,
            height=792,
            text_content="From: jane@example.com\nTo: john@example.com\nSubject: Project Update",
            word_count=200
        )
    ]
    
    # Detect boundaries
    boundaries = await detector.detect_boundaries(pages)
    
    for boundary in boundaries:
        print(f"Boundary: pages {boundary.start_page}-{boundary.end_page}")
        print(f"Type: {boundary.document_type}, Confidence: {boundary.confidence:.2%}")


# Example 2: Construction-focused configuration
async def construction_focused_example():
    """Example optimized for construction documents."""
    
    # Construction-optimized settings
    detector_configs = [
        # Traditional patterns for fast detection
        DetectorConfig(
            name="rule_based",
            enabled=True,
            weight=0.6,
            config={'min_confidence': 0.5}
        ),
        
        # Phi4-mini with construction boost
        DetectorConfig(
            name="phi4_mini",
            enabled=True,
            weight=1.0,
            config={
                'phi4_settings': {
                    'model_name': 'phi4-mini:3.8b',
                    'temperature': 0.1,
                    'max_chars_per_page': 1000,  # More context for complex docs
                    'use_few_shot': True,  # Include examples
                    'context_window_pages': 3  # Larger context
                }
            }
        ),
        
        # Construction-specific patterns with high weight
        DetectorConfig(
            name="construction",
            enabled=True,
            weight=1.3,  # Highest weight for domain expertise
            config={}
        )
    ]
    
    detector = EnhancedBoundaryDetector(
        min_confidence=0.65,  # Slightly lower threshold
        min_signals=1,
        detector_configs=detector_configs
    )
    
    # Example construction document pages
    pages = [
        PageInfo(
            page_number=1,
            width=612,
            height=792,
            text_content="RFI #2024-001\nProject: Downtown Tower\nTo: Architect\nFrom: General Contractor",
            word_count=50
        ),
        PageInfo(
            page_number=2,
            width=612,
            height=792,
            text_content="Question: Please clarify the specification for concrete strength...",
            word_count=200
        ),
        PageInfo(
            page_number=3,
            width=612,
            height=792,
            text_content="SUBMITTAL TRANSMITTAL\nSubmittal No: S-2024-042\nSpec Section: 03300",
            word_count=75
        )
    ]
    
    boundaries = await detector.detect_boundaries(pages)
    
    # Analyze which detectors found what
    for boundary in boundaries:
        print(f"\nBoundary at page {boundary.start_page}:")
        signal_summary = {}
        for signal in boundary.signals:
            detector_type = signal.metadata.get('model', signal.type)
            signal_summary[detector_type] = signal_summary.get(detector_type, 0) + 1
            
        print(f"  Detected by: {signal_summary}")
        print(f"  Document type: {boundary.document_type}")


# Example 3: Performance-optimized configuration
async def performance_optimized_example():
    """Example with performance optimizations for large PDFs."""
    
    detector_configs = [
        # Fast rule-based pre-filter
        DetectorConfig(
            name="rule_based",
            enabled=True,
            weight=0.8,
            config={'min_confidence': 0.7}
        ),
        
        # Phi4-mini in fast mode
        DetectorConfig(
            name="phi4_mini",
            enabled=True,
            weight=0.9,
            config={
                'phi4_settings': {
                    'model_name': 'phi4-mini:3.8b',
                    'temperature': 0.2,  # Slightly higher for speed
                    'max_chars_per_page': 500,  # Less context
                    'context_window_pages': 1,  # Minimal context
                    'batch_size': 10,  # Larger batches
                    'use_few_shot': False,  # Skip examples
                    'compress_whitespace': True,
                    'focus_on_headers': True,
                    'timeout_seconds': 15.0  # Shorter timeout
                }
            }
        )
    ]
    
    detector = EnhancedBoundaryDetector(
        min_confidence=0.75,  # Higher threshold for speed
        detector_configs=detector_configs
    )
    
    # Simulate large document
    pages = []
    for i in range(100):
        if i % 10 == 0:  # Every 10th page is a new document
            content = f"INVOICE #{1000 + i//10}\nBill To: Customer {i//10}"
        else:
            content = f"Page {i+1} content... continuing from previous..."
        
        pages.append(PageInfo(
            page_number=i+1,
            width=612,
            height=792,
            text_content=content,
            word_count=100 + (i % 50)
        ))
    
    import time
    start = time.time()
    boundaries = await detector.detect_boundaries(pages)
    elapsed = time.time() - start
    
    print(f"Processed {len(pages)} pages in {elapsed:.2f} seconds")
    print(f"Found {len(boundaries)} documents")
    print(f"Average: {elapsed/len(pages)*1000:.1f} ms per page")


# Example 4: Custom prompt engineering
class CustomPhi4MiniDetector(Phi4MiniBoundaryDetector):
    """Example of customizing prompts for specific use cases."""
    
    def _create_phi4_prompt(self, prev_text: str, curr_text: str, detected_patterns: List[str]) -> str:
        """Custom prompt for financial documents."""
        
        prompt = f"""You are analyzing financial and business documents. Determine if these are different documents.

PREVIOUS PAGE (last 500 chars):
{prev_text[-500:]}

CURRENT PAGE (first 500 chars):
{curr_text[:500]}

Look for:
- Invoice/PO numbers changing
- Different companies or addresses
- Date changes indicating new documents
- Financial document headers (Invoice, Statement, Receipt)

Answer in JSON: {{"boundary": true/false, "confidence": 0-1, "type": "invoice|statement|po|receipt|other", "reason": "why"}}"""
        
        return prompt


# Example 5: Hybrid detection with fallbacks
async def hybrid_detection_example():
    """Example showing hybrid detection with fallbacks."""
    
    class FallbackDetectorSystem:
        def __init__(self):
            self.primary_config = [
                get_phi4_mini_config(enable_construction_boost=True, fast_mode=False)
            ]
            self.fallback_config = [
                DetectorConfig(
                    name="rule_based",
                    enabled=True,
                    weight=1.0,
                    config={'min_confidence': 0.5}
                )
            ]
            
        async def detect_with_fallback(self, pages: List[PageInfo]) -> List:
            """Try primary detector, fall back if needed."""
            try:
                # Try phi4-mini first
                detector = EnhancedBoundaryDetector(
                    min_confidence=0.7,
                    detector_configs=self.primary_config
                )
                
                # Set timeout for LLM detection
                boundaries = await asyncio.wait_for(
                    detector.detect_boundaries(pages),
                    timeout=30.0
                )
                
                if boundaries:
                    logger.info("Successfully used phi4-mini detection")
                    return boundaries
                    
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Primary detection failed: {e}, using fallback")
            
            # Use fallback
            fallback_detector = EnhancedBoundaryDetector(
                min_confidence=0.6,
                detector_configs=self.fallback_config
            )
            
            return await fallback_detector.detect_boundaries(pages)
    
    # Example usage
    system = FallbackDetectorSystem()
    pages = [
        PageInfo(page_number=1, width=612, height=792, 
                text_content="Document 1 content...", word_count=100),
        PageInfo(page_number=2, width=612, height=792, 
                text_content="From: sender@example.com\nTo: recipient@example.com", word_count=50)
    ]
    
    boundaries = await system.detect_with_fallback(pages)
    print(f"Detected {len(boundaries)} boundaries with fallback system")


# Example 6: Monitoring and debugging
async def monitoring_example():
    """Example showing how to monitor phi4-mini performance."""
    
    class MonitoredPhi4Detector(Phi4MiniBoundaryDetector):
        def __init__(self, config: DetectorConfig):
            super().__init__(config)
            self.stats = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'total_time': 0.0,
                'confidence_distribution': []
            }
            
        async def _analyze_with_phi4(self, prev_page, curr_page, context_pages, quick_check):
            import time
            start = time.time()
            self.stats['total_calls'] += 1
            
            try:
                result = await super()._analyze_with_phi4(
                    prev_page, curr_page, context_pages, quick_check
                )
                self.stats['successful_calls'] += 1
                self.stats['confidence_distribution'].append(result[1])
                return result
            except Exception as e:
                self.stats['failed_calls'] += 1
                raise
            finally:
                self.stats['total_time'] += time.time() - start
                
        def print_stats(self):
            """Print performance statistics."""
            if self.stats['total_calls'] == 0:
                print("No calls made yet")
                return
                
            print("\n=== Phi4-mini Performance Stats ===")
            print(f"Total calls: {self.stats['total_calls']}")
            print(f"Success rate: {self.stats['successful_calls']/self.stats['total_calls']:.1%}")
            print(f"Average time: {self.stats['total_time']/self.stats['total_calls']:.3f}s")
            
            if self.stats['confidence_distribution']:
                avg_conf = sum(self.stats['confidence_distribution']) / len(self.stats['confidence_distribution'])
                print(f"Average confidence: {avg_conf:.2%}")
    
    # Use monitored detector
    config = DetectorConfig(
        name="phi4_mini_monitored",
        enabled=True,
        weight=1.0,
        config={'phi4_settings': {'model_name': 'phi4-mini:3.8b'}}
    )
    
    # Would need to register this custom class
    detector = MonitoredPhi4Detector(config)
    
    # Process some pages...
    # detector.print_stats()


# Example 7: Configuration for different document types
def get_document_specific_config(document_focus: str = "general") -> List[DetectorConfig]:
    """Get detector configuration optimized for specific document types."""
    
    configs = {
        "general": [
            get_phi4_mini_config(enable_construction_boost=False, fast_mode=False),
            DetectorConfig(name="rule_based", enabled=True, weight=0.8)
        ],
        
        "construction": [
            get_phi4_mini_config(enable_construction_boost=True, fast_mode=False),
            DetectorConfig(name="construction", enabled=True, weight=1.3),
            DetectorConfig(name="rule_based", enabled=True, weight=0.6)
        ],
        
        "financial": [
            DetectorConfig(
                name="phi4_mini",
                enabled=True,
                weight=1.0,
                config={
                    'phi4_settings': {
                        'model_name': 'phi4-mini:3.8b',
                        'temperature': 0.05,  # Very low for consistency
                        'max_chars_per_page': 1000,  # More context for tables
                        'focus_on_headers': True
                    }
                }
            ),
            DetectorConfig(name="rule_based", enabled=True, weight=0.9)
        ],
        
        "email": [
            DetectorConfig(name="rule_based", enabled=True, weight=1.1),  # Very good at emails
            DetectorConfig(
                name="phi4_mini",
                enabled=True,
                weight=0.7,
                config={
                    'phi4_settings': {
                        'model_name': 'phi4-mini:3.8b',
                        'max_chars_per_page': 600,  # Emails are usually shorter
                        'context_window_pages': 1
                    }
                }
            )
        ]
    }
    
    return configs.get(document_focus, configs["general"])


# Example 8: API endpoint integration
async def api_endpoint_example(pdf_path: str, config_name: str = "general"):
    """Example of how to integrate with FastAPI endpoints."""
    
    from .document_processor import DocumentProcessor
    
    # Initialize processor and detector
    processor = DocumentProcessor()
    detector_configs = get_document_specific_config(config_name)
    
    detector = EnhancedBoundaryDetector(
        min_confidence=0.7,
        detector_configs=detector_configs
    )
    
    # Process PDF
    pages = await processor.process_pdf(pdf_path)
    
    # Detect boundaries
    boundaries = await detector.detect_boundaries(pages)
    
    # Format for API response
    response = {
        "document_id": "example_123",
        "total_pages": len(pages),
        "detected_documents": len(boundaries),
        "boundaries": [
            {
                "start_page": b.start_page,
                "end_page": b.end_page,
                "confidence": b.confidence,
                "document_type": b.document_type.value if b.document_type else None,
                "detection_method": "phi4-mini" if any(
                    s.metadata.get('model') == 'phi4-mini:3.8b' 
                    for s in b.signals
                ) else "rule-based"
            }
            for b in boundaries
        ]
    }
    
    return response


# Main example runner
async def main():
    """Run various examples."""
    
    print("=== Running Phi4-mini Integration Examples ===\n")
    
    examples = [
        ("Basic phi4-mini", basic_phi4_mini_example),
        ("Construction-focused", construction_focused_example),
        ("Performance-optimized", performance_optimized_example),
        ("Hybrid with fallback", hybrid_detection_example),
    ]
    
    for name, example_func in examples:
        print(f"\n--- {name} Example ---")
        try:
            await example_func()
        except Exception as e:
            print(f"Error in {name}: {e}")
            
    print("\n=== Examples Complete ===")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run examples
    asyncio.run(main())