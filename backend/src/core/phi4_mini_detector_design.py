"""
Design and implementation for phi4-mini:3.8b boundary detection via Ollama.

This module provides an optimized LLM-based boundary detector specifically
designed for phi4-mini:3.8b, which is a smaller, faster model that can run
efficiently on modest hardware while maintaining good accuracy.

Key Design Principles:
1. Optimized prompts for phi4-mini's capabilities
2. Efficient context management for the 8k token limit
3. Construction document expertise with universal applicability
4. Fast inference with minimal resource usage
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import asyncio
from collections import defaultdict

import httpx

from .models import (
    PageInfo, 
    Boundary, 
    Signal, 
    SignalType, 
    DocumentType,
)
from .boundary_detector_enhanced import (
    BoundaryDetectorPlugin,
    BoundaryCandidate,
    DetectorConfig,
    ExtendedSignalType
)

logger = logging.getLogger(__name__)


@dataclass
class Phi4MiniConfig:
    """Configuration specific to phi4-mini model."""
    
    # Model settings
    model_name: str = "phi4-mini:3.8b"
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.1  # Low temperature for consistent results
    top_p: float = 0.9
    top_k: int = 40
    
    # Context management
    max_context_tokens: int = 4096  # Conservative limit for phi4-mini
    max_chars_per_page: int = 800   # Roughly 200 tokens per page
    context_window_pages: int = 2    # Look at 2 pages before/after
    
    # Optimization settings
    batch_size: int = 5              # Process multiple boundaries in one call
    cache_embeddings: bool = True    # Cache page embeddings if available
    use_structured_output: bool = True  # Use JSON mode for reliable parsing
    
    # Prompt optimization
    use_few_shot: bool = True        # Include examples in prompts
    compress_whitespace: bool = True  # Reduce tokens by compressing spaces
    focus_on_headers: bool = True    # Prioritize document headers
    
    # Performance tuning
    timeout_seconds: float = 30.0
    max_retries: int = 2
    enable_streaming: bool = False   # Disable for simpler implementation


@dataclass
class DocumentPattern:
    """Pattern for detecting specific document types."""
    
    name: str
    keywords: List[str]
    header_patterns: List[str]
    structural_hints: List[str]
    confidence_boost: float = 0.0


class Phi4MiniBoundaryDetector(BoundaryDetectorPlugin):
    """
    Optimized boundary detector for phi4-mini:3.8b model.
    
    This detector uses prompt engineering and context management
    specifically tailored for phi4-mini's strengths and limitations.
    """
    
    # Document patterns optimized for construction and general documents
    DOCUMENT_PATTERNS = {
        'email': DocumentPattern(
            name='email',
            keywords=['from:', 'to:', 'subject:', 'date:', 'sent:'],
            header_patterns=[r'^From:\s*', r'^To:\s*', r'^Subject:\s*'],
            structural_hints=['email header block', 'recipient list', 'timestamp'],
            confidence_boost=0.1
        ),
        'invoice': DocumentPattern(
            name='invoice',
            keywords=['invoice', 'bill to', 'ship to', 'total', 'amount due'],
            header_patterns=[r'Invoice\s*#', r'Invoice\s+Number'],
            structural_hints=['billing address', 'line items', 'totals section'],
            confidence_boost=0.05
        ),
        'rfi': DocumentPattern(
            name='rfi',
            keywords=['RFI', 'request for information', 'response required', 'clarification'],
            header_patterns=[r'RFI\s*#?\s*\d+', r'REQUEST\s+FOR\s+INFORMATION'],
            structural_hints=['question section', 'response section', 'project reference'],
            confidence_boost=0.15  # Higher boost for construction expertise
        ),
        'submittal': DocumentPattern(
            name='submittal',
            keywords=['submittal', 'shop drawing', 'product data', 'material sample'],
            header_patterns=[r'SUBMITTAL\s*#?\s*\d+', r'SUBMITTAL\s+TRANSMITTAL'],
            structural_hints=['spec section reference', 'approval stamps', 'transmittal form'],
            confidence_boost=0.15
        ),
        'change_order': DocumentPattern(
            name='change_order',
            keywords=['change order', 'contract modification', 'cost impact', 'PCO'],
            header_patterns=[r'CHANGE\s+ORDER\s*#?\s*\d+', r'PCO\s*#?\s*\d+'],
            structural_hints=['cost breakdown', 'schedule impact', 'justification'],
            confidence_boost=0.15
        ),
        'contract': DocumentPattern(
            name='contract',
            keywords=['agreement', 'terms and conditions', 'whereas', 'party', 'obligations'],
            header_patterns=[r'CONTRACT', r'AGREEMENT'],
            structural_hints=['parties section', 'terms section', 'signature blocks'],
            confidence_boost=0.0
        ),
        'letter': DocumentPattern(
            name='letter',
            keywords=['dear', 'sincerely', 'regards', 'attention'],
            header_patterns=[r'^Dear\s+', r'RE:\s*', r'ATTENTION:\s*'],
            structural_hints=['letterhead', 'salutation', 'closing'],
            confidence_boost=0.0
        ),
        'report': DocumentPattern(
            name='report',
            keywords=['executive summary', 'table of contents', 'findings', 'recommendations'],
            header_patterns=[r'TABLE\s+OF\s+CONTENTS', r'EXECUTIVE\s+SUMMARY'],
            structural_hints=['section numbers', 'page numbers', 'multi-page structure'],
            confidence_boost=0.0
        )
    }
    
    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.phi_config = Phi4MiniConfig(**config.config.get('phi4_settings', {}))
        self.client = httpx.AsyncClient(timeout=self.phi_config.timeout_seconds)
        self.pattern_cache: Dict[str, DocumentPattern] = {}
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        for doc_type, pattern in self.DOCUMENT_PATTERNS.items():
            # Compile header patterns
            pattern.header_patterns = [
                re.compile(p, re.IGNORECASE | re.MULTILINE) 
                for p in pattern.header_patterns
            ]
            self.pattern_cache[doc_type] = pattern
    
    async def detect_boundaries(
        self, 
        pages: List[PageInfo],
        context_window: int = 3
    ) -> List[BoundaryCandidate]:
        """
        Detect boundaries using phi4-mini with optimized prompting.
        
        Uses batching and smart context management for efficiency.
        """
        if not pages or len(pages) < 2:
            return []
            
        candidates = []
        
        # Use configured context window
        context_window = self.phi_config.context_window_pages
        
        # Process in batches for efficiency
        batch_size = self.phi_config.batch_size
        
        for i in range(1, len(pages), batch_size):
            batch_end = min(i + batch_size, len(pages))
            batch_candidates = await self._process_batch(
                pages, 
                start_idx=i, 
                end_idx=batch_end,
                context_window=context_window
            )
            candidates.extend(batch_candidates)
            
        return candidates
    
    async def _process_batch(
        self,
        pages: List[PageInfo],
        start_idx: int,
        end_idx: int,
        context_window: int
    ) -> List[BoundaryCandidate]:
        """Process a batch of potential boundaries."""
        batch_candidates = []
        
        for idx in range(start_idx, end_idx):
            # Get context pages
            context_start = max(0, idx - context_window)
            context_end = min(len(pages), idx + context_window + 1)
            
            # Quick pre-filter using patterns
            quick_check = self._quick_boundary_check(pages[idx-1], pages[idx])
            
            if quick_check['potential_boundary']:
                # Full LLM analysis only for potential boundaries
                is_boundary, confidence, doc_type, reasoning = await self._analyze_with_phi4(
                    pages[idx-1],
                    pages[idx],
                    pages[context_start:context_end],
                    quick_check
                )
                
                if is_boundary and confidence >= 0.5:
                    signal = Signal(
                        type=ExtendedSignalType.LLM_SEMANTIC,
                        confidence=confidence,
                        page_number=pages[idx].page_number,
                        description=f"phi4-mini: {reasoning}",
                        metadata={
                            'model': self.phi_config.model_name,
                            'document_type': doc_type,
                            'quick_check_boost': quick_check.get('confidence_boost', 0)
                        }
                    )
                    
                    candidate = BoundaryCandidate(
                        page_number=pages[idx].page_number,
                        signals=[signal],
                        confidence=confidence,
                        suggested_type=self._map_to_document_type(doc_type)
                    )
                    batch_candidates.append(candidate)
                    
        return batch_candidates
    
    def _quick_boundary_check(self, prev_page: PageInfo, curr_page: PageInfo) -> Dict[str, Any]:
        """
        Quick pattern-based check to filter obvious boundaries.
        Reduces LLM calls by pre-filtering.
        """
        result = {
            'potential_boundary': False,
            'detected_patterns': [],
            'confidence_boost': 0.0
        }
        
        if not prev_page.text_content or not curr_page.text_content:
            return result
            
        # Check current page for document headers
        curr_text_sample = curr_page.text_content[:self.phi_config.max_chars_per_page]
        
        for doc_type, pattern in self.pattern_cache.items():
            # Check keywords
            keyword_matches = sum(
                1 for kw in pattern.keywords 
                if kw.lower() in curr_text_sample.lower()
            )
            
            # Check header patterns
            header_matches = sum(
                1 for hp in pattern.header_patterns
                if hp.search(curr_text_sample)
            )
            
            if keyword_matches >= 2 or header_matches >= 1:
                result['potential_boundary'] = True
                result['detected_patterns'].append(doc_type)
                result['confidence_boost'] = max(
                    result['confidence_boost'], 
                    pattern.confidence_boost
                )
                
        # Check for significant text length change
        prev_len = len(prev_page.text_content)
        curr_len = len(curr_page.text_content)
        if prev_len > 0:
            length_ratio = abs(curr_len - prev_len) / prev_len
            if length_ratio > 0.7:
                result['potential_boundary'] = True
                
        return result
    
    async def _analyze_with_phi4(
        self,
        prev_page: PageInfo,
        curr_page: PageInfo,
        context_pages: List[PageInfo],
        quick_check: Dict[str, Any]
    ) -> Tuple[bool, float, str, str]:
        """
        Analyze boundary using phi4-mini with optimized prompting.
        """
        # Prepare context with token management
        prev_text = self._prepare_page_text(prev_page, focus_headers=True)
        curr_text = self._prepare_page_text(curr_page, focus_headers=True)
        
        # Create optimized prompt for phi4-mini
        prompt = self._create_phi4_prompt(
            prev_text, 
            curr_text, 
            quick_check['detected_patterns']
        )
        
        try:
            # Call Ollama with phi4-mini
            response = await self.client.post(
                f"{self.phi_config.ollama_url}/api/generate",
                json={
                    "model": self.phi_config.model_name,
                    "prompt": prompt,
                    "temperature": self.phi_config.temperature,
                    "top_p": self.phi_config.top_p,
                    "top_k": self.phi_config.top_k,
                    "stream": self.phi_config.enable_streaming,
                    "format": "json" if self.phi_config.use_structured_output else None
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Parse response
                if self.phi_config.use_structured_output:
                    analysis = json.loads(result.get('response', '{}'))
                else:
                    # Fallback parsing for non-JSON responses
                    analysis = self._parse_text_response(result.get('response', ''))
                
                is_boundary = analysis.get('boundary', False)
                confidence = float(analysis.get('confidence', 0.5))
                doc_type = analysis.get('type', 'unknown')
                reasoning = analysis.get('reason', 'No specific reason provided')
                
                # Apply confidence boost from quick check
                if is_boundary and quick_check.get('confidence_boost', 0) > 0:
                    confidence = min(1.0, confidence + quick_check['confidence_boost'])
                    
                return is_boundary, confidence, doc_type, reasoning
                
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return False, 0.0, 'unknown', 'API error'
                
        except Exception as e:
            logger.error(f"Error in phi4-mini analysis: {e}")
            # Fallback to quick check results
            if quick_check['potential_boundary'] and quick_check['detected_patterns']:
                return True, 0.6, quick_check['detected_patterns'][0], 'Fallback to pattern detection'
            return False, 0.0, 'unknown', f'Error: {str(e)}'
    
    def _prepare_page_text(self, page: PageInfo, focus_headers: bool = True) -> str:
        """
        Prepare page text optimized for phi4-mini's context limit.
        """
        if not page.text_content:
            return "[EMPTY PAGE]"
            
        text = page.text_content
        
        # Compress whitespace
        if self.phi_config.compress_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n', text)
            
        # Focus on headers and beginning
        if focus_headers and len(text) > self.phi_config.max_chars_per_page:
            # Take first portion and look for headers
            lines = text.split('\n')
            important_lines = []
            char_count = 0
            
            # First, get lines that look like headers
            for line in lines[:20]:  # Check first 20 lines
                if (len(line.strip()) < 100 and  # Short lines often headers
                    (line.isupper() or  # All caps
                     re.match(r'^[A-Z][A-Z\s]+:', line) or  # Header pattern
                     any(kw in line.lower() for pattern in self.DOCUMENT_PATTERNS.values() 
                         for kw in pattern.keywords[:3]))):
                    important_lines.append(line)
                    char_count += len(line)
                    
            # Then add remaining content up to limit
            remaining_chars = self.phi_config.max_chars_per_page - char_count
            if remaining_chars > 100:
                text_remainder = '\n'.join(lines[len(important_lines):])
                if len(text_remainder) > remaining_chars:
                    text_remainder = text_remainder[:remaining_chars] + "..."
                important_lines.append(text_remainder)
                
            text = '\n'.join(important_lines)
        elif len(text) > self.phi_config.max_chars_per_page:
            text = text[:self.phi_config.max_chars_per_page] + "..."
            
        return text
    
    def _create_phi4_prompt(
        self, 
        prev_text: str, 
        curr_text: str,
        detected_patterns: List[str]
    ) -> str:
        """
        Create an optimized prompt for phi4-mini.
        
        Key optimizations:
        1. Concise, direct instructions
        2. Few-shot examples for common cases
        3. Structured output format
        4. Focus on specific indicators
        """
        
        # Build pattern hints
        pattern_hint = ""
        if detected_patterns:
            pattern_hint = f"\nDetected patterns on current page: {', '.join(detected_patterns)}"
        
        # Few-shot examples for better accuracy
        few_shot = ""
        if self.phi_config.use_few_shot:
            few_shot = """
Examples:
1. Previous: "...Sincerely, John Smith" → Current: "From: jane@example.com To: john@example.com Subject: Meeting"
   Result: {"boundary": true, "confidence": 0.9, "type": "email", "reason": "Letter ending followed by email header"}

2. Previous: "...Page 3 of 3" → Current: "INVOICE #12345 Bill To: ABC Company"
   Result: {"boundary": true, "confidence": 0.85, "type": "invoice", "reason": "Document end followed by invoice header"}

3. Previous: "...total due: $5,000" → Current: "...payment terms: Net 30"
   Result: {"boundary": false, "confidence": 0.8, "type": "continuation", "reason": "Same invoice continues"}
"""
        
        prompt = f"""Analyze if these pages are from different documents. Focus on document headers, type changes, and context shifts.
{few_shot}
PREVIOUS PAGE (ending):
{prev_text}

CURRENT PAGE (beginning):
{curr_text}
{pattern_hint}

Determine if current page starts a new document. Consider:
- Document headers (Invoice #, RFI #, From:/To:, etc.)
- Context changes (different topics, dates, authors)
- Format changes (email to invoice, letter to report)
- Construction documents (RFIs, submittals, change orders)

Respond in JSON:
{{"boundary": true/false, "confidence": 0.0-1.0, "type": "email|invoice|rfi|submittal|change_order|contract|letter|report|other", "reason": "brief explanation"}}"""

        return prompt
    
    def _parse_text_response(self, response: str) -> Dict[str, Any]:
        """Fallback parser for non-JSON responses."""
        result = {
            'boundary': False,
            'confidence': 0.5,
            'type': 'unknown',
            'reason': 'Could not parse response'
        }
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            try:
                result = json.loads(json_match.group())
            except:
                pass
                
        # Fallback: look for keywords
        response_lower = response.lower()
        if 'boundary: true' in response_lower or 'new document' in response_lower:
            result['boundary'] = True
            
        # Extract confidence if mentioned
        conf_match = re.search(r'confidence[:\s]+([0-9.]+)', response_lower)
        if conf_match:
            result['confidence'] = float(conf_match.group(1))
            
        return result
    
    def _map_to_document_type(self, llm_type: str) -> DocumentType:
        """Map LLM-detected type to DocumentType enum."""
        mapping = {
            'email': DocumentType.EMAIL,
            'invoice': DocumentType.INVOICE,
            'contract': DocumentType.CONTRACT,
            'letter': DocumentType.LETTER,
            'report': DocumentType.REPORT,
            'rfi': DocumentType.REPORT,
            'submittal': DocumentType.REPORT,
            'change_order': DocumentType.FORM,
            'continuation': None,  # Not a new document
            'other': DocumentType.OTHER
        }
        return mapping.get(llm_type.lower(), DocumentType.OTHER)
    
    def get_signal_type(self) -> SignalType:
        return ExtendedSignalType.LLM_SEMANTIC
    
    async def __aexit__(self, *args):
        """Clean up HTTP client."""
        await self.client.aclose()


# Configuration examples for phi4-mini integration

def get_phi4_mini_config(
    enable_construction_boost: bool = True,
    fast_mode: bool = False
) -> DetectorConfig:
    """
    Get optimized configuration for phi4-mini boundary detection.
    
    Args:
        enable_construction_boost: Whether to boost construction document detection
        fast_mode: Use faster settings with slightly lower accuracy
    """
    phi4_settings = {
        'model_name': 'phi4-mini:3.8b',
        'ollama_url': 'http://localhost:11434',
        'temperature': 0.1 if not fast_mode else 0.2,
        'max_context_tokens': 4096 if not fast_mode else 2048,
        'max_chars_per_page': 800 if not fast_mode else 500,
        'context_window_pages': 2 if not fast_mode else 1,
        'batch_size': 5 if not fast_mode else 10,
        'use_few_shot': True if not fast_mode else False,
        'compress_whitespace': True,
        'focus_on_headers': True,
    }
    
    return DetectorConfig(
        name="phi4_mini",
        enabled=True,
        weight=0.85 if not enable_construction_boost else 0.9,
        config={'phi4_settings': phi4_settings}
    )


def get_enhanced_detector_with_phi4() -> List[DetectorConfig]:
    """
    Get a complete detector configuration with phi4-mini as primary LLM.
    
    Returns configuration for:
    1. Rule-based (fast, baseline)
    2. Phi4-mini (primary LLM)
    3. Construction-specific (domain expertise)
    """
    return [
        # Traditional rule-based detector
        DetectorConfig(
            name="rule_based",
            enabled=True,
            weight=0.7,  # Lower weight, but still important for speed
            config={'min_confidence': 0.6, 'min_signals': 1}
        ),
        
        # Phi4-mini as primary LLM detector
        get_phi4_mini_config(enable_construction_boost=True, fast_mode=False),
        
        # Construction-specific patterns
        DetectorConfig(
            name="construction",
            enabled=True,
            weight=1.1,  # High weight for construction expertise
            config={}
        ),
        
        # Optional: Disable VLM by default (resource intensive)
        DetectorConfig(
            name="vlm",
            enabled=False,
            weight=0.7,
            config={'vlm_model': 'dit-base'}
        )
    ]


# Example usage with phi4-mini
async def example_phi4_mini_detection():
    """Example of using phi4-mini for boundary detection."""
    from .boundary_detector_enhanced import EnhancedBoundaryDetector
    
    # Register phi4-mini detector plugin
    # This would be done in the main application initialization
    from .boundary_detector_enhanced import detector_classes
    detector_classes['phi4_mini'] = Phi4MiniBoundaryDetector
    
    # Configure enhanced detector with phi4-mini
    detector = EnhancedBoundaryDetector(
        min_confidence=0.65,
        min_signals=1,
        detector_configs=get_enhanced_detector_with_phi4()
    )
    
    # Process pages
    pages = []  # Would be populated from DocumentProcessor
    
    # Detect boundaries
    boundaries = await detector.detect_boundaries(pages, context_window=2)
    
    # Analyze results
    for boundary in boundaries:
        print(f"\nBoundary at page {boundary.start_page}:")
        print(f"  Confidence: {boundary.confidence:.2%}")
        print(f"  Document type: {boundary.document_type}")
        
        # Check which detectors contributed
        detector_types = defaultdict(int)
        for signal in boundary.signals:
            if signal.metadata.get('model') == 'phi4-mini:3.8b':
                print(f"  Phi4-mini: {signal.description}")
            detector_types[signal.type] += 1
            
        print(f"  Signal types: {dict(detector_types)}")


if __name__ == "__main__":
    # Test configuration
    config = get_phi4_mini_config(enable_construction_boost=True, fast_mode=False)
    print("Phi4-mini configuration:")
    print(json.dumps(config.config, indent=2))