"""
Phi-4 Mini powered boundary detection for intelligent document splitting.

This module uses Ollama with phi4-mini:3.8b to provide context-aware boundary
detection that can understand document semantics and make intelligent decisions
about where documents should be split.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

import ollama
from ollama import Client

from .models import (
    PageInfo,
    Boundary,
    Signal,
    SignalType,
    DocumentType,
)
from .boundary_detector import BoundaryDetector, BoundaryCandidate

logger = logging.getLogger(__name__)


@dataclass
class LLMAnalysis:
    """Results from LLM boundary analysis."""
    is_boundary: bool
    confidence: float
    reasoning: str
    document_type: Optional[DocumentType] = None
    context_signals: List[str] = None
    
    def __post_init__(self):
        if self.context_signals is None:
            self.context_signals = []


class Phi4MiniBoundaryDetector(BoundaryDetector):
    """
    Enhanced boundary detector using Phi-4 Mini for intelligent analysis.
    
    This detector extends the base BoundaryDetector by adding LLM-based
    analysis for ambiguous cases where traditional pattern matching may
    not be sufficient.
    """
    
    # Prompt template for boundary detection
    BOUNDARY_PROMPT = """You are analyzing a PDF document to detect boundaries between different documents.

Context:
- Previous page content (last 200 chars): {prev_content}
- Current page content (first 500 chars): {curr_content}
- Next page content (first 200 chars): {next_content}

Page metadata:
- Current page number: {page_num}
- Previous page word count: {prev_words}
- Current page word count: {curr_words}

Signals detected:
{signals}

Task: Determine if the current page starts a new document.

Consider:
1. Document headers (emails, invoices, letters, etc.)
2. Page numbering resets
3. Major format/layout changes
4. Content continuity
5. Document signatures or closings followed by new headers

Respond in JSON format:
{{
    "is_boundary": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "document_type": "email/invoice/letter/report/other",
    "context_signals": ["signal1", "signal2"]
}}"""

    # Pre-filtering thresholds
    MIN_AMBIGUOUS_CONFIDENCE = 0.4
    MAX_AMBIGUOUS_CONFIDENCE = 0.75
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        ollama_host: Optional[str] = None,
        min_confidence: float = 0.6,
        min_signals: int = 1,
        enable_visual_analysis: bool = True,
        use_llm_for_ambiguous: bool = True,
        llm_batch_size: int = 5,
        llm_timeout: Optional[float] = None
    ):
        """
        Initialize Phi-4 Mini boundary detector.
        
        Args:
            model_name: Ollama model to use
            ollama_host: Ollama API host (defaults to localhost:11434)
            min_confidence: Minimum confidence threshold for boundaries
            min_signals: Minimum number of signals required
            enable_visual_analysis: Whether to use visual signal detection
            use_llm_for_ambiguous: Whether to use LLM for ambiguous cases
            llm_batch_size: Number of pages to analyze in parallel
            llm_timeout: Timeout for LLM calls in seconds
        """
        super().__init__(min_confidence, min_signals, enable_visual_analysis)
        
        # Import settings
        from .config import settings
        
        self.model_name = model_name or settings.llm_model
        self.use_llm_for_ambiguous = use_llm_for_ambiguous
        self.llm_batch_size = llm_batch_size
        self.llm_timeout = llm_timeout if llm_timeout is not None else settings.llm_timeout
        
        # Initialize Ollama client
        try:
            # Use configured ollama URL or default
            host = ollama_host or settings.ollama_url
            self.client = Client(host=host)
            # Test connection
            self._test_ollama_connection()
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama client: {e}")
            logger.warning("Falling back to pattern-based detection only")
            self.use_llm_for_ambiguous = False
            self.client = None
    
    def _test_ollama_connection(self):
        """Test Ollama connection and model availability."""
        try:
            # Check if model is available
            response = self.client.list()
            
            # Handle both dict and object responses
            if hasattr(response, 'models'):
                models_list = response.models
            else:
                models_list = response.get('models', [])
            
            # Extract model names
            model_names = []
            for m in models_list:
                if isinstance(m, dict):
                    model_names.append(m.get('name', ''))
                elif hasattr(m, 'name'):
                    model_names.append(m.name)
            
            if self.model_name not in model_names:
                logger.warning(f"Model {self.model_name} not found in Ollama")
                logger.info(f"Available models: {model_names}")
                logger.info(f"Pulling {self.model_name}...")
                self.client.pull(self.model_name)
                
        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            raise
    
    def detect_boundaries(
        self,
        pages: List[PageInfo],
        context_window: int = 3
    ) -> List[Boundary]:
        """
        Detect document boundaries using pattern matching and LLM analysis.
        
        Args:
            pages: List of PageInfo objects from document processor
            context_window: Number of pages to look ahead/behind for context
            
        Returns:
            List of detected boundaries with confidence scores
        """
        if not pages:
            return []
        
        # First, use base detector for initial boundary detection
        initial_boundaries = super().detect_boundaries(pages, context_window)
        
        if not self.use_llm_for_ambiguous or not self.client:
            return initial_boundaries
        
        # Identify ambiguous cases for LLM analysis
        ambiguous_candidates = self._identify_ambiguous_cases(
            pages, initial_boundaries
        )
        
        if not ambiguous_candidates:
            logger.info("No ambiguous cases found, using pattern-based results")
            return initial_boundaries
        
        logger.info(f"Found {len(ambiguous_candidates)} ambiguous cases for LLM analysis")
        
        # Analyze ambiguous cases with LLM
        llm_results = self._analyze_with_llm(pages, ambiguous_candidates)
        
        # Merge results
        final_boundaries = self._merge_results(
            initial_boundaries, llm_results, pages
        )
        
        return final_boundaries
    
    def _identify_ambiguous_cases(
        self,
        pages: List[PageInfo],
        boundaries: List[Boundary]
    ) -> List[Tuple[int, BoundaryCandidate]]:
        """
        Identify pages that might be boundaries but have ambiguous signals.
        
        Returns list of (page_index, candidate) tuples for LLM analysis.
        """
        ambiguous = []
        boundary_pages = {b.start_page for b in boundaries}
        
        # Re-analyze each page to find ambiguous cases
        for i in range(1, len(pages)):  # Skip first page
            page = pages[i]
            
            # Skip if already a high-confidence boundary
            if page.page_number in boundary_pages:
                boundary = next(b for b in boundaries if b.start_page == page.page_number)
                if boundary.confidence >= self.MAX_AMBIGUOUS_CONFIDENCE:
                    continue
            
            # Collect all signals for this page
            signals = []
            
            # Check various signals (similar to base detector)
            email_signal = self._detect_email_header(page)
            if email_signal:
                signals.append(email_signal)
                
            doc_signal = self._detect_document_header(page)
            if doc_signal:
                signals.append(doc_signal)
                
            page_num_signal = self._detect_page_number_reset(page)
            if page_num_signal:
                signals.append(page_num_signal)
                
            if i > 0:
                layout_signal = self._detect_layout_change(pages[i-1], page)
                if layout_signal:
                    signals.append(layout_signal)
                    
                type_signal = self._detect_document_type_change(pages[i-1], page)
                if type_signal:
                    signals.append(type_signal)
            
            # Calculate confidence
            if signals:
                confidence = self._calculate_confidence(signals)
                
                # Check if it's in the ambiguous range
                if (self.MIN_AMBIGUOUS_CONFIDENCE <= confidence <= self.MAX_AMBIGUOUS_CONFIDENCE or
                    (len(signals) == 1 and confidence < self.min_confidence)):
                    
                    candidate = BoundaryCandidate(
                        page_number=page.page_number,
                        signals=signals,
                        confidence=confidence,
                        suggested_type=self._detect_document_type(page)
                    )
                    ambiguous.append((i, candidate))
        
        return ambiguous
    
    def _analyze_with_llm(
        self,
        pages: List[PageInfo],
        candidates: List[Tuple[int, BoundaryCandidate]]
    ) -> List[Tuple[int, LLMAnalysis]]:
        """
        Analyze ambiguous candidates using the LLM.
        
        Returns list of (page_index, analysis) tuples.
        """
        results = []
        
        # Process in batches for efficiency
        for i in range(0, len(candidates), self.llm_batch_size):
            batch = candidates[i:i + self.llm_batch_size]
            batch_results = []
            
            # Use ThreadPoolExecutor for parallel LLM calls
            with ThreadPoolExecutor(max_workers=min(len(batch), 5)) as executor:
                futures = []
                
                for page_idx, candidate in batch:
                    future = executor.submit(
                        self._analyze_single_page,
                        pages, page_idx, candidate
                    )
                    futures.append((page_idx, future))
                
                # Collect results with timeout
                for page_idx, future in futures:
                    try:
                        analysis = future.result(timeout=self.llm_timeout)
                        if analysis:
                            batch_results.append((page_idx, analysis))
                    except Exception as e:
                        logger.error(f"LLM analysis failed for page {page_idx}: {e}")
            
            results.extend(batch_results)
        
        return results
    
    def _analyze_single_page(
        self,
        pages: List[PageInfo],
        page_idx: int,
        candidate: BoundaryCandidate
    ) -> Optional[LLMAnalysis]:
        """Analyze a single page with the LLM."""
        try:
            # Prepare context
            curr_page = pages[page_idx]
            prev_page = pages[page_idx - 1] if page_idx > 0 else None
            next_page = pages[page_idx + 1] if page_idx < len(pages) - 1 else None
            
            # Extract text snippets
            curr_content = (curr_page.text_content or "")[:500]
            prev_content = (prev_page.text_content or "")[-200:] if prev_page else "N/A"
            next_content = (next_page.text_content or "")[:200] if next_page else "N/A"
            
            # Format signals
            signals_text = "\n".join([
                f"- {s.type.value}: {s.description} (confidence: {s.confidence:.2f})"
                for s in candidate.signals
            ])
            
            # Build prompt
            prompt = self.BOUNDARY_PROMPT.format(
                prev_content=prev_content,
                curr_content=curr_content,
                next_content=next_content,
                page_num=curr_page.page_number,
                prev_words=prev_page.word_count if prev_page else 0,
                curr_words=curr_page.word_count,
                signals=signals_text or "No strong signals detected"
            )
            
            # Call LLM
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for consistency
                    "top_p": 0.9,
                    "num_predict": 200,  # Limit response length
                }
            )
            
            # Parse response
            result_text = response['response'].strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group())
                
                # Map document type string to enum
                doc_type_str = result_json.get('document_type', 'other').lower()
                doc_type = self._map_document_type(doc_type_str)
                
                return LLMAnalysis(
                    is_boundary=result_json.get('is_boundary', False),
                    confidence=float(result_json.get('confidence', 0.5)),
                    reasoning=result_json.get('reasoning', 'No reasoning provided'),
                    document_type=doc_type,
                    context_signals=result_json.get('context_signals', [])
                )
            else:
                logger.warning(f"Failed to parse LLM response: {result_text}")
                return None
                
        except Exception as e:
            logger.error(f"Error analyzing page {page_idx}: {e}")
            return None
    
    def _map_document_type(self, type_str: str) -> DocumentType:
        """Map string document type to DocumentType enum."""
        type_map = {
            'email': DocumentType.EMAIL,
            'invoice': DocumentType.INVOICE,
            'letter': DocumentType.LETTER,
            'report': DocumentType.REPORT,
            'contract': DocumentType.CONTRACT,
            'form': DocumentType.FORM,
            'memo': DocumentType.MEMO,
            'receipt': DocumentType.RECEIPT,
            'presentation': DocumentType.PRESENTATION,
            'spreadsheet': DocumentType.SPREADSHEET,
            # Map common variations
            'purchase_order': DocumentType.INVOICE,  # Treat PO as invoice
            'po': DocumentType.INVOICE,
            'quote': DocumentType.INVOICE,  # Treat quote as invoice
            'quotation': DocumentType.INVOICE,
        }
        return type_map.get(type_str.lower(), DocumentType.OTHER)
    
    def _merge_results(
        self,
        initial_boundaries: List[Boundary],
        llm_results: List[Tuple[int, LLMAnalysis]],
        pages: List[PageInfo]
    ) -> List[Boundary]:
        """
        Merge initial boundaries with LLM analysis results.
        
        The LLM results can:
        1. Confirm ambiguous boundaries (increase confidence)
        2. Reject false positives (remove boundaries)
        3. Add new boundaries that were missed
        """
        # Create a mutable copy of boundaries
        boundaries = initial_boundaries.copy()
        boundary_pages = {b.start_page: b for b in boundaries}
        
        # Process LLM results
        for page_idx, analysis in llm_results:
            page = pages[page_idx]
            page_num = page.page_number
            
            if analysis.is_boundary and analysis.confidence >= self.min_confidence:
                # LLM confirms this is a boundary
                if page_num in boundary_pages:
                    # Update existing boundary with higher confidence
                    boundary = boundary_pages[page_num]
                    boundary.confidence = max(boundary.confidence, analysis.confidence)
                    
                    # Add LLM signal
                    boundary.signals.append(Signal(
                        type=SignalType.TEXT_PATTERN,
                        confidence=analysis.confidence,
                        page_number=page_num,
                        description=f"LLM analysis: {analysis.reasoning}"
                    ))
                    
                    # Update document type if provided
                    if analysis.document_type:
                        boundary.document_type = analysis.document_type
                else:
                    # Add new boundary detected by LLM
                    new_boundary = Boundary(
                        start_page=page_num,
                        end_page=page_num,  # Will be updated later
                        confidence=analysis.confidence,
                        signals=[Signal(
                            type=SignalType.TEXT_PATTERN,
                            confidence=analysis.confidence,
                            page_number=page_num,
                            description=f"LLM detected: {analysis.reasoning}"
                        )],
                        document_type=analysis.document_type
                    )
                    boundaries.append(new_boundary)
                    
            elif not analysis.is_boundary and page_num in boundary_pages:
                # LLM rejects this boundary if it has low confidence
                boundary = boundary_pages[page_num]
                if boundary.confidence < 0.7:  # Only remove low-confidence boundaries
                    boundaries = [b for b in boundaries if b.start_page != page_num]
        
        # Sort boundaries by start page
        boundaries.sort(key=lambda b: b.start_page)
        
        # Recalculate end pages
        for i in range(len(boundaries) - 1):
            boundaries[i].end_page = boundaries[i + 1].start_page - 1
        if boundaries and pages:
            boundaries[-1].end_page = pages[-1].page_number
        
        logger.info(f"Final boundary count: {len(boundaries)} "
                   f"(initial: {len(initial_boundaries)}, "
                   f"LLM analyzed: {len(llm_results)})")
        
        return boundaries
    
    def explain_boundary(
        self,
        pages: List[PageInfo],
        page_number: int
    ) -> Optional[str]:
        """
        Get a detailed explanation of why a page is or isn't a boundary.
        
        This is useful for debugging and user understanding.
        """
        if not self.client or page_number < 1 or page_number > len(pages):
            return None
            
        page_idx = page_number - 1
        page = pages[page_idx]
        
        # Collect all signals
        signals = []
        if self._detect_email_header(page):
            signals.append("Email headers detected")
        if self._detect_document_header(page):
            signals.append("Document header pattern found")
        if self._detect_page_number_reset(page):
            signals.append("Page numbering reset")
        if page_idx > 0 and self._detect_layout_change(pages[page_idx-1], page):
            signals.append("Significant layout change")
        if page.is_mostly_empty:
            signals.append("Page is mostly empty")
            
        # Get LLM explanation
        candidate = BoundaryCandidate(
            page_number=page_number,
            signals=[],  # We'll describe signals in text
            confidence=0.5,
            suggested_type=None
        )
        
        analysis = self._analyze_single_page(pages, page_idx, candidate)
        
        if analysis:
            explanation = f"Page {page_number} Analysis:\n"
            explanation += f"- Is boundary: {'Yes' if analysis.is_boundary else 'No'}\n"
            explanation += f"- Confidence: {analysis.confidence:.0%}\n"
            explanation += f"- Document type: {analysis.document_type.value if analysis.document_type else 'Unknown'}\n"
            explanation += f"- Reasoning: {analysis.reasoning}\n"
            if signals:
                explanation += f"- Pattern signals: {', '.join(signals)}\n"
            return explanation
        else:
            return f"Unable to analyze page {page_number}"