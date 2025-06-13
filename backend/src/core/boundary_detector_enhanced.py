"""
Enhanced boundary detector with LLM integration and plugin architecture.

This module extends the base boundary detection with:
1. Plugin/strategy pattern for swappable detection methods
2. LLM-based boundary detection using Ollama
3. VLM (Vision Language Model) support via Docling
4. Maintains construction document expertise while adding universality
"""

import re
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Type
from dataclasses import dataclass
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import httpx
from docling.datamodel.base_models import DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from .models import (
    PageInfo, 
    Boundary, 
    Signal, 
    SignalType, 
    DocumentType,
    BoundingBox
)
from .boundary_detector import BoundaryDetector, BoundaryCandidate

logger = logging.getLogger(__name__)


# Define additional signal types for new detection methods
class ExtendedSignalType(Enum):
    """Extended signal types including LLM-based detections."""
    LLM_CONTEXT = "llm_context"
    LLM_SEMANTIC = "llm_semantic"
    VLM_VISUAL = "vlm_visual"
    CONSTRUCTION_PATTERN = "construction_pattern"


@dataclass
class DetectorConfig:
    """Configuration for a boundary detector plugin."""
    name: str
    enabled: bool = True
    weight: float = 1.0
    config: Dict[str, Any] = None


class BoundaryDetectorPlugin(ABC):
    """
    Abstract base class for boundary detection plugins.
    
    Each plugin implements a specific detection strategy that can be
    enabled/disabled and weighted for confidence calculation.
    """
    
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.name = config.name
        self.weight = config.weight
        self.enabled = config.enabled
        
    @abstractmethod
    async def detect_boundaries(
        self, 
        pages: List[PageInfo],
        context_window: int = 3
    ) -> List[BoundaryCandidate]:
        """
        Detect boundaries using this plugin's method.
        
        Args:
            pages: List of pages to analyze
            context_window: Number of pages to look ahead/behind
            
        Returns:
            List of boundary candidates with signals
        """
        pass
    
    @abstractmethod
    def get_signal_type(self) -> SignalType:
        """Get the primary signal type this detector produces."""
        pass


class RuleBasedDetectorPlugin(BoundaryDetectorPlugin):
    """
    Rule-based detector using the original pattern matching approach.
    Maintains construction document expertise.
    """
    
    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.base_detector = BoundaryDetector(
            min_confidence=config.config.get('min_confidence', 0.6),
            min_signals=config.config.get('min_signals', 1)
        )
        
    async def detect_boundaries(
        self, 
        pages: List[PageInfo],
        context_window: int = 3
    ) -> List[BoundaryCandidate]:
        """Use existing rule-based detection."""
        # Run synchronously but wrap in async
        boundaries = await asyncio.get_event_loop().run_in_executor(
            None,
            self.base_detector.detect_boundaries,
            pages,
            context_window
        )
        
        # Convert boundaries to candidates
        candidates = []
        for boundary in boundaries:
            candidate = BoundaryCandidate(
                page_number=boundary.start_page,
                signals=boundary.signals,
                confidence=boundary.confidence,
                suggested_type=boundary.document_type
            )
            candidates.append(candidate)
            
        return candidates
    
    def get_signal_type(self) -> SignalType:
        return SignalType.TEXT_PATTERN


class LLMBoundaryDetectorPlugin(BoundaryDetectorPlugin):
    """
    LLM-based boundary detection using Ollama for semantic understanding.
    
    This detector analyzes text content using an LLM to identify:
    - Semantic document boundaries
    - Context shifts between documents
    - Document type changes based on content understanding
    """
    
    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.ollama_url = config.config.get('ollama_url', 'http://localhost:11434')
        self.model_name = config.config.get('model_name', 'llama3.2')
        self.temperature = config.config.get('temperature', 0.1)
        self.max_context_chars = config.config.get('max_context_chars', 2000)
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def detect_boundaries(
        self, 
        pages: List[PageInfo],
        context_window: int = 3
    ) -> List[BoundaryCandidate]:
        """Detect boundaries using LLM semantic analysis."""
        candidates = []
        
        # Process pages with context windows
        for i in range(1, len(pages)):  # Start from second page
            # Get context from surrounding pages
            start_idx = max(0, i - context_window)
            end_idx = min(len(pages), i + context_window + 1)
            context_pages = pages[start_idx:end_idx]
            
            # Analyze if this is a boundary
            is_boundary, confidence, doc_type = await self._analyze_boundary(
                pages[i-1],  # Previous page
                pages[i],    # Current page
                context_pages
            )
            
            if is_boundary:
                signal = Signal(
                    type=ExtendedSignalType.LLM_SEMANTIC,
                    confidence=confidence,
                    page_number=pages[i].page_number,
                    description=f"LLM detected semantic boundary (type: {doc_type})",
                    metadata={
                        'detected_type': doc_type,
                        'model': self.model_name
                    }
                )
                
                candidate = BoundaryCandidate(
                    page_number=pages[i].page_number,
                    signals=[signal],
                    confidence=confidence,
                    suggested_type=self._map_to_document_type(doc_type)
                )
                candidates.append(candidate)
                
        return candidates
    
    async def _analyze_boundary(
        self,
        prev_page: PageInfo,
        curr_page: PageInfo,
        context_pages: List[PageInfo]
    ) -> Tuple[bool, float, str]:
        """Use LLM to analyze if there's a boundary between pages."""
        
        # Prepare context for LLM
        prev_text = self._truncate_text(prev_page.text_content, self.max_context_chars // 2)
        curr_text = self._truncate_text(curr_page.text_content, self.max_context_chars // 2)
        
        prompt = self._create_boundary_prompt(prev_text, curr_text)
        
        try:
            # Call Ollama API
            response = await self.client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False,
                    "format": "json"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = json.loads(result.get('response', '{}'))
                
                is_boundary = analysis.get('is_boundary', False)
                confidence = analysis.get('confidence', 0.0)
                doc_type = analysis.get('document_type', 'other')
                
                return is_boundary, confidence, doc_type
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return False, 0.0, 'unknown'
                
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return False, 0.0, 'unknown'
    
    def _create_boundary_prompt(self, prev_text: str, curr_text: str) -> str:
        """Create prompt for boundary detection."""
        return f"""You are a document boundary detection expert. Analyze if there is a document boundary between these two pages.

PREVIOUS PAGE CONTENT:
{prev_text}

CURRENT PAGE CONTENT:
{curr_text}

Analyze if these pages belong to different documents. Consider:
1. Document type changes (email to invoice, letter to contract, etc.)
2. Context shifts (different topics, authors, dates)
3. Document headers and formatting changes
4. Construction industry patterns (RFIs, submittals, change orders, etc.)

Respond in JSON format:
{{
    "is_boundary": true/false,
    "confidence": 0.0-1.0,
    "document_type": "email|invoice|contract|letter|rfi|submittal|change_order|report|other",
    "reasoning": "brief explanation"
}}"""
    
    def _truncate_text(self, text: str, max_chars: int) -> str:
        """Truncate text to maximum characters while preserving words."""
        if not text or len(text) <= max_chars:
            return text or ""
            
        # Try to break at word boundary
        truncated = text[:max_chars]
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.8:  # If we found a space in last 20%
            truncated = truncated[:last_space]
            
        return truncated + "..."
    
    def _map_to_document_type(self, llm_type: str) -> DocumentType:
        """Map LLM document type to our DocumentType enum."""
        mapping = {
            'email': DocumentType.EMAIL,
            'invoice': DocumentType.INVOICE,
            'contract': DocumentType.CONTRACT,
            'letter': DocumentType.LETTER,
            'report': DocumentType.REPORT,
            'rfi': DocumentType.REPORT,
            'submittal': DocumentType.REPORT,
            'change_order': DocumentType.FORM
        }
        return mapping.get(llm_type.lower(), DocumentType.OTHER)
    
    def get_signal_type(self) -> SignalType:
        return ExtendedSignalType.LLM_SEMANTIC
    
    async def __aexit__(self, *args):
        """Clean up HTTP client."""
        await self.client.aclose()


class VLMBoundaryDetectorPlugin(BoundaryDetectorPlugin):
    """
    Vision Language Model detector using Docling's VLM capabilities.
    
    Analyzes visual page layout and structure to detect boundaries based on:
    - Visual separators and layout changes
    - Page structure and formatting
    - Visual document type indicators
    """
    
    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.enable_vlm = config.config.get('enable_vlm', True)
        self.vlm_model = config.config.get('vlm_model', 'dit-base')
        self._setup_vlm_pipeline()
        
    def _setup_vlm_pipeline(self):
        """Configure Docling for VLM analysis."""
        pipeline_options = PdfPipelineOptions()
        
        # Enable vision model for layout analysis
        pipeline_options.do_ocr = True
        pipeline_options.do_layout_analysis = True
        pipeline_options.layout_model = self.vlm_model
        
        # Configure for visual analysis
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        
    async def detect_boundaries(
        self, 
        pages: List[PageInfo],
        context_window: int = 3
    ) -> List[BoundaryCandidate]:
        """Detect boundaries using visual analysis."""
        candidates = []
        
        # Analyze visual changes between consecutive pages
        for i in range(1, len(pages)):
            visual_change_score = self._analyze_visual_change(
                pages[i-1],
                pages[i]
            )
            
            if visual_change_score > 0.6:  # Threshold for visual boundary
                signal = Signal(
                    type=ExtendedSignalType.VLM_VISUAL,
                    confidence=visual_change_score,
                    page_number=pages[i].page_number,
                    description="VLM detected visual boundary",
                    metadata={
                        'visual_score': visual_change_score,
                        'model': self.vlm_model
                    }
                )
                
                candidate = BoundaryCandidate(
                    page_number=pages[i].page_number,
                    signals=[signal],
                    confidence=visual_change_score,
                    suggested_type=None  # VLM doesn't determine type
                )
                candidates.append(candidate)
                
        return candidates
    
    def _analyze_visual_change(self, prev_page: PageInfo, curr_page: PageInfo) -> float:
        """Analyze visual changes between pages."""
        score = 0.0
        
        # Check layout element changes
        prev_layouts = {elem.get('type', '') for elem in prev_page.layout_elements}
        curr_layouts = {elem.get('type', '') for elem in curr_page.layout_elements}
        
        # Layout type changes
        layout_diff = len(prev_layouts.symmetric_difference(curr_layouts))
        if layout_diff > 2:
            score += 0.3
            
        # Check for significant density changes
        if prev_page.word_count > 0:
            density_ratio = abs(curr_page.word_count - prev_page.word_count) / prev_page.word_count
            if density_ratio > 0.5:
                score += 0.2
                
        # Check for table/image presence changes
        if prev_page.has_tables != curr_page.has_tables:
            score += 0.2
        if prev_page.has_images != curr_page.has_images:
            score += 0.2
            
        # Check aspect ratio changes (orientation)
        prev_aspect = prev_page.width / prev_page.height if prev_page.height > 0 else 1
        curr_aspect = curr_page.width / curr_page.height if curr_page.height > 0 else 1
        if abs(prev_aspect - curr_aspect) > 0.1:
            score += 0.1
            
        return min(score, 1.0)
    
    def get_signal_type(self) -> SignalType:
        return ExtendedSignalType.VLM_VISUAL


class ConstructionSpecificDetectorPlugin(BoundaryDetectorPlugin):
    """
    Specialized detector for construction industry documents.
    
    Detects construction-specific patterns:
    - RFIs (Request for Information)
    - Submittals
    - Change Orders
    - Shop Drawings
    - Progress Reports
    - Safety Documents
    """
    
    CONSTRUCTION_PATTERNS = {
        'rfi': {
            'patterns': [
                r'RFI\s*#?\s*\d+',
                r'REQUEST\s+FOR\s+INFORMATION',
                r'RFI\s+LOG\s+NO',
                r'RESPONSE\s+REQUIRED\s+BY'
            ],
            'type': 'rfi',
            'confidence': 0.9
        },
        'submittal': {
            'patterns': [
                r'SUBMITTAL\s*#?\s*\d+',
                r'SUBMITTAL\s+TRANSMITTAL',
                r'SHOP\s+DRAWING',
                r'PRODUCT\s+DATA',
                r'MATERIAL\s+SAMPLE'
            ],
            'type': 'submittal',
            'confidence': 0.85
        },
        'change_order': {
            'patterns': [
                r'CHANGE\s+ORDER\s*#?\s*\d+',
                r'CONTRACT\s+MODIFICATION',
                r'COST\s+IMPACT',
                r'SCHEDULE\s+IMPACT',
                r'PCO\s*#?\s*\d+'  # Potential Change Order
            ],
            'type': 'change_order',
            'confidence': 0.9
        },
        'daily_report': {
            'patterns': [
                r'DAILY\s+(?:CONSTRUCTION\s+)?REPORT',
                r'WEATHER\s+CONDITIONS',
                r'MANPOWER\s+ON\s+SITE',
                r'WORK\s+PERFORMED\s+TODAY',
                r'EQUIPMENT\s+ON\s+SITE'
            ],
            'type': 'daily_report',
            'confidence': 0.8
        },
        'safety': {
            'patterns': [
                r'SAFETY\s+(?:MEETING|REPORT|INSPECTION)',
                r'JOB\s+HAZARD\s+ANALYSIS',
                r'INCIDENT\s+REPORT',
                r'NEAR\s+MISS',
                r'TOOLBOX\s+TALK'
            ],
            'type': 'safety_document',
            'confidence': 0.85
        },
        'meeting_minutes': {
            'patterns': [
                r'MEETING\s+MINUTES',
                r'ATTENDEES\s*:',
                r'ACTION\s+ITEMS',
                r'NEXT\s+MEETING',
                r'OAC\s+MEETING'  # Owner-Architect-Contractor
            ],
            'type': 'meeting_minutes',
            'confidence': 0.8
        }
    }
    
    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.compiled_patterns = self._compile_patterns()
        
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for efficiency."""
        compiled = {}
        for doc_type, info in self.CONSTRUCTION_PATTERNS.items():
            compiled[doc_type] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in info['patterns']
            ]
        return compiled
        
    async def detect_boundaries(
        self, 
        pages: List[PageInfo],
        context_window: int = 3
    ) -> List[BoundaryCandidate]:
        """Detect construction-specific document boundaries."""
        candidates = []
        
        for i, page in enumerate(pages):
            # Check for construction document patterns
            construction_signal = self._detect_construction_patterns(page)
            
            if construction_signal:
                # Check if this is different from previous page's type
                is_boundary = True
                if i > 0:
                    prev_signal = self._detect_construction_patterns(pages[i-1])
                    if prev_signal and prev_signal.metadata.get('construction_type') == construction_signal.metadata.get('construction_type'):
                        is_boundary = False
                        
                if is_boundary:
                    candidate = BoundaryCandidate(
                        page_number=page.page_number,
                        signals=[construction_signal],
                        confidence=construction_signal.confidence,
                        suggested_type=DocumentType.REPORT  # Most construction docs are reports
                    )
                    candidates.append(candidate)
                    
        return candidates
    
    def _detect_construction_patterns(self, page: PageInfo) -> Optional[Signal]:
        """Detect construction-specific patterns in page."""
        if not page.text_content:
            return None
            
        # Check first 1000 characters for patterns
        text_sample = page.text_content[:1000]
        
        for doc_type, patterns in self.compiled_patterns.items():
            matches = sum(1 for pattern in patterns if pattern.search(text_sample))
            
            if matches >= 2:  # Need at least 2 pattern matches
                info = self.CONSTRUCTION_PATTERNS[doc_type]
                return Signal(
                    type=ExtendedSignalType.CONSTRUCTION_PATTERN,
                    confidence=info['confidence'],
                    page_number=page.page_number,
                    description=f"Construction document detected: {info['type']}",
                    metadata={
                        'construction_type': info['type'],
                        'pattern_matches': matches
                    }
                )
                
        return None
    
    def get_signal_type(self) -> SignalType:
        return ExtendedSignalType.CONSTRUCTION_PATTERN


class EnhancedBoundaryDetector:
    """
    Enhanced boundary detector with plugin architecture.
    
    Combines multiple detection strategies:
    - Rule-based (original patterns)
    - LLM-based (semantic understanding)
    - VLM-based (visual analysis)
    - Construction-specific patterns
    
    Each plugin can be enabled/disabled and weighted for final confidence.
    """
    
    def __init__(
        self,
        min_confidence: float = 0.6,
        min_signals: int = 1,
        detector_configs: Optional[List[DetectorConfig]] = None
    ):
        """
        Initialize enhanced detector with plugin configuration.
        
        Args:
            min_confidence: Minimum confidence for boundary detection
            min_signals: Minimum number of signals required
            detector_configs: List of detector configurations
        """
        self.min_confidence = min_confidence
        self.min_signals = min_signals
        self.detectors: List[BoundaryDetectorPlugin] = []
        
        # Default configuration if none provided
        if detector_configs is None:
            detector_configs = self._get_default_configs()
            
        # Initialize detector plugins
        self._initialize_detectors(detector_configs)
        
    def _get_default_configs(self) -> List[DetectorConfig]:
        """Get default detector configurations."""
        return [
            DetectorConfig(
                name="rule_based",
                enabled=True,
                weight=1.0,
                config={'min_confidence': 0.6, 'min_signals': 1}
            ),
            DetectorConfig(
                name="llm",
                enabled=True,
                weight=0.8,
                config={
                    'ollama_url': 'http://localhost:11434',
                    'model_name': 'llama3.2',
                    'temperature': 0.1
                }
            ),
            DetectorConfig(
                name="construction",
                enabled=True,
                weight=1.2,  # Higher weight for domain expertise
                config={}
            ),
            DetectorConfig(
                name="vlm",
                enabled=False,  # Disabled by default (requires more resources)
                weight=0.7,
                config={'vlm_model': 'dit-base'}
            )
        ]
        
    def _initialize_detectors(self, configs: List[DetectorConfig]):
        """Initialize detector plugins from configurations."""
        detector_classes = {
            'rule_based': RuleBasedDetectorPlugin,
            'llm': LLMBoundaryDetectorPlugin,
            'vlm': VLMBoundaryDetectorPlugin,
            'construction': ConstructionSpecificDetectorPlugin
        }
        
        for config in configs:
            if config.enabled and config.name in detector_classes:
                detector_class = detector_classes[config.name]
                detector = detector_class(config)
                self.detectors.append(detector)
                logger.info(f"Initialized {config.name} detector with weight {config.weight}")
                
    async def detect_boundaries(
        self,
        pages: List[PageInfo],
        context_window: int = 3
    ) -> List[Boundary]:
        """
        Detect boundaries using all enabled detectors.
        
        Combines results from multiple detectors and calculates
        final confidence scores.
        """
        all_candidates: Dict[int, List[BoundaryCandidate]] = defaultdict(list)
        
        # Run all detectors in parallel
        tasks = []
        for detector in self.detectors:
            task = detector.detect_boundaries(pages, context_window)
            tasks.append((detector, task))
            
        # Gather results
        for detector, task in tasks:
            try:
                candidates = await task
                # Group by page number
                for candidate in candidates:
                    all_candidates[candidate.page_number].append(candidate)
                    
            except Exception as e:
                logger.error(f"Error in {detector.name} detector: {e}")
                
        # Combine candidates and create final boundaries
        boundaries = self._combine_candidates(all_candidates, pages)
        
        return boundaries
    
    def _combine_candidates(
        self,
        candidates_by_page: Dict[int, List[BoundaryCandidate]],
        pages: List[PageInfo]
    ) -> List[Boundary]:
        """Combine candidates from multiple detectors."""
        boundaries = []
        
        # Always include first page
        boundaries.append(Boundary(
            start_page=1,
            end_page=1,
            confidence=1.0,
            signals=[Signal(
                type=SignalType.DOCUMENT_HEADER,
                confidence=1.0,
                page_number=1,
                description="First page"
            )],
            document_type=self._detect_document_type_ensemble(pages[0], candidates_by_page.get(1, []))
        ))
        
        # Process each page with candidates
        for page_num, candidates in sorted(candidates_by_page.items()):
            if page_num == 1:
                continue  # Skip first page
                
            # Combine signals from all candidates
            all_signals = []
            type_votes = defaultdict(float)
            
            for candidate in candidates:
                all_signals.extend(candidate.signals)
                if candidate.suggested_type:
                    # Weight type votes by confidence
                    type_votes[candidate.suggested_type] += candidate.confidence
                    
            # Calculate combined confidence
            if all_signals:
                combined_confidence = self._calculate_combined_confidence(all_signals)
                
                # Determine most likely document type
                suggested_type = None
                if type_votes:
                    suggested_type = max(type_votes.items(), key=lambda x: x[1])[0]
                    
                # Create boundary if confidence meets threshold
                if combined_confidence >= self.min_confidence and len(all_signals) >= self.min_signals:
                    # Update previous boundary's end page
                    if boundaries:
                        boundaries[-1].end_page = page_num - 1
                        
                    boundary = Boundary(
                        start_page=page_num,
                        end_page=page_num,
                        confidence=combined_confidence,
                        signals=all_signals,
                        document_type=suggested_type
                    )
                    boundaries.append(boundary)
                    
        # Set last boundary's end page
        if boundaries and pages:
            boundaries[-1].end_page = pages[-1].page_number
            
        return boundaries
    
    def _calculate_combined_confidence(self, signals: List[Signal]) -> float:
        """Calculate combined confidence from multiple signals."""
        if not signals:
            return 0.0
            
        # Group signals by type
        signals_by_type = defaultdict(list)
        for signal in signals:
            signals_by_type[signal.type].append(signal)
            
        # Calculate weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        
        # Get weights from detector configurations
        type_weights = {
            SignalType.DOCUMENT_HEADER: 0.85,
            SignalType.EMAIL_HEADER: 0.9,
            SignalType.PAGE_NUMBER_RESET: 0.7,
            SignalType.LAYOUT_CHANGE: 0.6,
            SignalType.WHITE_SPACE: 0.5,
            SignalType.DOCUMENT_TYPE_CHANGE: 0.8,
            ExtendedSignalType.LLM_SEMANTIC: 0.8,
            ExtendedSignalType.VLM_VISUAL: 0.7,
            ExtendedSignalType.CONSTRUCTION_PATTERN: 1.2,
        }
        
        for signal_type, type_signals in signals_by_type.items():
            # Average confidence for this signal type
            avg_confidence = sum(s.confidence for s in type_signals) / len(type_signals)
            weight = type_weights.get(signal_type, 0.5)
            
            weighted_sum += avg_confidence * weight
            total_weight += weight
            
        if total_weight == 0:
            return 0.0
            
        base_confidence = weighted_sum / total_weight
        
        # Boost for multiple signal types
        signal_diversity = len(signals_by_type)
        if signal_diversity >= 3:
            base_confidence = min(1.0, base_confidence * 1.2)
        elif signal_diversity >= 2:
            base_confidence = min(1.0, base_confidence * 1.1)
            
        return base_confidence
    
    def _detect_document_type_ensemble(
        self,
        page: PageInfo,
        candidates: List[BoundaryCandidate]
    ) -> Optional[DocumentType]:
        """Detect document type using ensemble of methods."""
        # Start with rule-based detection
        base_detector = BoundaryDetector()
        base_type = base_detector._detect_document_type(page)
        
        # Collect type votes from candidates
        type_votes = defaultdict(float)
        if base_type:
            type_votes[base_type] = 1.0
            
        for candidate in candidates:
            if candidate.suggested_type:
                type_votes[candidate.suggested_type] += candidate.confidence
                
        # Return highest voted type
        if type_votes:
            return max(type_votes.items(), key=lambda x: x[1])[0]
            
        return DocumentType.OTHER


# Example usage and integration
async def main():
    """Example of using the enhanced boundary detector."""
    
    # Configure detectors
    detector_configs = [
        DetectorConfig(
            name="rule_based",
            enabled=True,
            weight=1.0,
            config={'min_confidence': 0.6}
        ),
        DetectorConfig(
            name="llm",
            enabled=True,
            weight=0.8,
            config={
                'ollama_url': 'http://localhost:11434',
                'model_name': 'llama3.2',
                'max_context_chars': 1500
            }
        ),
        DetectorConfig(
            name="construction",
            enabled=True,
            weight=1.2,  # Higher weight for construction expertise
            config={}
        )
    ]
    
    # Initialize enhanced detector
    detector = EnhancedBoundaryDetector(
        min_confidence=0.7,
        min_signals=1,
        detector_configs=detector_configs
    )
    
    # Process pages (would come from DocumentProcessor)
    pages = []  # List[PageInfo]
    
    # Detect boundaries
    boundaries = await detector.detect_boundaries(pages, context_window=3)
    
    # Print results
    for boundary in boundaries:
        print(f"Boundary at pages {boundary.page_range}")
        print(f"  Confidence: {boundary.confidence:.2f}")
        print(f"  Type: {boundary.document_type}")
        print(f"  Signals: {len(boundary.signals)}")
        for signal in boundary.signals:
            print(f"    - {signal.type}: {signal.description}")


if __name__ == "__main__":
    asyncio.run(main())