# Enhanced Boundary Detection Design

## Overview

The enhanced boundary detection system extends the original rule-based detector with a flexible plugin architecture that supports multiple detection strategies including LLM-based semantic analysis, VLM-based visual analysis, and specialized construction document detection.

## Architecture

### Plugin System

The system uses a plugin-based architecture where each detector implements the `BoundaryDetectorPlugin` interface:

```python
class BoundaryDetectorPlugin(ABC):
    @abstractmethod
    async def detect_boundaries(self, pages: List[PageInfo], context_window: int) -> List[BoundaryCandidate]
    
    @abstractmethod
    def get_signal_type(self) -> SignalType
```

### Available Detectors

1. **RuleBasedDetectorPlugin**: Original pattern-matching approach
   - Email headers, document headers, page numbers
   - Fast and reliable for structured documents
   - Weight: 1.0 (baseline)

2. **LLMBoundaryDetectorPlugin**: Semantic understanding via Ollama
   - Analyzes context shifts between pages
   - Understands document meaning and purpose
   - Weight: 0.8 (slightly lower due to potential variability)

3. **VLMBoundaryDetectorPlugin**: Visual layout analysis via Docling
   - Detects visual structure changes
   - Analyzes layout, whitespace, and formatting
   - Weight: 0.7 (supplementary signal)

4. **ConstructionSpecificDetectorPlugin**: Industry-specific patterns
   - RFIs, submittals, change orders, safety docs
   - Higher confidence for construction documents
   - Weight: 1.2 (domain expertise bonus)

## Key Features

### 1. Flexible Configuration

Detectors can be enabled/disabled and weighted independently:

```python
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
            'model_name': 'llama3.2'
        }
    )
]
```

### 2. Multi-Signal Fusion

The system combines signals from multiple detectors:
- Weighted confidence calculation
- Signal diversity bonus (multiple detector agreement)
- Configurable minimum thresholds

### 3. Preset Configurations

Pre-configured detector combinations for common use cases:
- **Construction-focused**: Prioritizes construction patterns + LLM validation
- **General-purpose**: Balanced weighting across methods
- **High-accuracy**: All detectors including VLM
- **Fast**: Rule-based only for speed

### 4. Adaptive Mode

Automatically selects optimal configuration based on document analysis:
- Detects document characteristics (scanned, construction, size)
- Chooses appropriate detector combination
- Post-processes results based on document type

## LLM Integration (Ollama)

The LLM detector uses Ollama for semantic boundary detection:

```python
# Ollama API call
response = await client.post(
    f"{ollama_url}/api/generate",
    json={
        "model": "llama3.2",
        "prompt": boundary_detection_prompt,
        "temperature": 0.1,
        "format": "json"
    }
)
```

Key aspects:
- Context-aware prompts with surrounding page text
- JSON-structured responses for reliability
- Low temperature for consistency
- Configurable context window size

## VLM Integration (Docling)

The VLM detector leverages Docling's visual analysis capabilities:

```python
pipeline_options = PdfPipelineOptions()
pipeline_options.do_layout_analysis = True
pipeline_options.layout_model = "dit-large"  # Document Image Transformer
pipeline_options.generate_page_images = True
```

Visual features analyzed:
- Layout type (single/multi-column)
- Headers and footers
- Whitespace distribution
- Visual similarity between pages

## Construction Document Expertise

Maintains specialized detection for construction industry:

### Supported Document Types
- RFIs (Request for Information)
- Submittals and shop drawings
- Change orders and modifications
- Daily reports and logs
- Safety documents and inspections
- Meeting minutes (OAC meetings)

### Pattern Matching
```python
CONSTRUCTION_PATTERNS = {
    'rfi': {
        'patterns': [
            r'RFI\s*#?\s*\d+',
            r'REQUEST\s+FOR\s+INFORMATION',
            r'RESPONSE\s+REQUIRED\s+BY'
        ],
        'confidence': 0.9
    }
}
```

## Usage Examples

### Basic Usage
```python
# Create enhanced detector
detector = EnhancedBoundaryDetector(
    min_confidence=0.7,
    detector_configs=detector_configs
)

# Detect boundaries
boundaries = await detector.detect_boundaries(pages)
```

### Adaptive Mode
```python
adaptive = AdaptiveBoundaryDetector()
boundaries = await adaptive.detect_with_adaptation(pdf_path)
```

### API Integration
```python
# Enhanced API endpoint
@router.post("/api/v2/documents/{id}/analyze")
async def analyze_document_enhanced(
    document_id: str,
    options: ProcessingOptionsRequest
):
    # Configurable detection with presets
    pass
```

## Performance Considerations

1. **Parallel Processing**: All detectors run concurrently
2. **Resource Management**: VLM disabled by default (GPU intensive)
3. **Caching**: LLM responses can be cached for repeated content
4. **Batch Processing**: Efficient handling of multiple documents

## Future Enhancements

1. **Custom LLM Fine-tuning**: Train models on construction documents
2. **Enhanced VLM Models**: Use newer vision transformers
3. **Learning System**: Adapt weights based on user feedback
4. **Cloud Integration**: Support for cloud-based LLMs/VLMs
5. **Performance Optimization**: GPU acceleration for VLM

## Configuration Best Practices

1. **Start Simple**: Begin with rule-based + construction detectors
2. **Add LLM**: Enable for better semantic understanding
3. **VLM for Complex Layouts**: Enable when visual structure varies
4. **Monitor Performance**: Track processing time and accuracy
5. **Adjust Weights**: Fine-tune based on your document types

## Testing

The enhanced system maintains compatibility with existing tests while adding:
- Mock Ollama server for LLM testing
- Visual feature extraction validation
- Multi-detector fusion testing
- Preset configuration validation