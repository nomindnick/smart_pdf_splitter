# Phi4-mini:3.8b Boundary Detection Design

## Overview

This document describes the design and implementation of phi4-mini:3.8b integration for enhanced boundary detection in the Smart PDF Splitter. The design focuses on leveraging phi4-mini's capabilities while maintaining construction document expertise and adding universal document support.

## Architecture

### 1. Plugin-Based Integration

The phi4-mini detector is implemented as a plugin in the existing enhanced boundary detection system:

```python
# Detector hierarchy
BoundaryDetectorPlugin (ABC)
├── RuleBasedDetectorPlugin (existing patterns)
├── LLMBoundaryDetectorPlugin (generic LLM)
├── Phi4MiniBoundaryDetector (phi4-mini optimized)
├── VLMBoundaryDetectorPlugin (visual analysis)
└── ConstructionSpecificDetectorPlugin (domain expertise)
```

### 2. Key Components

#### Phi4MiniBoundaryDetector
- Optimized specifically for phi4-mini:3.8b model characteristics
- Efficient token management for 8k context window
- Specialized prompts for faster inference
- Pre-filtering to reduce LLM calls

#### Prompt Library
- Multiple prompt strategies (concise, detailed, few-shot, CoT)
- Document-type specific prompts
- Adaptive prompt selection based on context
- Token-optimized formatting

#### Configuration System
- Deployment modes (development, production, high-performance, etc.)
- Document focus settings (general, construction, financial, email)
- Resource management options
- Feature flags for fine-tuning

## Implementation Details

### 1. Context Management

```python
# Efficient context window usage
max_context_tokens: 4096  # Conservative for phi4-mini
max_chars_per_page: 800   # ~200 tokens per page
context_window_pages: 2   # Look at 2 pages before/after
```

### 2. Pre-filtering Strategy

Before calling the LLM, we use pattern matching to identify potential boundaries:

```python
def _quick_boundary_check(self, prev_page, curr_page):
    # Check for document headers
    # Check for keyword patterns
    # Check for significant text length changes
    # Return potential_boundary flag and confidence boost
```

### 3. Prompt Optimization

Prompts are optimized for phi4-mini's strengths:

```python
# Concise, structured prompts
# JSON output format for reliable parsing
# Few-shot examples for common cases
# Focus on specific indicators
```

### 4. Confidence Calculation

The system combines signals from multiple detectors:

```python
# Weighted confidence scoring
phi4_mini_weight: 0.85
rule_based_weight: 0.7
construction_weight: 1.2  # Higher for domain expertise
```

## Usage Examples

### 1. Basic Setup

```python
from backend.src.core.phi4_mini_detector_design import Phi4MiniBoundaryDetector
from backend.src.core.boundary_detector_enhanced import EnhancedBoundaryDetector

# Configure detector
config = get_phi4_mini_config(
    enable_construction_boost=True,
    fast_mode=False
)

# Create enhanced detector with phi4-mini
detector = EnhancedBoundaryDetector(
    min_confidence=0.7,
    detector_configs=[config]
)

# Process PDF
boundaries = await detector.detect_boundaries(pages)
```

### 2. Construction-Focused Configuration

```python
# Get preset for construction documents
config = get_preset_config("construction_expert")

# Or build custom configuration
config = (ConfigBuilder()
    .set_deployment_mode("production")
    .set_document_focus("construction")
    .enable_feature("construction", True)
    .set_resource_limits(max_memory_mb=3072)
    .build())
```

### 3. High-Performance Mode

```python
# For processing large volumes
config = get_preset_config("high_volume")

# Fast mode settings:
# - Smaller context windows
# - Larger batch sizes
# - No few-shot examples
# - Higher temperature for speed
```

## Performance Considerations

### 1. Resource Usage

- **Memory**: ~2-4GB for phi4-mini model
- **Inference**: ~0.5-2 seconds per boundary check
- **Batch processing**: 5-10 pages per batch

### 2. Optimization Strategies

1. **Pre-filtering**: Reduce LLM calls by 60-80%
2. **Batching**: Process multiple boundaries in parallel
3. **Caching**: Store results for repeated patterns
4. **Context compression**: Minimize tokens while preserving information

### 3. Scalability

- Concurrent request handling
- Configurable timeouts and retries
- Graceful fallback to rule-based detection

## Integration Points

### 1. API Endpoints

The phi4-mini detector integrates seamlessly with existing endpoints:

```python
POST /api/documents/upload
GET  /api/documents/{id}/boundaries
PUT  /api/documents/{id}/boundaries
```

### 2. Configuration Options

Users can select detection methods via API:

```python
{
    "detection_config": {
        "mode": "production",
        "focus": "construction",
        "enable_phi4_mini": true
    }
}
```

### 3. Monitoring

Built-in statistics tracking:

```python
- Total LLM calls
- Success/failure rates
- Average inference time
- Confidence distribution
```

## Deployment Modes

### 1. Development Mode
- Full logging and debugging
- All features enabled
- Larger context windows
- Temperature: 0.1

### 2. Production Mode
- Balanced performance/accuracy
- Standard feature set
- Optimized context usage
- Temperature: 0.1

### 3. High-Performance Mode
- Minimal context windows
- Larger batch sizes
- No few-shot examples
- Temperature: 0.2

### 4. High-Accuracy Mode
- Maximum context usage
- All enhancement features
- Lower temperature: 0.05
- Multiple validation passes

### 5. Low-Resource Mode
- Minimal memory usage
- Aggressive batching
- Simplified prompts
- Temperature: 0.3

## Advantages of Phi4-mini

1. **Size**: 3.8B parameters fits in modest hardware
2. **Speed**: Fast inference compared to larger models
3. **Quality**: Good understanding of document structure
4. **Efficiency**: Lower resource requirements
5. **Flexibility**: Works well with structured outputs

## Construction Document Expertise

The system maintains specialized knowledge for construction documents:

1. **RFIs** (Request for Information)
2. **Submittals** and shop drawings
3. **Change orders** and modifications
4. **Daily reports** and logs
5. **Meeting minutes**
6. **Safety documents**

These receive confidence boosts and specialized prompts.

## Future Enhancements

1. **Fine-tuning**: Custom phi4-mini model for document boundaries
2. **Multi-model ensemble**: Combine multiple small models
3. **Active learning**: Improve from user corrections
4. **Embedding cache**: Store document embeddings
5. **Streaming support**: Process documents as they arrive

## Configuration Best Practices

1. **Start with presets**: Use provided configurations
2. **Monitor performance**: Track accuracy and speed
3. **Adjust weights**: Fine-tune based on document types
4. **Enable caching**: Reduce redundant processing
5. **Set appropriate timeouts**: Balance speed vs reliability

## Error Handling

The system includes robust error handling:

1. **Timeout protection**: Configurable request timeouts
2. **Fallback detection**: Rule-based backup
3. **Partial results**: Return what's available
4. **Retry logic**: Automatic retries for transient errors
5. **Graceful degradation**: Continue with reduced functionality

## Testing Strategy

1. **Unit tests**: Individual component testing
2. **Integration tests**: Full pipeline validation
3. **Performance tests**: Speed and resource usage
4. **Accuracy tests**: Against ground truth data
5. **Edge cases**: Empty pages, corrupted text, etc.

## Conclusion

The phi4-mini integration provides an excellent balance of accuracy, speed, and resource efficiency for document boundary detection. By combining it with rule-based and construction-specific detectors, the system achieves both universal applicability and domain expertise.