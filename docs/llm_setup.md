# LLM Boundary Detection Setup Guide

This guide explains how to set up and use the LLM-powered boundary detection feature in the Smart PDF Splitter.

## Overview

The Smart PDF Splitter includes advanced LLM-based boundary detection using Phi-4 Mini through Ollama. This provides:

- Intelligent document boundary detection using semantic understanding
- Ambiguous case resolution when pattern matching is uncertain
- Enhanced accuracy for complex document structures
- Contextual analysis of page transitions

## Prerequisites

1. **Ollama Installation**
   - Download and install Ollama from [https://ollama.ai](https://ollama.ai)
   - Verify installation: `ollama --version`

2. **Phi-4 Mini Model**
   - Pull the model: `ollama pull phi4-mini:3.8b`
   - Verify model is available: `ollama list`

## Configuration

### 1. Environment Variables

Create or update your `.env` file in the backend directory:

```env
# LLM Configuration
OLLAMA_URL=http://localhost:11434
LLM_MODEL=phi4-mini:3.8b
LLM_TEMPERATURE=0.1
LLM_TIMEOUT=30
ENABLE_LLM_DETECTION=true
```

### 2. Start Ollama Service

Ensure Ollama is running:

```bash
# Start Ollama service (if not already running)
ollama serve
```

## API Usage

### Enhanced Detection Endpoints

The LLM detection is available through the enhanced API endpoints at `/api/v2`:

#### 1. Analyze Document with LLM

```bash
POST /api/v2/documents/{document_id}/analyze
Content-Type: application/json

{
  "preset": "general",
  "enable_llm": true,
  "llm_model": "phi4-mini:3.8b",
  "adaptive_mode": false,
  "context_window": 3
}
```

#### 2. Compare Detection Methods

```bash
GET /api/v2/documents/{document_id}/compare-methods?include_vlm=false
```

This returns a comparison of different detection methods including LLM-based detection.

#### 3. Available Presets

- `construction`: Optimized for construction documents with LLM support
- `general`: Balanced configuration with LLM enabled
- `high_accuracy`: Maximum accuracy using all detection methods
- `fast`: Pattern-based only (no LLM)

## How LLM Detection Works

### 1. Initial Pattern Detection
The system first performs traditional pattern-based detection looking for:
- Email headers (From:, To:, Subject:)
- Document headers (Invoice #, Purchase Order, etc.)
- Page number resets
- Layout changes

### 2. Ambiguous Case Identification
Pages with confidence scores between 0.4 and 0.75 are flagged as ambiguous.

### 3. LLM Analysis
For ambiguous cases, the LLM analyzes:
- Context from previous and next pages
- Page metadata (word count, formatting)
- Detected signals and their confidence
- Semantic content understanding

### 4. Result Merging
LLM results are merged with pattern detection to:
- Confirm ambiguous boundaries
- Reject false positives
- Add missed boundaries
- Improve confidence scores

## Testing LLM Integration

### 1. Run Integration Test

```bash
cd backend
source ../venv/bin/activate
python test_llm_integration.py
```

### 2. Run Unit Tests

```bash
pytest tests/test_phi4_mini_detector.py -v
```

### 3. Manual Testing

Use the provided test script or API endpoints to test with your documents:

```python
from src.core.phi4_mini_detector import Phi4MiniBoundaryDetector

detector = Phi4MiniBoundaryDetector(
    model_name="phi4-mini:3.8b",
    use_llm_for_ambiguous=True,
    min_confidence=0.6
)

boundaries = detector.detect_boundaries(pages)
```

## Performance Considerations

- **Batch Processing**: LLM calls are batched for efficiency
- **Timeout**: Default 30 seconds per LLM call
- **Context Window**: Adjustable (default 3 pages)
- **Caching**: Results are cached for repeated analysis

## Troubleshooting

### Common Issues

1. **"Model not found" error**
   - Run: `ollama pull phi4-mini:3.8b`
   - Verify: `ollama list`

2. **Connection errors**
   - Check Ollama is running: `curl http://localhost:11434/api/tags`
   - Verify OLLAMA_URL in .env file

3. **Slow performance**
   - Reduce context window size
   - Use "fast" preset for large documents
   - Check system resources

### Debug Mode

Enable debug logging to see detailed LLM analysis:

```python
import logging
logging.getLogger('src.core.phi4_mini_detector').setLevel(logging.DEBUG)
```

## Advanced Configuration

### Custom LLM Models

To use a different model:

1. Pull the model: `ollama pull <model-name>`
2. Update .env: `LLM_MODEL=<model-name>`
3. Adjust temperature and timeout as needed

### Custom Detection Logic

Extend the `Phi4MiniBoundaryDetector` class:

```python
class CustomLLMDetector(Phi4MiniBoundaryDetector):
    def _analyze_single_page(self, pages, page_idx, candidate):
        # Custom analysis logic
        return super()._analyze_single_page(pages, page_idx, candidate)
```

## Best Practices

1. **Model Selection**: Phi-4 Mini provides good balance of speed and accuracy
2. **Confidence Thresholds**: Adjust based on document types
3. **Context Window**: Larger windows improve accuracy but increase processing time
4. **Fallback Strategy**: Always have pattern-based detection as fallback

## Future Enhancements

- Support for additional LLM providers (OpenAI, Anthropic)
- Fine-tuned models for specific document types
- Multi-language support
- Vision-language models for layout analysis