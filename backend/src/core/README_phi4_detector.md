# Phi-4 Mini Boundary Detector

The `Phi4MiniBoundaryDetector` is an enhanced boundary detection module that combines traditional pattern matching with LLM-powered analysis using Phi-4 Mini (3.8B parameters) via Ollama.

## Overview

This detector extends the base `BoundaryDetector` class by adding intelligent analysis capabilities for ambiguous cases where traditional pattern matching alone may not be sufficient. It uses a two-stage approach:

1. **Pattern-based detection**: Fast, rule-based detection for clear boundaries
2. **LLM analysis**: Intelligent analysis for ambiguous cases using Phi-4 Mini

## Features

- **Hybrid approach**: Combines pattern matching with LLM intelligence
- **Pre-filtering**: Only sends ambiguous cases to the LLM to minimize API calls
- **Batch processing**: Processes multiple pages in parallel for efficiency
- **Confidence scoring**: Provides confidence scores for all boundary decisions
- **Explanation capability**: Can explain why a page is or isn't a boundary
- **Fallback mechanism**: Gracefully falls back to pattern-only detection if LLM is unavailable

## Installation

1. Install Ollama:
```bash
# On Linux
curl -fsSL https://ollama.com/install.sh | sh

# On macOS
brew install ollama

# On Windows
# Download from https://ollama.com/download
```

2. Pull the Phi-4 Mini model:
```bash
ollama pull phi4-mini:3.8b
```

3. Install the Python package:
```bash
pip install ollama
```

## Usage

### Basic Usage

```python
from src.core import Phi4MiniBoundaryDetector, DocumentProcessor

# Initialize the detector
detector = Phi4MiniBoundaryDetector(
    model_name="phi4-mini:3.8b",
    use_llm_for_ambiguous=True,
    min_confidence=0.6
)

# Process a document
processor = DocumentProcessor()
pages = list(processor.process_document("path/to/document.pdf"))

# Detect boundaries
boundaries = detector.detect_boundaries(pages)

# Print results
for boundary in boundaries:
    print(f"Document {boundary.start_page}-{boundary.end_page}: "
          f"{boundary.document_type.value} (confidence: {boundary.confidence:.0%})")
```

### Configuration Options

```python
detector = Phi4MiniBoundaryDetector(
    # LLM Configuration
    model_name="phi4-mini:3.8b",      # Ollama model to use
    ollama_host="localhost:11434",    # Ollama API host (optional)
    use_llm_for_ambiguous=True,       # Enable LLM for ambiguous cases
    
    # Processing Configuration
    llm_batch_size=5,                 # Number of pages to analyze in parallel
    llm_timeout=30.0,                 # Timeout for LLM calls in seconds
    
    # Detection Configuration
    min_confidence=0.6,               # Minimum confidence threshold
    min_signals=1,                    # Minimum signals required
    enable_visual_analysis=True       # Enable visual signal detection
)
```

### Advanced Features

#### 1. Explain Boundary Decisions

```python
# Get detailed explanation for why a page is/isn't a boundary
explanation = detector.explain_boundary(pages, page_number=3)
print(explanation)
```

Output:
```
Page 3 Analysis:
- Is boundary: Yes
- Confidence: 85%
- Document type: invoice
- Reasoning: Clear invoice header with document number
- Pattern signals: Document header pattern found, Layout change
```

#### 2. Compare Detection Methods

```python
# Compare results with and without LLM
detector_no_llm = Phi4MiniBoundaryDetector(use_llm_for_ambiguous=False)
detector_with_llm = Phi4MiniBoundaryDetector(use_llm_for_ambiguous=True)

boundaries_pattern = detector_no_llm.detect_boundaries(pages)
boundaries_llm = detector_with_llm.detect_boundaries(pages)

print(f"Pattern-only: {len(boundaries_pattern)} documents")
print(f"Pattern + LLM: {len(boundaries_llm)} documents")
```

#### 3. Custom Ollama Host

```python
# Connect to remote Ollama instance
detector = Phi4MiniBoundaryDetector(
    ollama_host="http://remote-server:11434",
    model_name="phi4-mini:3.8b"
)
```

## How It Works

### 1. Pre-filtering Logic

The detector identifies ambiguous cases based on:
- Confidence scores between 0.4 and 0.75
- Single signals with low confidence
- Conflicting signals

### 2. LLM Analysis

For ambiguous cases, the LLM analyzes:
- Current page content (first 500 chars)
- Previous page content (last 200 chars)
- Next page content (first 200 chars)
- Detected signals and metadata
- Page layout characteristics

### 3. Result Merging

The LLM results can:
- Confirm ambiguous boundaries (increase confidence)
- Reject false positives (remove boundaries)
- Add new boundaries that were missed

## Performance Considerations

- **LLM calls are minimized**: Only ambiguous cases are sent to the LLM
- **Batch processing**: Multiple pages analyzed in parallel
- **Timeout protection**: Configurable timeout prevents hanging
- **Fallback mechanism**: Works without LLM if Ollama is unavailable

## Troubleshooting

### Ollama Connection Issues

```python
# Test Ollama connection
import ollama
client = ollama.Client()
print(client.list())  # Should show available models
```

### Model Not Found

```bash
# Pull the model if not available
ollama pull phi4-mini:3.8b

# List available models
ollama list
```

### Performance Issues

```python
# Reduce batch size for lower memory usage
detector = Phi4MiniBoundaryDetector(
    llm_batch_size=2,  # Process fewer pages at once
    llm_timeout=60.0   # Increase timeout for slower systems
)
```

## Example Demo Script

See `backend/examples/phi4_detector_demo.py` for a complete demonstration:

```bash
# Basic usage
python examples/phi4_detector_demo.py path/to/document.pdf

# Compare with/without LLM
python examples/phi4_detector_demo.py path/to/document.pdf --compare

# Pattern-only mode
python examples/phi4_detector_demo.py path/to/document.pdf --no-llm
```