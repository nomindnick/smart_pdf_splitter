# Universal PDF Splitter Design

## Overview

This document outlines the design for making the Smart PDF Splitter more universal while maintaining its strength in construction documents. The design integrates multiple detection methods to achieve high accuracy across diverse document types.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Document Processing Pipeline               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Docling    │───▶│   Feature    │───▶│  Boundary    │  │
│  │   Parser     │    │  Extraction  │    │  Detection   │  │
│  └─────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │         │
│         ▼                    ▼                    ▼         │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │     OCR     │    │    Visual    │    │   Hybrid     │  │
│  │  (Multiple) │    │   Features   │    │  Detector    │  │
│  └─────────────┘    └──────────────┘    └──────────────┘  │
│                              │                    │         │
│                              ▼                    ▼         │
│                     ┌──────────────┐    ┌──────────────┐  │
│                     │     VLM      │    │  Confidence  │  │
│                     │  (Optional)  │    │   Scoring    │  │
│                     └──────────────┘    └──────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Detection Modules:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │    Text      │  │   Visual     │  │    LLM-Based     │  │
│  │  Detector    │  │  Detector    │  │ (phi4-mini:3.8b) │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│         │                  │                    │           │
│         └──────────────────┴────────────────────┘          │
│                            │                                │
│                    ┌───────▼────────┐                      │
│                    │ Signal Merger  │                      │
│                    │ & Confidence   │                      │
│                    └────────────────┘                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Detection Methods

### 1. Enhanced Text-Based Detection (Existing + Improvements)

**Current Strengths:**
- Strong construction document pattern matching
- Email header detection
- Invoice keyword detection
- Page number reset detection

**Enhancements:**
- Multi-language support via configurable patterns
- Fuzzy matching for variations
- Domain-specific pattern plugins
- Expandable document type taxonomy

### 2. Visual Boundary Detection (New)

**Capabilities:**
- Layout structure analysis (columns, margins, orientation)
- Font family and size changes
- Color scheme detection
- Visual separator identification (lines, boxes)
- Header/footer pattern analysis
- Logo and letterhead detection
- Whitespace pattern analysis

**Implementation:**
```python
visual_detector = VisualBoundaryDetector(
    config=VisualProcessingConfig(
        enable_picture_classification=True,
        enable_layout_analysis=True,
        visual_memory_limit_mb=2048
    )
)
```

### 3. LLM-Based Detection with phi4-mini:3.8b (New)

**Approach:**
- Analyzes text transitions between pages
- Context-aware boundary detection
- Handles ambiguous cases
- Provides reasoning for decisions

**Key Features:**
- Pre-filtering to reduce LLM calls by 60-80%
- Optimized prompts for phi4-mini's 8k context window
- Construction document expertise built into prompts
- Fallback to rule-based detection on failures

**Implementation:**
```python
llm_detector = Phi4MiniBoundaryDetector(
    ollama_base_url="http://localhost:11434",
    model_name="phi4-mini:3.8b",
    config=Phi4MiniConfig(
        mode="production",
        document_focus="mixed",
        enable_pre_filtering=True
    )
)
```

### 4. Hybrid Detection System

**Signal Combination:**
```python
# Configurable weights for different detection methods
detector = HybridBoundaryDetector(
    text_weight=0.3,      # Traditional pattern matching
    visual_weight=0.3,    # Visual feature analysis
    llm_weight=0.4,       # LLM-based detection
    confidence_threshold=0.7
)
```

**Confidence Boosting:**
- When multiple detectors agree, confidence increases
- Disagreements trigger additional analysis
- User can see which signals contributed to each boundary

## Configuration Profiles

### 1. Construction Documents (Default)
```python
config = DetectionConfig(
    profile="construction",
    text_weight=0.4,
    visual_weight=0.2,
    llm_weight=0.4,
    enable_construction_patterns=True,
    llm_document_focus="construction"
)
```

### 2. General Business Documents
```python
config = DetectionConfig(
    profile="business",
    text_weight=0.3,
    visual_weight=0.3,
    llm_weight=0.4,
    enable_all_patterns=True,
    llm_document_focus="mixed"
)
```

### 3. Legal Documents
```python
config = DetectionConfig(
    profile="legal",
    text_weight=0.2,
    visual_weight=0.4,
    llm_weight=0.4,
    enable_legal_patterns=True,
    llm_document_focus="general"
)
```

### 4. Low Resource Mode
```python
config = DetectionConfig(
    profile="low_resource",
    text_weight=0.6,
    visual_weight=0.4,
    llm_weight=0.0,  # Disable LLM
    visual_features_minimal=True
)
```

## Memory Management

### Resource Usage by Component

| Component | Base Memory | Per Page | Notes |
|-----------|------------|----------|-------|
| Text Detection | 100MB | 1-2MB | Minimal overhead |
| Visual Detection | 200MB | 50-100MB | With page images |
| LLM Detection | 2-4GB | 10-20MB | Model loaded once |
| VLM (Optional) | 4-10GB | 50MB | Advanced visual understanding |

### Optimization Strategies

1. **Streaming Processing**: Process pages in batches
2. **Feature Selection**: Enable only needed visual features
3. **LLM Pre-filtering**: Reduce LLM calls by 60-80%
4. **Configurable Quality**: Trade accuracy for memory

## Implementation Phases

### Phase 1: Core Infrastructure (Current)
- ✅ Basic text detection
- ✅ Document processor with OCR
- ✅ Plugin architecture

### Phase 2: Enhanced Detection (Proposed)
- [ ] Visual boundary detector
- [ ] LLM integration with phi4-mini
- [ ] Hybrid detection system
- [ ] Configuration profiles

### Phase 3: Advanced Features
- [ ] VLM integration for complex documents
- [ ] Multi-language support
- [ ] Custom pattern training
- [ ] Real-time confidence adjustment

### Phase 4: Production Hardening
- [ ] Performance optimization
- [ ] Comprehensive error handling
- [ ] Monitoring and metrics
- [ ] API rate limiting

## Usage Examples

### Basic Usage
```python
from smart_pdf_splitter import UniversalPDFSplitter

splitter = UniversalPDFSplitter(profile="business")
boundaries = splitter.detect_boundaries("document.pdf")
```

### Advanced Configuration
```python
splitter = UniversalPDFSplitter(
    detection_config={
        "text_weight": 0.3,
        "visual_weight": 0.3,
        "llm_weight": 0.4,
        "enable_ocr": True,
        "ocr_engine": "tesseract",
        "llm_model": "phi4-mini:3.8b",
        "visual_features": ["layout", "fonts", "logos"],
        "confidence_threshold": 0.7
    }
)
```

### Construction-Specific Usage
```python
splitter = UniversalPDFSplitter(
    profile="construction",
    custom_patterns={
        "submittal": r"submittal\s+#?\d+",
        "rfi": r"(request for information|rfi)\s+#?\d+",
        "change_order": r"change order\s+#?\d+"
    }
)
```

## API Changes

### New Endpoints
```
POST /api/documents/upload
  - profile: string (construction|business|legal|custom)
  - detection_config: object (optional)

GET /api/documents/{id}/boundaries
  - include_confidence_details: boolean
  - include_detection_signals: boolean

POST /api/detection/test
  - page_transition: object
  - detection_method: string (text|visual|llm|hybrid)
```

## Benefits

1. **Universal Application**: Works across document types and languages
2. **Maintains Expertise**: Construction document detection remains strong
3. **Configurable**: Different profiles for different use cases
4. **Scalable**: Can add new detection methods as plugins
5. **Transparent**: Shows which signals detected each boundary
6. **Efficient**: Pre-filtering and smart resource management

## Migration Path

1. Current users continue with existing detection
2. New detection methods available via configuration
3. Gradual migration with A/B testing
4. Full hybrid detection as default after validation

## Future Enhancements

1. **Machine Learning**: Train custom models on user feedback
2. **Cloud Integration**: Offload LLM/VLM processing
3. **Pattern Library**: Shareable detection patterns
4. **Auto-tuning**: Adjust weights based on document corpus
5. **Multilingual**: Expand beyond English documents