# Smart PDF Splitter - Implementation Plan

## Project Overview

The Smart PDF Splitter is a modular application that automatically detects document boundaries within multi-document PDFs and splits them into individual PDF files. The system provides a visual interface for users to verify and adjust the automatically detected boundaries before exporting.

### Key Requirements
- Process PDFs containing multiple documents (emails, invoices, reports, etc.)
- Automatically detect document boundaries
- Provide visual interface for boundary verification/adjustment
- Export individual PDFs for each document
- Work locally without GPU (under 16GB RAM usage)
- Cross-platform (Ubuntu, Windows, cloud deployment)
- Modular design for future integration into RAG pipeline

## Technical Architecture

### Core Technologies
- **Backend**: Python 3.11+ with FastAPI
- **Document Processing**: Docling (IBM's document parser)
- **PDF Manipulation**: PyMuPDF (fitz) for splitting operations
- **Frontend**: React + TypeScript with PDF.js
- **Database**: SQLite (local) / PostgreSQL (cloud)
- **Queue**: Celery + Redis for async processing
- **Testing**: pytest (backend), Jest (frontend)

### System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  React Frontend │────▶│  FastAPI Backend│────▶│  Docling Engine │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                        │
         │                       ▼                        ▼
         │              ┌─────────────────┐     ┌─────────────────┐
         │              │                 │     │                 │
         └─────────────▶│  PDF.js Viewer  │     │ Boundary Detector│
                        │                 │     │                 │
                        └─────────────────┘     └─────────────────┘
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
1. **Project Setup**
   - Initialize project structure
   - Set up development environment
   - Configure testing framework
   - Create Docker configurations

2. **Basic Document Processing**
   - Integrate Docling for PDF parsing
   - Extract document structure and metadata
   - Create data models for document representation
   - Implement basic PDF loading and validation

3. **Testing Foundation**
   - Set up test framework with sample PDFs
   - Create test cases based on ground truth JSON
   - Implement comparison logic for boundary detection accuracy

### Phase 2: Boundary Detection Engine (Week 2)
1. **Content-Based Detection**
   - Analyze document headers and footers
   - Detect email patterns (From:, To:, Subject:)
   - Identify invoice/report headers
   - Recognize document type transitions

2. **Visual-Based Detection**
   - Analyze page layout changes
   - Detect white space patterns
   - Identify page numbering resets
   - Track font and style changes

3. **Hybrid Approach**
   - Combine content and visual signals
   - Implement confidence scoring
   - Create fallback strategies
   - Handle edge cases

### Phase 3: Backend API Development (Week 3)
1. **Core API Endpoints**
   - POST /api/documents/upload - Upload PDF
   - GET /api/documents/{id}/boundaries - Get detected boundaries
   - PUT /api/documents/{id}/boundaries - Update boundaries
   - POST /api/documents/{id}/split - Execute split operation
   - GET /api/documents/{id}/download/{doc_index} - Download individual PDF

2. **Processing Pipeline**
   - Implement async processing with Celery
   - Create job status tracking
   - Add progress reporting
   - Handle large file processing

3. **Data Management**
   - Design database schema
   - Implement file storage strategy
   - Create cleanup routines
   - Add caching layer

### Phase 4: Frontend Development (Week 4)
1. **PDF Viewer Component**
   - Integrate PDF.js for rendering
   - Implement page navigation
   - Add zoom and pan controls
   - Create thumbnail view

2. **Boundary Editor Interface**
   - Visual boundary markers
   - Drag-and-drop adjustment
   - Add/remove boundaries
   - Confidence indicators

3. **Document Management**
   - Upload interface
   - Processing status display
   - Document list view
   - Batch operations

### Phase 5: Integration & Testing (Week 5)
1. **End-to-End Testing**
   - Integration tests with sample PDFs
   - Performance testing
   - Memory usage validation
   - Cross-browser testing

2. **Optimization**
   - Memory usage optimization
   - Processing speed improvements
   - Frontend performance tuning
   - Error handling refinement

3. **Deployment Preparation**
   - Docker containerization
   - CI/CD pipeline setup
   - Documentation
   - Deployment scripts

## Boundary Detection Algorithm Details

### Primary Detection Methods

1. **Document Type Patterns**
   ```python
   EMAIL_PATTERNS = [
       r"From:\s*.*",
       r"To:\s*.*",
       r"Subject:\s*.*",
       r"Date:\s*.*"
   ]
   
   INVOICE_PATTERNS = [
       r"Invoice\s*#?\s*:?\s*\d+",
       r"Bill\s*To\s*:",
       r"Purchase\s*Order"
   ]
   ```

2. **Page Transition Signals**
   - Page numbering: "Page 1 of X" → new document
   - Header/footer changes
   - Significant white space (>50% of page)
   - Document end markers ("End of Document", signatures)

3. **Layout Analysis**
   - Column count changes
   - Font family/size distribution
   - Image/table density
   - Text alignment patterns

### Confidence Scoring
```python
class BoundaryConfidence:
    def calculate(self, signals: List[Signal]) -> float:
        weights = {
            SignalType.DOCUMENT_HEADER: 0.9,
            SignalType.PAGE_NUMBER_RESET: 0.8,
            SignalType.LAYOUT_CHANGE: 0.6,
            SignalType.WHITE_SPACE: 0.5
        }
        # Weighted average of signals
```

## Data Models

### Core Models
```python
class Document:
    id: str
    filename: str
    upload_date: datetime
    status: ProcessingStatus
    total_pages: int
    detected_boundaries: List[Boundary]
    
class Boundary:
    start_page: int
    end_page: int
    confidence: float
    document_type: Optional[DocumentType]
    metadata: Dict[str, Any]
    
class SplitResult:
    original_document_id: str
    split_documents: List[SplitDocument]
    
class SplitDocument:
    index: int
    filename: str
    start_page: int
    end_page: int
    document_type: Optional[str]
    download_url: str
```

## Memory Management Strategy

1. **Streaming Processing**
   - Process PDFs page by page
   - Don't load entire PDF into memory
   - Use temporary files for intermediate results

2. **Caching Strategy**
   - Cache extracted text/metadata in Redis
   - Store thumbnails separately
   - Implement LRU eviction

3. **Resource Limits**
   - Max file size: 500MB
   - Max pages: 1000
   - Concurrent processing: 2 documents
   - Memory limit per process: 4GB

## Error Handling

1. **Common Error Scenarios**
   - Corrupted PDFs
   - Password-protected files
   - Unsupported PDF versions
   - Memory exhaustion
   - Network timeouts

2. **Recovery Strategies**
   - Automatic retries with backoff
   - Partial processing recovery
   - Graceful degradation
   - User-friendly error messages

## Security Considerations

1. **File Handling**
   - Virus scanning on upload
   - File type validation
   - Sandboxed processing
   - Secure file storage

2. **API Security**
   - Rate limiting
   - API key authentication
   - CORS configuration
   - Input validation

## Performance Targets

- **Processing Speed**: 10 pages/second
- **Memory Usage**: <4GB for 100-page PDF
- **API Response Time**: <100ms for metadata
- **Accuracy**: >90% boundary detection on test set

## Future Enhancements (RAG Integration)

1. **Modular Extraction**
   - Text extraction module
   - Metadata extraction module
   - Document classification module
   - Vector embedding module

2. **API Extensions**
   - Webhook support for processing completion
   - Batch processing API
   - Document search API
   - Analytics endpoint

## Development Guidelines

1. **Code Organization**
   - Single responsibility principle
   - Dependency injection
   - Interface-based design
   - Comprehensive logging

2. **Testing Standards**
   - Minimum 80% code coverage
   - Integration tests for all APIs
   - Performance benchmarks
   - Memory leak detection

3. **Documentation**
   - API documentation (OpenAPI)
   - Code comments for complex logic
   - Architecture decision records
   - User guides

## Monitoring & Observability

1. **Metrics**
   - Processing time per document
   - Memory usage patterns
   - API request rates
   - Error rates by type

2. **Logging**
   - Structured logging (JSON)
   - Request tracing
   - Error stack traces
   - Performance profiling

## Deployment Strategy

1. **Local Development**
   - Docker Compose setup
   - Hot reloading
   - Local Redis/PostgreSQL

2. **Production Deployment**
   - Kubernetes manifests
   - Auto-scaling configuration
   - Health checks
   - Backup strategies

This implementation plan provides a comprehensive roadmap for building the Smart PDF Splitter with emphasis on modularity, performance, and future extensibility for RAG integration.