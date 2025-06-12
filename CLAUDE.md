# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the Smart PDF Splitter project.

## Project Context

This is a modular PDF splitting application that:
1. Takes multi-document PDFs as input
2. Automatically detects document boundaries using content and visual analysis
3. Provides a visual interface for users to verify/adjust boundaries
4. Exports individual PDFs for each detected document
5. Is designed to be integrated into a larger RAG (Retrieval Augmented Generation) pipeline

## Key Technical Decisions

- **Backend**: FastAPI with Python 3.11+
- **Document Processing**: Docling (IBM's document parsing library)
- **PDF Manipulation**: PyMuPDF (fitz)
- **Frontend**: React + TypeScript with PDF.js
- **Testing**: Test-driven development using pytest and Jest
- **Memory Constraint**: Must run on systems with 32GB RAM, target <16GB usage

## Development Commands

### Backend Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
pip install -r backend/requirements-dev.txt

# Run tests
pytest backend/tests/ -v

# Run specific test
pytest backend/tests/test_boundary_detector.py::test_email_detection -v

# Run with coverage
pytest backend/tests/ --cov=backend/src --cov-report=html

# Run backend server
cd backend
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Run linting
ruff check backend/
black backend/ --check
mypy backend/src/

# Format code
black backend/
ruff format backend/
```

### Frontend Development
```bash
# Install dependencies
cd frontend
npm install

# Run development server
npm run dev

# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# Build for production
npm run build

# Run linting
npm run lint

# Format code
npm run format
```

### Docker Commands
```bash
# Build and run all services
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop all services
docker-compose down

# Clean up volumes
docker-compose down -v
```

## Project Structure

```
smart_pdf_splitter/
├── backend/
│   ├── src/
│   │   ├── core/              # Core business logic
│   │   │   ├── document_processor.py    # Docling integration
│   │   │   ├── boundary_detector.py     # Boundary detection algorithms
│   │   │   ├── pdf_splitter.py         # PDF splitting operations
│   │   │   └── models.py               # Pydantic data models
│   │   ├── api/               # FastAPI application
│   │   │   ├── main.py
│   │   │   ├── routes/
│   │   │   └── dependencies.py
│   │   └── utils/             # Utilities
│   ├── tests/                 # Backend tests
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── services/          # API clients
│   │   └── types/            # TypeScript types
│   └── package.json
├── tests/                     # Integration tests & test data
│   ├── test_files/
│   │   ├── Test_PDF_Set_1.pdf
│   │   └── Test_PDF_Set_Ground_Truth.json
└── docker-compose.yml
```

## Testing Strategy

### Test-Driven Development
1. Always check existing tests before implementing features
2. Write tests for new functionality before implementation
3. Use the ground truth JSON to validate boundary detection accuracy
4. Test memory usage stays under 16GB for large PDFs

### Key Test Files
- `Test_PDF_Set_1.pdf`: Contains 14 different documents (emails, invoices, etc.)
- `Test_PDF_Set_Ground_Truth.json`: Expected boundaries and document types
- Test should verify we correctly identify all 14 documents with their page ranges

## Boundary Detection Logic

The boundary detector uses multiple signals:

1. **Content Patterns**
   - Email headers (From:, To:, Subject:)
   - Document headers (Invoice #, Purchase Order, etc.)
   - Page numbers resetting to 1
   - Document type transitions

2. **Visual Signals**
   - Significant white space between documents
   - Layout changes (columns, fonts, alignment)
   - Header/footer changes

3. **Confidence Scoring**
   - Each signal has a weight
   - Combined score determines boundary confidence
   - User can see confidence in UI

## Memory Management

- Stream PDFs page-by-page, don't load entirely into memory
- Use temporary files for intermediate processing
- Clear caches regularly
- Monitor memory usage in tests

## API Endpoints

```
POST   /api/documents/upload              # Upload PDF for processing
GET    /api/documents/{id}/status         # Check processing status
GET    /api/documents/{id}/boundaries     # Get detected boundaries
PUT    /api/documents/{id}/boundaries     # Update boundaries manually
POST   /api/documents/{id}/split          # Execute split with current boundaries
GET    /api/documents/{id}/download/{idx} # Download individual split PDF
DELETE /api/documents/{id}                # Clean up document and files
```

## Common Patterns

### Adding a New Boundary Detection Method
1. Add method to `BoundaryDetector` class
2. Add signal type to `SignalType` enum
3. Update confidence calculation weights
4. Add tests with specific test PDFs
5. Update documentation

### Adding a New Document Type
1. Add to `DocumentType` enum
2. Add detection patterns
3. Update ground truth JSON for tests
4. Add UI support for new type

## Performance Considerations

- Target: Process 10 pages/second
- Memory limit: 4GB per document
- Use async/await for I/O operations
- Cache extracted text/metadata
- Profile memory usage for large PDFs

## Future RAG Integration

Keep these modules separate and reusable:
- Document parsing (via Docling)
- Text extraction
- Metadata extraction
- Boundary detection
- PDF manipulation

These will be extracted and used in the larger RAG pipeline.

## Debugging Tips

1. **Boundary Detection Issues**
   - Enable debug logging in `boundary_detector.py`
   - Check signal scores for each page
   - Visualize detected boundaries in test output

2. **Memory Issues**
   - Use memory profiler: `mprof run python script.py`
   - Check for PDF streams not being closed
   - Monitor Redis memory usage

3. **Docling Issues**
   - Check Docling version compatibility
   - Enable Docling debug mode
   - Verify PDF is not corrupted

## Code Style

- Use type hints everywhere
- Follow PEP 8 for Python
- Use ESLint/Prettier settings for TypeScript
- Meaningful variable names
- Document complex algorithms
- Keep functions small and focused

## Security Notes

- Validate all file uploads
- Sanitize filenames
- Use secure random IDs
- Clean up temporary files
- Rate limit API endpoints
- Never trust user input for page ranges