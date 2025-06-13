# Smart PDF Splitter

A modular PDF splitting application that automatically detects document boundaries within multi-document PDFs and provides a visual interface for verification and adjustment.

## Features

- 🔍 **Automatic Boundary Detection**: Uses content and visual analysis to detect where one document ends and another begins
- 🤖 **LLM-Powered Intelligence**: Optional Phi-4 Mini integration for enhanced boundary detection using semantic understanding
- 👀 **Visual Verification Interface**: Review and adjust detected boundaries before splitting
- 📄 **Multiple Document Types**: Supports emails, invoices, reports, contracts, and more
- 🚀 **High Performance**: Processes up to 10 pages per second with memory-efficient streaming
- 🐳 **Docker Support**: Easy deployment with Docker Compose
- 🔧 **Modular Design**: Built for future integration into RAG pipelines

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd smart_pdf_splitter

# Start all services
docker-compose up --build

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Local Development

#### Backend Setup

```bash
# Create virtual environment
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Copy environment variables
cp .env.example .env

# Run tests
pytest

# Start the backend server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup

```bash
# Install dependencies
cd frontend
npm install

# Start development server
npm run dev

# Run tests
npm test
```

## Architecture

The application consists of:

- **Backend**: FastAPI-based REST API with Python 3.11+
- **Document Processing**: Powered by IBM's Docling library
- **Frontend**: React + TypeScript with PDF.js for visualization
- **Queue**: Celery + Redis for asynchronous processing
- **Database**: PostgreSQL for document metadata

## API Documentation

Once the backend is running, visit http://localhost:8000/docs for the interactive API documentation.

### Key Endpoints

- `POST /api/documents/upload` - Upload a PDF for processing
- `GET /api/documents/{id}/boundaries` - Get detected boundaries
- `PUT /api/documents/{id}/boundaries` - Update boundaries manually
- `POST /api/documents/{id}/split` - Execute split operation
- `GET /api/documents/{id}/download/{idx}` - Download split PDF

## Testing

The project includes comprehensive test coverage:

```bash
# Backend tests
cd backend
pytest -v

# Frontend tests
cd frontend
npm test

# Integration tests
pytest tests/ -v -m integration
```

## Configuration

Key environment variables:

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `MAX_UPLOAD_SIZE`: Maximum file size (default: 500MB)
- `MAX_PAGES_PER_DOCUMENT`: Maximum pages per PDF (default: 1000)
- `ENABLE_LLM_DETECTION`: Enable LLM-powered boundary detection (default: false)
- `LLM_MODEL`: Ollama model to use (default: phi4-mini:3.8b)

See `backend/.env.example` for all configuration options.

### LLM-Powered Detection (Optional)

For enhanced boundary detection using LLM:

1. Install [Ollama](https://ollama.ai)
2. Pull the Phi-4 Mini model: `ollama pull phi4-mini:3.8b`
3. Set `ENABLE_LLM_DETECTION=true` in your `.env` file

See [docs/llm_setup.md](docs/llm_setup.md) for detailed setup instructions.

## Memory Requirements

- Minimum: 8GB RAM
- Recommended: 16GB RAM
- Target memory usage: <4GB per document

## Contributing

1. Follow test-driven development practices
2. Maintain >80% code coverage
3. Use type hints in Python code
4. Follow the established code style

## License

[Your License Here]

## Acknowledgments

- Built with [Docling](https://github.com/DS4SD/docling) for document understanding
- Uses [PDF.js](https://mozilla.github.io/pdf.js/) for PDF rendering