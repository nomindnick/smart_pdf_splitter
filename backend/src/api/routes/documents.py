"""Document processing API routes."""

import os
import uuid
from typing import Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile, BackgroundTasks, Query
from fastapi.responses import FileResponse

from ...core.config import settings
from ...core.models import (
    Boundary,
    BoundaryUpdateRequest,
    Document,
    DocumentUploadResponse,
    ProcessingStatus,
    ProcessingStatusResponse,
    SplitRequest,
    SplitResult,
)

router = APIRouter()

# Temporary in-memory storage (replace with database)
documents_store: Dict[str, Document] = {}


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    enable_visual_detection: bool = Query(default=None, description="Enable visual boundary detection"),
    enable_llm_detection: bool = Query(default=None, description="Enable LLM-based boundary detection"),
    enable_intelligent_ocr: bool = Query(default=None, description="Enable intelligent OCR strategy"),
) -> DocumentUploadResponse:
    """Upload a PDF document for processing."""
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Check file size
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=400, 
            detail=f"File size exceeds {settings.max_upload_size_mb}MB limit"
        )
    
    # Generate unique document ID
    document_id = str(uuid.uuid4())
    
    # Save file temporarily
    upload_dir = settings.upload_dir
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{document_id}.pdf")
    
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Create document record
    document = Document(
        id=document_id,
        filename=file.filename,
        status=ProcessingStatus.PENDING,
        total_pages=1,  # Will be updated during processing
        file_size=file_size,
        original_path=file_path,
    )
    
    documents_store[document_id] = document
    
    # Use settings defaults if not specified
    visual_detection = enable_visual_detection if enable_visual_detection is not None else settings.enable_visual_detection
    llm_detection = enable_llm_detection if enable_llm_detection is not None else settings.enable_llm_detection
    intelligent_ocr = enable_intelligent_ocr if enable_intelligent_ocr is not None else settings.enable_intelligent_ocr
    
    # Queue processing task with detection options
    background_tasks.add_task(
        process_document, 
        document_id,
        visual_detection,
        llm_detection,
        intelligent_ocr
    )
    
    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        status=ProcessingStatus.PENDING,
    )


@router.get("/{document_id}/status", response_model=ProcessingStatusResponse)
async def get_processing_status(document_id: str) -> ProcessingStatusResponse:
    """Get the processing status of a document."""
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = documents_store[document_id]
    
    return ProcessingStatusResponse(
        document_id=document_id,
        status=document.status,
        detected_boundaries=document.detected_boundaries if document.status == ProcessingStatus.COMPLETED else None,
        processing_time=document.processing_time,
        message=document.error_message if document.status == ProcessingStatus.FAILED else None,
    )


@router.get("/{document_id}/boundaries", response_model=List[Boundary])
async def get_boundaries(document_id: str) -> List[Boundary]:
    """Get detected document boundaries."""
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = documents_store[document_id]
    
    if document.status != ProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Document processing not completed. Current status: {document.status}",
        )
    
    return document.detected_boundaries


@router.put("/{document_id}/boundaries")
async def update_boundaries(
    document_id: str,
    request: BoundaryUpdateRequest,
) -> dict:
    """Update document boundaries manually."""
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = documents_store[document_id]
    
    if document.status != ProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Document processing not completed. Current status: {document.status}",
        )
    
    # Validate boundaries
    for boundary in request.boundaries:
        if boundary.end_page > document.total_pages:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid boundary: end_page {boundary.end_page} exceeds total pages {document.total_pages}",
            )
    
    # Update boundaries
    document.detected_boundaries = request.boundaries
    
    return {"message": "Boundaries updated successfully", "boundary_count": len(request.boundaries)}


@router.post("/{document_id}/split", response_model=SplitResult)
async def split_document(
    document_id: str,
    request: SplitRequest = SplitRequest(),
) -> SplitResult:
    """Split the document based on current boundaries."""
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = documents_store[document_id]
    
    if document.status != ProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Document processing not completed. Current status: {document.status}",
        )
    
    if not document.detected_boundaries:
        raise HTTPException(
            status_code=400,
            detail="No boundaries detected. Cannot split document.",
        )
    
    # TODO: Implement actual PDF splitting logic
    # For now, return mock result
    split_result = SplitResult(
        original_document_id=document_id,
        split_documents=[],  # Will be populated by splitting logic
    )
    
    return split_result


@router.get("/{document_id}/download/{index}")
async def download_split_document(document_id: str, index: int) -> FileResponse:
    """Download a specific split document."""
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # TODO: Implement actual file serving
    # For now, return 404
    raise HTTPException(status_code=404, detail="Split document not found")


@router.get("/{document_id}/quality")
async def get_document_quality(document_id: str) -> dict:
    """Get OCR quality summary for a document."""
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = documents_store[document_id]
    
    if document.status != ProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Document processing not completed. Current status: {document.status}",
        )
    
    # Create a service instance to access the method
    service = DocumentService()
    return service.get_document_quality_summary(document)


@router.get("/{document_id}/pages/{page_number}/ocr")
async def get_page_ocr_details(document_id: str, page_number: int) -> dict:
    """Get detailed OCR information for a specific page."""
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = documents_store[document_id]
    
    if document.status != ProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Document processing not completed. Current status: {document.status}",
        )
    
    # Create a service instance to access the method
    service = DocumentService()
    details = service.get_page_ocr_details(document, page_number)
    
    if "error" in details:
        raise HTTPException(status_code=400, detail=details["error"])
    
    return details


@router.delete("/{document_id}")
async def delete_document(document_id: str) -> dict:
    """Delete a document and all associated files."""
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = documents_store[document_id]
    
    # Clean up files
    if document.original_path and os.path.exists(document.original_path):
        os.remove(document.original_path)
    
    # Remove from store
    del documents_store[document_id]
    
    return {"message": "Document deleted successfully"}


# Import document service
from ..services.document_service import DocumentService

# Document service will be initialized per request with specific options
document_service = None


# Background task for document processing
async def process_document(
    document_id: str,
    enable_visual_detection: bool = True,
    enable_llm_detection: bool = False,
    enable_intelligent_ocr: bool = True
):
    """Process document with enhanced OCR and configurable detection."""
    if document_id not in documents_store:
        return
    
    document = documents_store[document_id]
    
    # Initialize document service with specific options
    service = DocumentService(
        enable_adaptive_ocr=True,
        enable_visual_detection=enable_visual_detection,
        enable_llm_detection=enable_llm_detection,
        enable_intelligent_ocr=enable_intelligent_ocr
    )
    
    # Process document
    processed_document = await service.process_document(document)
    
    # Update store
    documents_store[document_id] = processed_document