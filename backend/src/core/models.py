"""Core data models for the Smart PDF Splitter."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class ProcessingStatus(str, Enum):
    """Status of document processing."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Types of documents that can be detected."""
    
    EMAIL = "email"
    EMAIL_CHAIN = "email_chain"
    INVOICE = "invoice"
    RECEIPT = "receipt"
    REPORT = "report"
    FORM = "form"
    CONTRACT = "contract"
    LETTER = "letter"
    MEMO = "memo"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"
    OTHER = "other"


class SignalType(str, Enum):
    """Types of signals used for boundary detection."""
    
    DOCUMENT_HEADER = "document_header"
    PAGE_NUMBER_RESET = "page_number_reset"
    LAYOUT_CHANGE = "layout_change"
    WHITE_SPACE = "white_space"
    EMAIL_HEADER = "email_header"
    DOCUMENT_TYPE_CHANGE = "document_type_change"
    VISUAL_SEPARATOR = "visual_separator"
    TEXT_PATTERN = "text_pattern"


class BoundingBox(BaseModel):
    """Represents a bounding box on a page."""
    
    x: float = Field(..., description="X coordinate (left)")
    y: float = Field(..., description="Y coordinate (top)")
    width: float = Field(..., description="Width of the box")
    height: float = Field(..., description="Height of the box")
    
    @validator("width", "height")
    def positive_dimensions(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Dimensions must be positive")
        return v


class Signal(BaseModel):
    """Represents a boundary detection signal."""
    
    type: SignalType
    confidence: float = Field(..., ge=0.0, le=1.0)
    page_number: int = Field(..., ge=1)
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Boundary(BaseModel):
    """Represents a document boundary."""
    
    start_page: int = Field(..., ge=1)
    end_page: int = Field(..., ge=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    document_type: Optional[DocumentType] = None
    signals: List[Signal] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator("end_page")
    def end_after_start(cls, v: int, values: dict) -> int:
        if "start_page" in values and v < values["start_page"]:
            raise ValueError("End page must be >= start page")
        return v
    
    @property
    def page_count(self) -> int:
        """Number of pages in this document."""
        return self.end_page - self.start_page + 1
    
    @property
    def page_range(self) -> str:
        """String representation of page range."""
        if self.start_page == self.end_page:
            return str(self.start_page)
        return f"{self.start_page}-{self.end_page}"


class DocumentMetadata(BaseModel):
    """Metadata extracted from a document."""
    
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    page_count: int = Field(..., ge=1)
    file_size: int = Field(..., ge=0)
    mime_type: str = "application/pdf"
    language: Optional[str] = None
    
    # Document-specific metadata
    email_from: Optional[str] = None
    email_to: Optional[List[str]] = None
    email_subject: Optional[str] = None
    email_date: Optional[datetime] = None
    
    invoice_number: Optional[str] = None
    invoice_date: Optional[datetime] = None
    invoice_amount: Optional[float] = None
    
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class PageInfo(BaseModel):
    """Information about a single page."""
    
    page_number: int = Field(..., ge=1)
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)
    text_content: Optional[str] = None
    layout_elements: List[Dict[str, Any]] = Field(default_factory=list)
    has_images: bool = False
    has_tables: bool = False
    word_count: int = Field(default=0, ge=0)
    
    @property
    def is_mostly_empty(self) -> bool:
        """Check if page is mostly empty (less than 50 words)."""
        return self.word_count < 50


class Document(BaseModel):
    """Represents a document being processed."""
    
    id: str = Field(..., description="Unique document ID")
    filename: str
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    status: ProcessingStatus = ProcessingStatus.PENDING
    total_pages: int = Field(..., ge=1)
    file_size: int = Field(..., ge=0)
    
    # Processing results
    detected_boundaries: List[Boundary] = Field(default_factory=list)
    page_info: List[PageInfo] = Field(default_factory=list)
    metadata: Optional[DocumentMetadata] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    
    # File paths
    original_path: Optional[str] = None
    processed_path: Optional[str] = None


class SplitDocument(BaseModel):
    """Represents a single document after splitting."""
    
    index: int = Field(..., ge=0)
    filename: str
    start_page: int = Field(..., ge=1)
    end_page: int = Field(..., ge=1)
    document_type: Optional[DocumentType] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    
    @property
    def page_count(self) -> int:
        """Number of pages in this split document."""
        return self.end_page - self.start_page + 1


class SplitResult(BaseModel):
    """Result of splitting a document."""
    
    original_document_id: str
    created_date: datetime = Field(default_factory=datetime.utcnow)
    split_documents: List[SplitDocument] = Field(default_factory=list)
    total_documents: int = Field(default=0, ge=0)
    
    @validator("total_documents", always=True)
    def set_total_documents(cls, v: int, values: dict) -> int:
        if "split_documents" in values:
            return len(values["split_documents"])
        return v


# Request/Response Models for API

class DocumentUploadResponse(BaseModel):
    """Response for document upload."""
    
    document_id: str
    filename: str
    status: ProcessingStatus
    message: str = "Document uploaded successfully"


class BoundaryUpdateRequest(BaseModel):
    """Request to update document boundaries."""
    
    boundaries: List[Boundary]
    
    @validator("boundaries")
    def non_empty_boundaries(cls, v: List[Boundary]) -> List[Boundary]:
        if not v:
            raise ValueError("At least one boundary must be provided")
        return v


class ProcessingStatusResponse(BaseModel):
    """Response for processing status check."""
    
    document_id: str
    status: ProcessingStatus
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
    message: Optional[str] = None
    detected_boundaries: Optional[List[Boundary]] = None
    processing_time: Optional[float] = None


class SplitRequest(BaseModel):
    """Request to split a document."""
    
    confirm_boundaries: bool = Field(
        default=True,
        description="Whether to use the current boundaries without modification"
    )
    output_format: str = Field(
        default="pdf",
        description="Output format for split documents"
    )
    naming_pattern: str = Field(
        default="{original_name}_part_{index:03d}",
        description="Naming pattern for split documents"
    )