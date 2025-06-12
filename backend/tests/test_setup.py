"""Test basic setup and imports."""

import pytest


def test_imports():
    """Test that all core modules can be imported."""
    from src.core import models
    from src.api import main
    from src.api.routes import documents, health
    
    assert models.ProcessingStatus.PENDING == "pending"
    assert hasattr(main, "app")


def test_models():
    """Test basic model creation."""
    from src.core.models import Document, ProcessingStatus, Boundary
    
    # Test Document creation
    doc = Document(
        id="test-123",
        filename="test.pdf",
        total_pages=10,
        file_size=1024,
    )
    assert doc.id == "test-123"
    assert doc.status == ProcessingStatus.PENDING
    
    # Test Boundary creation
    boundary = Boundary(
        start_page=1,
        end_page=5,
        confidence=0.85,
    )
    assert boundary.page_count == 5
    assert boundary.page_range == "1-5"


def test_boundary_validation():
    """Test boundary validation."""
    from src.core.models import Boundary
    
    # Test valid boundary
    boundary = Boundary(start_page=1, end_page=1, confidence=0.9)
    assert boundary.page_range == "1"
    
    # Test invalid boundary (end < start)
    with pytest.raises(ValueError):
        Boundary(start_page=5, end_page=3, confidence=0.9)