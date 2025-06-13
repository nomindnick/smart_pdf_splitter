#!/usr/bin/env python3
"""Test script to verify all key packages can be imported without conflicts."""

import sys

def test_imports():
    """Test importing all key packages."""
    packages = [
        ("fastapi", "FastAPI"),
        ("pydantic", "BaseModel"),
        ("pydantic_settings", "BaseSettings"),
        ("docling.document_converter", "DocumentConverter"),
        ("uvicorn", None),
        ("sqlalchemy", None),
        ("celery", None),
        ("redis", None),
    ]
    
    failed = []
    
    for package, obj in packages:
        try:
            if obj:
                exec(f"from {package} import {obj}")
            else:
                exec(f"import {package}")
            print(f"✓ Successfully imported {package}")
        except ImportError as e:
            failed.append((package, str(e)))
            print(f"✗ Failed to import {package}: {e}")
    
    # Test pydantic version
    try:
        import pydantic
        print(f"\nPydantic version: {pydantic.__version__}")
    except:
        pass
    
    if failed:
        print(f"\n{len(failed)} packages failed to import")
        sys.exit(1)
    else:
        print("\nAll packages imported successfully!")
        sys.exit(0)

if __name__ == "__main__":
    test_imports()