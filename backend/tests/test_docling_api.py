"""Test to explore actual Docling API structure."""

from docling.document_converter import DocumentConverter
from docling.datamodel.document import ConversionResult
from docling_core.types.doc.document import DoclingDocument
from pathlib import Path

def explore_docling_api():
    """Explore the structure of Docling objects."""
    # Create a converter
    converter = DocumentConverter()
    print(f"Converter type: {type(converter)}")
    
    # Create a mock result to explore structure
    # First, let's see what methods are available
    print("\nDocumentConverter methods:")
    for attr in dir(converter):
        if not attr.startswith('_'):
            print(f"  - {attr}")
    
    # Check ConversionResult structure
    print("\nConversionResult attributes:")
    for attr in dir(ConversionResult):
        if not attr.startswith('_'):
            print(f"  - {attr}")
    
    # Check DoclingDocument structure  
    print("\nDoclingDocument attributes:")
    for attr in dir(DoclingDocument):
        if not attr.startswith('_'):
            print(f"  - {attr}")
    
    print("\nAPI exploration complete!")

if __name__ == "__main__":
    explore_docling_api()