"""Setup configuration for Smart PDF Splitter backend."""

from setuptools import setup, find_packages

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="smart-pdf-splitter",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A modular PDF splitting application with automatic document boundary detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/smart-pdf-splitter",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.1",
        "python-multipart>=0.0.9",
        "docling>=2.5.2",
        "PyMuPDF>=1.24.2",
        "sqlalchemy>=2.0.29",
        "alembic>=1.13.1",
        "celery>=5.3.6",
        "redis>=5.0.3",
        "pydantic>=2.6.4",
        "pydantic-settings>=2.2.1",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "python-dotenv>=1.0.1",
        "httpx>=0.27.0",
        "aiofiles>=23.2.1",
        "Pillow>=10.3.0",
        "opencv-python-headless>=4.9.0.80",
        "pypdf>=4.1.0",
        "structlog>=24.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.1.1",
            "pytest-asyncio>=0.23.6",
            "pytest-cov>=5.0.0",
            "pytest-mock>=3.12.0",
            "black>=24.3.0",
            "ruff>=0.3.4",
            "mypy>=1.9.0",
        ],
        "ocr": [
            "pytesseract>=0.3.10",
        ],
    },
    entry_points={
        "console_scripts": [
            "smart-pdf-splitter=src.cli:main",
        ],
    },
)