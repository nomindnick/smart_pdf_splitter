"""Configuration management for the Smart PDF Splitter."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # LLM Configuration
    ollama_url: str = Field(default="http://localhost:11434", env="OLLAMA_URL")
    llm_model: str = Field(default="phi4-mini:3.8b", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")
    llm_timeout: int = Field(default=30, env="LLM_TIMEOUT")
    enable_llm_detection: bool = Field(default=False, env="ENABLE_LLM_DETECTION")
    
    # Visual Processing Configuration
    enable_visual_detection: bool = Field(default=True, env="ENABLE_VISUAL_DETECTION")
    visual_confidence_threshold: float = Field(default=0.5, env="VISUAL_CONFIDENCE_THRESHOLD")
    visual_memory_limit_mb: int = Field(default=2048, env="VISUAL_MEMORY_LIMIT_MB")
    page_image_resolution: int = Field(default=150, env="PAGE_IMAGE_RESOLUTION")
    
    # Intelligent OCR Configuration
    enable_intelligent_ocr: bool = Field(default=True, env="ENABLE_INTELLIGENT_OCR")
    ocr_high_quality_threshold: float = Field(default=0.8, env="OCR_HIGH_QUALITY_THRESHOLD")
    ocr_medium_quality_threshold: float = Field(default=0.6, env="OCR_MEDIUM_QUALITY_THRESHOLD")
    
    # Processing Limits
    max_upload_size_mb: int = Field(default=500, env="MAX_UPLOAD_SIZE_MB")
    max_ocr_pages: int = Field(default=100, env="MAX_OCR_PAGES")
    processing_timeout_seconds: int = Field(default=600, env="PROCESSING_TIMEOUT_SECONDS")
    
    # Storage Configuration
    upload_dir: str = Field(default="/tmp/pdf_uploads", env="UPLOAD_DIR")
    temp_dir: str = Field(default="/tmp/pdf_processing", env="TEMP_DIR")
    
    # CORS Configuration
    cors_origins: str = Field(default="http://localhost:3000", env="CORS_ORIGINS")
    
    # Database Configuration (for future use)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    @property
    def max_upload_size_bytes(self) -> int:
        """Convert MB to bytes."""
        return self.max_upload_size_mb * 1024 * 1024


# Create settings instance
settings = Settings()