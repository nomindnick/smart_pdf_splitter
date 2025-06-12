"""
Enhanced API routes integrating the advanced boundary detection system.

This module shows how to integrate the plugin-based boundary detection
into the FastAPI application.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field

from ..core.boundary_detector_enhanced import (
    EnhancedBoundaryDetector,
    DetectorConfig
)
from ..core.detector_config_examples import (
    BoundaryDetectionPresets,
    AdaptiveBoundaryDetector
)
from ..core.models import (
    Document,
    ProcessingStatus,
    Boundary,
    ProcessingStatusResponse
)
from ..core.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["enhanced"])


class DetectorConfigRequest(BaseModel):
    """Request model for custom detector configuration."""
    
    detectors: List[Dict[str, Any]] = Field(
        description="List of detector configurations"
    )
    min_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )
    min_signals: int = Field(
        default=1,
        ge=1,
        description="Minimum number of signals required"
    )


class ProcessingOptionsRequest(BaseModel):
    """Advanced processing options."""
    
    preset: Optional[str] = Field(
        default=None,
        description="Preset configuration name"
    )
    enable_llm: bool = Field(
        default=True,
        description="Enable LLM-based detection"
    )
    llm_model: Optional[str] = Field(
        default="llama3.2",
        description="LLM model to use"
    )
    enable_vlm: bool = Field(
        default=False,
        description="Enable VLM visual analysis"
    )
    enable_construction: bool = Field(
        default=True,
        description="Enable construction-specific detection"
    )
    adaptive_mode: bool = Field(
        default=False,
        description="Use adaptive configuration selection"
    )
    context_window: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Context window for analysis"
    )


class BoundaryAnalysisResponse(BaseModel):
    """Detailed boundary analysis response."""
    
    document_id: str
    boundaries: List[Boundary]
    detection_methods_used: List[str]
    processing_time: float
    confidence_distribution: Dict[str, int]
    document_types_found: Dict[str, int]
    

# Dependency to get detector configuration
async def get_detector_config(
    options: ProcessingOptionsRequest = Body(...)
) -> List[DetectorConfig]:
    """Get detector configuration based on processing options."""
    
    # Use preset if specified
    if options.preset:
        presets = BoundaryDetectionPresets()
        preset_map = {
            "construction": presets.get_construction_focused_config,
            "general": presets.get_general_purpose_config,
            "high_accuracy": presets.get_high_accuracy_config,
            "fast": presets.get_fast_config,
        }
        
        if options.preset in preset_map:
            return preset_map[options.preset]()
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown preset: {options.preset}"
            )
    
    # Build custom configuration
    configs = []
    
    # Always include rule-based
    configs.append(DetectorConfig(
        name="rule_based",
        enabled=True,
        weight=1.0,
        config={'min_confidence': 0.6}
    ))
    
    # Add LLM if enabled
    if options.enable_llm:
        configs.append(DetectorConfig(
            name="llm",
            enabled=True,
            weight=0.8,
            config={
                'ollama_url': 'http://localhost:11434',
                'model_name': options.llm_model,
                'temperature': 0.1
            }
        ))
    
    # Add VLM if enabled
    if options.enable_vlm:
        configs.append(DetectorConfig(
            name="vlm",
            enabled=True,
            weight=0.7,
            config={'vlm_model': 'dit-base'}
        ))
    
    # Add construction-specific if enabled
    if options.enable_construction:
        configs.append(DetectorConfig(
            name="construction",
            enabled=True,
            weight=1.2,
            config={}
        ))
    
    return configs


@router.post("/documents/{document_id}/analyze", response_model=BoundaryAnalysisResponse)
async def analyze_document_enhanced(
    document_id: str,
    options: ProcessingOptionsRequest = Body(default=ProcessingOptionsRequest()),
    detector_configs: Optional[List[DetectorConfig]] = Depends(get_detector_config)
):
    """
    Analyze document with enhanced boundary detection.
    
    This endpoint provides advanced boundary detection with configurable
    detection methods and detailed analysis results.
    """
    try:
        # Get document (would be from database in real implementation)
        document = await get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if we should use adaptive mode
        if options.adaptive_mode:
            adaptive = AdaptiveBoundaryDetector()
            boundaries = await adaptive.detect_with_adaptation(
                Path(document.original_path)
            )
            detection_methods = ["adaptive"]
        else:
            # Create enhanced detector
            detector = EnhancedBoundaryDetector(
                min_confidence=0.6,
                min_signals=1,
                detector_configs=detector_configs
            )
            
            # Process document
            processor = DocumentProcessor()
            pages = list(processor.process_document(Path(document.original_path)))
            
            # Detect boundaries
            import time
            start_time = time.time()
            boundaries = await detector.detect_boundaries(
                pages,
                context_window=options.context_window
            )
            processing_time = time.time() - start_time
            
            # Get detection methods used
            detection_methods = [d.name for d in detector.detectors if d.enabled]
        
        # Analyze results
        confidence_distribution = analyze_confidence_distribution(boundaries)
        document_types = analyze_document_types(boundaries)
        
        return BoundaryAnalysisResponse(
            document_id=document_id,
            boundaries=boundaries,
            detection_methods_used=detection_methods,
            processing_time=processing_time,
            confidence_distribution=confidence_distribution,
            document_types_found=document_types
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/{document_id}/detect-custom")
async def detect_with_custom_config(
    document_id: str,
    config_request: DetectorConfigRequest
):
    """
    Detect boundaries using custom detector configuration.
    
    Allows fine-grained control over detection parameters.
    """
    try:
        # Parse detector configs
        detector_configs = []
        for cfg in config_request.detectors:
            detector_configs.append(DetectorConfig(
                name=cfg['name'],
                enabled=cfg.get('enabled', True),
                weight=cfg.get('weight', 1.0),
                config=cfg.get('config', {})
            ))
        
        # Create detector
        detector = EnhancedBoundaryDetector(
            min_confidence=config_request.min_confidence,
            min_signals=config_request.min_signals,
            detector_configs=detector_configs
        )
        
        # Get document
        document = await get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Process
        processor = DocumentProcessor()
        pages = list(processor.process_document(Path(document.original_path)))
        
        # Detect
        boundaries = await detector.detect_boundaries(pages)
        
        return {
            "document_id": document_id,
            "boundaries": boundaries,
            "total_detected": len(boundaries)
        }
        
    except Exception as e:
        logger.error(f"Error in custom detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/compare-methods")
async def compare_detection_methods(
    document_id: str,
    include_vlm: bool = Query(default=False, description="Include VLM analysis")
):
    """
    Compare results from different detection methods.
    
    Useful for testing and optimization.
    """
    try:
        document = await get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Process document once
        processor = DocumentProcessor()
        pages = list(processor.process_document(Path(document.original_path)))
        
        results = {}
        
        # Test each preset
        presets = {
            "rule_based_only": BoundaryDetectionPresets.get_fast_config(),
            "general_purpose": BoundaryDetectionPresets.get_general_purpose_config(),
            "construction_focused": BoundaryDetectionPresets.get_construction_focused_config(),
        }
        
        if include_vlm:
            presets["high_accuracy"] = BoundaryDetectionPresets.get_high_accuracy_config()
        
        for name, config in presets.items():
            detector = EnhancedBoundaryDetector(
                min_confidence=0.6,
                min_signals=1,
                detector_configs=config
            )
            
            boundaries = await detector.detect_boundaries(pages)
            
            results[name] = {
                "boundary_count": len(boundaries),
                "boundaries": [
                    {
                        "pages": b.page_range,
                        "confidence": b.confidence,
                        "type": b.document_type.value if b.document_type else None
                    }
                    for b in boundaries
                ],
                "avg_confidence": sum(b.confidence for b in boundaries) / len(boundaries) if boundaries else 0
            }
        
        return {
            "document_id": document_id,
            "total_pages": len(pages),
            "comparison_results": results
        }
        
    except Exception as e:
        logger.error(f"Error in method comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/detection-presets")
async def list_detection_presets():
    """List available detection presets with descriptions."""
    return {
        "presets": [
            {
                "name": "construction",
                "description": "Optimized for construction documents (RFIs, submittals, etc.)",
                "detectors": ["construction", "rule_based", "llm"],
                "use_case": "Construction industry documents"
            },
            {
                "name": "general",
                "description": "Balanced configuration for general documents",
                "detectors": ["rule_based", "llm", "construction"],
                "use_case": "Mixed document types"
            },
            {
                "name": "high_accuracy",
                "description": "Maximum accuracy using all detection methods",
                "detectors": ["rule_based", "llm", "vlm", "construction"],
                "use_case": "Critical documents requiring highest accuracy"
            },
            {
                "name": "fast",
                "description": "Fast processing using only pattern matching",
                "detectors": ["rule_based", "construction"],
                "use_case": "Large documents or time-sensitive processing"
            }
        ]
    }


@router.post("/batch-analyze")
async def batch_analyze_documents(
    document_ids: List[str] = Body(..., description="List of document IDs to analyze"),
    preset: str = Query(default="general", description="Detection preset to use")
):
    """
    Analyze multiple documents in batch using the same configuration.
    
    Efficient for processing document sets with similar characteristics.
    """
    try:
        # Get preset configuration
        preset_configs = {
            "construction": BoundaryDetectionPresets.get_construction_focused_config(),
            "general": BoundaryDetectionPresets.get_general_purpose_config(),
            "high_accuracy": BoundaryDetectionPresets.get_high_accuracy_config(),
            "fast": BoundaryDetectionPresets.get_fast_config(),
        }
        
        if preset not in preset_configs:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown preset: {preset}"
            )
        
        # Create detector
        detector = EnhancedBoundaryDetector(
            min_confidence=0.6,
            min_signals=1,
            detector_configs=preset_configs[preset]
        )
        
        # Process each document
        results = {}
        processor = DocumentProcessor()
        
        for doc_id in document_ids:
            try:
                document = await get_document(doc_id)
                if not document:
                    results[doc_id] = {"error": "Document not found"}
                    continue
                
                # Process
                pages = list(processor.process_document(Path(document.original_path)))
                boundaries = await detector.detect_boundaries(pages)
                
                results[doc_id] = {
                    "success": True,
                    "boundary_count": len(boundaries),
                    "boundaries": boundaries
                }
                
            except Exception as e:
                results[doc_id] = {"error": str(e)}
        
        return {
            "preset_used": preset,
            "documents_processed": len(document_ids),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
async def get_document(document_id: str) -> Optional[Document]:
    """Get document by ID (placeholder for database lookup)."""
    # In real implementation, this would fetch from database
    # For now, return a mock document
    return Document(
        id=document_id,
        filename=f"document_{document_id}.pdf",
        total_pages=10,
        file_size=1024000,
        status=ProcessingStatus.COMPLETED,
        original_path=f"/tmp/documents/{document_id}.pdf"
    )


def analyze_confidence_distribution(boundaries: List[Boundary]) -> Dict[str, int]:
    """Analyze confidence score distribution."""
    distribution = {
        "high": 0,      # >= 0.8
        "medium": 0,    # 0.6 - 0.8
        "low": 0        # < 0.6
    }
    
    for boundary in boundaries:
        if boundary.confidence >= 0.8:
            distribution["high"] += 1
        elif boundary.confidence >= 0.6:
            distribution["medium"] += 1
        else:
            distribution["low"] += 1
            
    return distribution


def analyze_document_types(boundaries: List[Boundary]) -> Dict[str, int]:
    """Count document types found."""
    types = {}
    
    for boundary in boundaries:
        if boundary.document_type:
            type_name = boundary.document_type.value
            types[type_name] = types.get(type_name, 0) + 1
            
    return types