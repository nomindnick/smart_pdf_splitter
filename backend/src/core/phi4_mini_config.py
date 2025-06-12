"""
Configuration module for phi4-mini boundary detection.

Provides easy configuration options for different use cases and
deployment scenarios.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .boundary_detector_enhanced import DetectorConfig


class DeploymentMode(str, Enum):
    """Deployment modes with different resource/accuracy tradeoffs."""
    
    DEVELOPMENT = "development"      # Full features, debugging enabled
    PRODUCTION = "production"        # Balanced performance/accuracy
    HIGH_PERFORMANCE = "high_perf"   # Optimized for speed
    HIGH_ACCURACY = "high_accuracy"  # Optimized for accuracy
    LOW_RESOURCE = "low_resource"    # Minimal resource usage


class DocumentFocus(str, Enum):
    """Document type focus for optimization."""
    
    GENERAL = "general"
    CONSTRUCTION = "construction"
    FINANCIAL = "financial"
    EMAIL = "email"
    MIXED = "mixed"


@dataclass
class Phi4MiniDeploymentConfig:
    """Complete deployment configuration for phi4-mini."""
    
    # Deployment settings
    mode: DeploymentMode = DeploymentMode.PRODUCTION
    document_focus: DocumentFocus = DocumentFocus.GENERAL
    
    # Model settings
    model_name: str = "phi4-mini:3.8b"
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    
    # Performance settings
    enable_caching: bool = True
    enable_batching: bool = True
    max_concurrent_requests: int = 5
    
    # Feature flags
    enable_construction_boost: bool = True
    enable_few_shot_prompts: bool = True
    enable_fallback_detection: bool = True
    enable_confidence_calibration: bool = True
    
    # Resource limits
    max_memory_mb: int = 4096  # Maximum memory for model
    request_timeout_seconds: float = 30.0
    
    def to_detector_configs(self) -> List[DetectorConfig]:
        """Convert to detector configurations."""
        configs = []
        
        # Get mode-specific settings
        mode_settings = MODE_SETTINGS[self.mode]
        
        # Phi4-mini configuration
        phi4_settings = {
            'model_name': self.model_name,
            'ollama_url': self.ollama_url,
            'temperature': mode_settings['temperature'],
            'max_context_tokens': mode_settings['max_context_tokens'],
            'max_chars_per_page': mode_settings['max_chars_per_page'],
            'context_window_pages': mode_settings['context_window_pages'],
            'batch_size': mode_settings['batch_size'],
            'use_few_shot': self.enable_few_shot_prompts and mode_settings.get('use_few_shot', True),
            'compress_whitespace': True,
            'focus_on_headers': True,
            'timeout_seconds': self.request_timeout_seconds,
            'cache_embeddings': self.enable_caching,
            'top_p': mode_settings.get('top_p', 0.9),
            'top_k': mode_settings.get('top_k', 40),
        }
        
        # Focus-specific adjustments
        if self.document_focus == DocumentFocus.CONSTRUCTION:
            phi4_settings['max_chars_per_page'] = min(phi4_settings['max_chars_per_page'] + 200, 1200)
            phi4_settings['context_window_pages'] = min(phi4_settings['context_window_pages'] + 1, 4)
            
        configs.append(DetectorConfig(
            name="phi4_mini",
            enabled=True,
            weight=mode_settings['phi4_weight'],
            config={'phi4_settings': phi4_settings}
        ))
        
        # Add complementary detectors based on mode
        if mode_settings.get('enable_rule_based', True):
            configs.append(DetectorConfig(
                name="rule_based",
                enabled=True,
                weight=mode_settings.get('rule_weight', 0.7),
                config={'min_confidence': 0.6}
            ))
            
        if self.enable_construction_boost and self.document_focus in [DocumentFocus.CONSTRUCTION, DocumentFocus.MIXED]:
            configs.append(DetectorConfig(
                name="construction",
                enabled=True,
                weight=1.2,
                config={}
            ))
            
        return configs


# Mode-specific settings
MODE_SETTINGS = {
    DeploymentMode.DEVELOPMENT: {
        'temperature': 0.1,
        'max_context_tokens': 4096,
        'max_chars_per_page': 1000,
        'context_window_pages': 3,
        'batch_size': 3,
        'use_few_shot': True,
        'phi4_weight': 0.9,
        'rule_weight': 0.7,
        'enable_rule_based': True,
        'top_p': 0.95,
        'top_k': 50,
    },
    
    DeploymentMode.PRODUCTION: {
        'temperature': 0.1,
        'max_context_tokens': 4096,
        'max_chars_per_page': 800,
        'context_window_pages': 2,
        'batch_size': 5,
        'use_few_shot': True,
        'phi4_weight': 0.85,
        'rule_weight': 0.8,
        'enable_rule_based': True,
        'top_p': 0.9,
        'top_k': 40,
    },
    
    DeploymentMode.HIGH_PERFORMANCE: {
        'temperature': 0.2,
        'max_context_tokens': 2048,
        'max_chars_per_page': 500,
        'context_window_pages': 1,
        'batch_size': 10,
        'use_few_shot': False,
        'phi4_weight': 0.8,
        'rule_weight': 0.9,
        'enable_rule_based': True,
        'top_p': 0.85,
        'top_k': 30,
    },
    
    DeploymentMode.HIGH_ACCURACY: {
        'temperature': 0.05,
        'max_context_tokens': 6144,
        'max_chars_per_page': 1200,
        'context_window_pages': 4,
        'batch_size': 2,
        'use_few_shot': True,
        'phi4_weight': 0.95,
        'rule_weight': 0.6,
        'enable_rule_based': True,
        'top_p': 0.95,
        'top_k': 60,
    },
    
    DeploymentMode.LOW_RESOURCE: {
        'temperature': 0.3,
        'max_context_tokens': 1024,
        'max_chars_per_page': 300,
        'context_window_pages': 1,
        'batch_size': 15,
        'use_few_shot': False,
        'phi4_weight': 0.7,
        'rule_weight': 1.0,
        'enable_rule_based': True,
        'top_p': 0.8,
        'top_k': 20,
    }
}


# Preset configurations for common scenarios

def get_preset_config(preset: str) -> Phi4MiniDeploymentConfig:
    """Get a preset configuration for common scenarios."""
    
    presets = {
        "default": Phi4MiniDeploymentConfig(
            mode=DeploymentMode.PRODUCTION,
            document_focus=DocumentFocus.GENERAL
        ),
        
        "construction_expert": Phi4MiniDeploymentConfig(
            mode=DeploymentMode.PRODUCTION,
            document_focus=DocumentFocus.CONSTRUCTION,
            enable_construction_boost=True
        ),
        
        "high_volume": Phi4MiniDeploymentConfig(
            mode=DeploymentMode.HIGH_PERFORMANCE,
            document_focus=DocumentFocus.MIXED,
            max_concurrent_requests=10
        ),
        
        "accuracy_critical": Phi4MiniDeploymentConfig(
            mode=DeploymentMode.HIGH_ACCURACY,
            document_focus=DocumentFocus.GENERAL,
            enable_confidence_calibration=True
        ),
        
        "edge_deployment": Phi4MiniDeploymentConfig(
            mode=DeploymentMode.LOW_RESOURCE,
            document_focus=DocumentFocus.GENERAL,
            max_memory_mb=2048,
            enable_caching=True
        ),
        
        "financial_docs": Phi4MiniDeploymentConfig(
            mode=DeploymentMode.HIGH_ACCURACY,
            document_focus=DocumentFocus.FINANCIAL,
            enable_construction_boost=False
        ),
        
        "email_processing": Phi4MiniDeploymentConfig(
            mode=DeploymentMode.HIGH_PERFORMANCE,
            document_focus=DocumentFocus.EMAIL,
            enable_construction_boost=False
        )
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        
    return presets[preset]


# Environment-based configuration

def get_config_from_env() -> Phi4MiniDeploymentConfig:
    """Create configuration from environment variables."""
    
    config = Phi4MiniDeploymentConfig()
    
    # Read environment variables
    if mode := os.getenv("PHI4_DEPLOYMENT_MODE"):
        config.mode = DeploymentMode(mode)
        
    if focus := os.getenv("PHI4_DOCUMENT_FOCUS"):
        config.document_focus = DocumentFocus(focus)
        
    if model := os.getenv("PHI4_MODEL_NAME"):
        config.model_name = model
        
    if url := os.getenv("OLLAMA_URL"):
        config.ollama_url = url
        
    # Boolean flags
    config.enable_caching = os.getenv("PHI4_ENABLE_CACHING", "true").lower() == "true"
    config.enable_batching = os.getenv("PHI4_ENABLE_BATCHING", "true").lower() == "true"
    config.enable_construction_boost = os.getenv("PHI4_CONSTRUCTION_BOOST", "true").lower() == "true"
    
    # Numeric settings
    if max_concurrent := os.getenv("PHI4_MAX_CONCURRENT"):
        config.max_concurrent_requests = int(max_concurrent)
        
    if timeout := os.getenv("PHI4_TIMEOUT_SECONDS"):
        config.request_timeout_seconds = float(timeout)
        
    if max_memory := os.getenv("PHI4_MAX_MEMORY_MB"):
        config.max_memory_mb = int(max_memory)
        
    return config


# Configuration validation

def validate_config(config: Phi4MiniDeploymentConfig) -> List[str]:
    """Validate configuration and return any warnings."""
    
    warnings = []
    
    # Check Ollama availability
    import httpx
    try:
        response = httpx.get(f"{config.ollama_url}/api/tags", timeout=5.0)
        if response.status_code == 200:
            models = response.json().get('models', [])
            if not any(m['name'] == config.model_name for m in models):
                warnings.append(f"Model {config.model_name} not found in Ollama")
        else:
            warnings.append(f"Ollama API not accessible at {config.ollama_url}")
    except Exception as e:
        warnings.append(f"Cannot connect to Ollama: {e}")
        
    # Check resource settings
    if config.mode == DeploymentMode.LOW_RESOURCE and config.max_memory_mb > 2048:
        warnings.append("Low resource mode but high memory limit set")
        
    if config.mode == DeploymentMode.HIGH_ACCURACY and config.request_timeout_seconds < 20:
        warnings.append("High accuracy mode may need longer timeouts")
        
    # Check feature compatibility
    if config.document_focus == DocumentFocus.CONSTRUCTION and not config.enable_construction_boost:
        warnings.append("Construction focus without construction boost enabled")
        
    return warnings


# Configuration builder for interactive setup

class ConfigBuilder:
    """Interactive configuration builder."""
    
    def __init__(self):
        self.config = Phi4MiniDeploymentConfig()
        
    def set_deployment_mode(self, mode: str) -> 'ConfigBuilder':
        """Set deployment mode."""
        self.config.mode = DeploymentMode(mode)
        return self
        
    def set_document_focus(self, focus: str) -> 'ConfigBuilder':
        """Set document focus."""
        self.config.document_focus = DocumentFocus(focus)
        return self
        
    def set_ollama_url(self, url: str) -> 'ConfigBuilder':
        """Set Ollama URL."""
        self.config.ollama_url = url
        return self
        
    def enable_feature(self, feature: str, enabled: bool = True) -> 'ConfigBuilder':
        """Enable/disable a feature."""
        feature_map = {
            'caching': 'enable_caching',
            'batching': 'enable_batching',
            'construction': 'enable_construction_boost',
            'few_shot': 'enable_few_shot_prompts',
            'fallback': 'enable_fallback_detection',
            'calibration': 'enable_confidence_calibration'
        }
        
        if attr := feature_map.get(feature):
            setattr(self.config, attr, enabled)
        else:
            raise ValueError(f"Unknown feature: {feature}")
            
        return self
        
    def set_resource_limits(
        self,
        max_memory_mb: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        max_concurrent: Optional[int] = None
    ) -> 'ConfigBuilder':
        """Set resource limits."""
        if max_memory_mb is not None:
            self.config.max_memory_mb = max_memory_mb
        if timeout_seconds is not None:
            self.config.request_timeout_seconds = timeout_seconds
        if max_concurrent is not None:
            self.config.max_concurrent_requests = max_concurrent
        return self
        
    def build(self) -> Phi4MiniDeploymentConfig:
        """Build and validate configuration."""
        warnings = validate_config(self.config)
        if warnings:
            print("Configuration warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        return self.config


# Example configurations

def example_configs():
    """Show example configurations."""
    
    # Using presets
    print("=== Preset Configurations ===")
    for preset_name in ["default", "construction_expert", "high_volume"]:
        config = get_preset_config(preset_name)
        print(f"\n{preset_name}:")
        print(f"  Mode: {config.mode}")
        print(f"  Focus: {config.document_focus}")
        print(f"  Detectors: {len(config.to_detector_configs())}")
        
    # Using builder
    print("\n=== Custom Configuration ===")
    custom = (ConfigBuilder()
        .set_deployment_mode("production")
        .set_document_focus("construction")
        .enable_feature("caching", True)
        .enable_feature("construction", True)
        .set_resource_limits(max_memory_mb=3072, timeout_seconds=25.0)
        .build())
    
    print("Custom config built:")
    print(f"  Mode: {custom.mode}")
    print(f"  Memory limit: {custom.max_memory_mb}MB")
    
    # From environment
    print("\n=== Environment Configuration ===")
    env_config = get_config_from_env()
    print(f"Ollama URL: {env_config.ollama_url}")
    print(f"Model: {env_config.model_name}")


if __name__ == "__main__":
    example_configs()