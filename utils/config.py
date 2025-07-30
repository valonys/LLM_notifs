import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for AI models"""
    name: str
    provider: str
    api_key_env: str
    max_tokens: int = 2000
    temperature: float = 0.7
    timeout: int = 60
    retry_attempts: int = 3

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""
    model_name: str
    dimension: int
    max_sequence_length: int
    batch_size: int = 32
    normalize: bool = True
    cache_embeddings: bool = True

@dataclass
class CacheConfig:
    """Configuration for caching system"""
    enable_memory_cache: bool = True
    enable_disk_cache: bool = True
    max_memory_items: int = 1000
    default_ttl: int = 3600
    cleanup_interval: int = 3600

@dataclass
class DatabaseConfig:
    """Configuration for database connections"""
    type: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    name: str = "digitwin"
    username: str = ""
    password: str = ""
    connection_pool_size: int = 10

class Config:
    """Comprehensive configuration management system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config.json"
        self.config_data = {}
        
        # Default configurations
        self._init_default_configs()
        
        # Load configuration
        self._load_configuration()
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info("Configuration system initialized")
    
    def _init_default_configs(self):
        """Initialize default configuration values"""
        self.default_config = {
            "app": {
                "name": "DigiTwin Enhanced RAG System",
                "version": "2.0.0",
                "debug": False,
                "log_level": "INFO",
                "host": "0.0.0.0",
                "port": 5000
            },
            "models": {
                "primary_model": "EE Smartest Agent",
                "fallback_model": "EdJa-Valonys",
                "available_models": [
                    {
                        "name": "EE Smartest Agent",
                        "provider": "xai",
                        "api_key_env": "API_KEY",
                        "model_name": "grok-4-latest",
                        "max_tokens": 4000,
                        "temperature": 0.7
                    },
                    {
                        "name": "EdJa-Valonys",
                        "provider": "cerebras",
                        "api_key_env": "CEREBRAS_API_KEY",
                        "model_name": "llama3.1-70b",
                        "max_tokens": 2000,
                        "temperature": 0.6
                    },
                    {
                        "name": "JI Divine Agent",
                        "provider": "cerebras",
                        "api_key_env": "CEREBRAS_API_KEY",
                        "model_name": "llama3.1-8b",
                        "max_tokens": 2000,
                        "temperature": 0.8
                    },
                    {
                        "name": "OpenAI GPT-4",
                        "provider": "openai",
                        "api_key_env": "OPENAI_API_KEY",
                        "model_name": "gpt-4",
                        "max_tokens": 3000,
                        "temperature": 0.7
                    }
                ]
            },
            "embeddings": {
                "primary_model": "all-MiniLM-L6-v2",
                "fallback_model": "simple_hash",
                "batch_size": 32,
                "normalize": True,
                "cache_embeddings": True,
                "available_models": [
                    {
                        "model_name": "all-MiniLM-L6-v2",
                        "dimension": 384,
                        "max_sequence_length": 256,
                        "best_for": ["general", "similarity"]
                    },
                    {
                        "model_name": "multi-qa-MiniLM-L6-cos-v1",
                        "dimension": 384,
                        "max_sequence_length": 512,
                        "best_for": ["question_answering", "retrieval"]
                    },
                    {
                        "model_name": "all-mpnet-base-v2",
                        "dimension": 768,
                        "max_sequence_length": 384,
                        "best_for": ["high_quality", "semantic_search"]
                    }
                ]
            },
            "chunking": {
                "strategy": "hierarchical",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "min_chunk_size": 100,
                "max_chunk_size": 2000,
                "semantic_similarity_threshold": 0.7,
                "enable_overlapping_chunks": True
            },
            "retrieval": {
                "default_k": 5,
                "max_k": 20,
                "similarity_threshold": 0.5,
                "rerank_results": True,
                "enable_query_expansion": True,
                "hybrid_search_weights": {
                    "dense": 0.7,
                    "sparse": 0.3
                }
            },
            "cache": {
                "enable_memory_cache": True,
                "enable_disk_cache": True,
                "max_memory_items": 1000,
                "default_ttl": 3600,
                "cleanup_interval": 3600,
                "cache_directory": ".cache"
            },
            "database": {
                "type": "sqlite",
                "name": "digitwin.db",
                "enable_connection_pooling": True,
                "connection_pool_size": 10
            },
            "monitoring": {
                "enable_sentry": True,
                "enable_prometheus": True,
                "prometheus_port": 8001,
                "log_requests": True,
                "performance_monitoring": True
            },
            "security": {
                "enable_rate_limiting": True,
                "requests_per_minute": 60,
                "enable_api_key_validation": True,
                "require_https": False
            },
            "industrial": {
                "enable_domain_specific_models": True,
                "safety_keywords": [
                    "safety", "hazard", "risk", "danger", "warning",
                    "violation", "incident", "accident", "injury"
                ],
                "equipment_keywords": [
                    "equipment", "machinery", "component", "system",
                    "pump", "valve", "compressor", "turbine"
                ],
                "compliance_keywords": [
                    "compliance", "regulation", "standard", "procedure",
                    "audit", "requirement", "certification"
                ]
            }
        }
    
    def _load_configuration(self):
        """Load configuration from file and environment variables"""
        # Start with defaults
        self.config_data = self.default_config.copy()
        
        # Load from config file if it exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                self._deep_merge(self.config_data, file_config)
                logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                logger.error(f"Failed to load config file: {str(e)}")
        
        # Override with environment variables
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # App configuration
        if os.getenv('DEBUG'):
            self.config_data['app']['debug'] = os.getenv('DEBUG').lower() == 'true'
        
        if os.getenv('LOG_LEVEL'):
            self.config_data['app']['log_level'] = os.getenv('LOG_LEVEL')
        
        if os.getenv('PORT'):
            try:
                self.config_data['app']['port'] = int(os.getenv('PORT'))
            except ValueError:
                logger.warning("Invalid PORT environment variable")
        
        # API Keys - these are critical
        api_keys = {
            'API_KEY': 'XAI API Key',
            'CEREBRAS_API_KEY': 'Cerebras API Key',
            'OPENAI_API_KEY': 'OpenAI API Key',
            'HUGGINGFACE_TOKEN': 'HuggingFace Token',
            'SENTRY_DSN': 'Sentry DSN'
        }
        
        for env_key, description in api_keys.items():
            value = os.getenv(env_key)
            if value:
                self.config_data.setdefault('api_keys', {})[env_key] = value
                logger.info(f"{description} loaded from environment")
            else:
                logger.warning(f"{description} not found in environment")
        
        # Cache configuration
        if os.getenv('CACHE_TTL'):
            try:
                self.config_data['cache']['default_ttl'] = int(os.getenv('CACHE_TTL'))
            except ValueError:
                logger.warning("Invalid CACHE_TTL environment variable")
        
        # Database configuration
        if os.getenv('DATABASE_URL'):
            # Parse database URL (e.g., postgresql://user:pass@localhost:5432/db)
            db_url = os.getenv('DATABASE_URL')
            # This would need proper URL parsing in production
            self.config_data['database']['url'] = db_url
    
    def _deep_merge(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Deep merge two dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _validate_configuration(self):
        """Validate configuration values"""
        try:
            # Validate required sections
            required_sections = ['app', 'models', 'embeddings', 'chunking', 'retrieval']
            for section in required_sections:
                if section not in self.config_data:
                    logger.error(f"Required configuration section '{section}' missing")
            
            # Validate model configurations
            self._validate_model_configs()
            
            # Validate numeric values
            self._validate_numeric_values()
            
            # Validate API keys
            self._validate_api_keys()
            
            logger.info("Configuration validation completed")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
    
    def _validate_model_configs(self):
        """Validate model configurations"""
        models = self.config_data.get('models', {}).get('available_models', [])
        
        for model in models:
            required_fields = ['name', 'provider', 'api_key_env']
            for field in required_fields:
                if field not in model:
                    logger.warning(f"Model configuration missing required field: {field}")
            
            # Validate numeric values
            if 'max_tokens' in model and not isinstance(model['max_tokens'], int):
                logger.warning(f"Invalid max_tokens for model {model.get('name', 'unknown')}")
            
            if 'temperature' in model:
                temp = model['temperature']
                if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                    logger.warning(f"Invalid temperature for model {model.get('name', 'unknown')}")
    
    def _validate_numeric_values(self):
        """Validate numeric configuration values"""
        numeric_configs = [
            ('chunking.chunk_size', 100, 10000),
            ('chunking.chunk_overlap', 0, 1000),
            ('retrieval.default_k', 1, 50),
            ('cache.max_memory_items', 100, 10000),
            ('cache.default_ttl', 60, 86400)
        ]
        
        for config_path, min_val, max_val in numeric_configs:
            value = self.get_nested_value(config_path)
            if value is not None:
                if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                    logger.warning(f"Invalid value for {config_path}: {value}")
    
    def _validate_api_keys(self):
        """Validate API key availability"""
        models = self.config_data.get('models', {}).get('available_models', [])
        api_keys = self.config_data.get('api_keys', {})
        
        for model in models:
            api_key_env = model.get('api_key_env')
            if api_key_env and api_key_env not in api_keys:
                logger.warning(f"API key for {model.get('name', 'unknown')} not found: {api_key_env}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        return self.get_nested_value(key, default)
    
    def get_nested_value(self, key_path: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config_data
        
        for key_part in keys[:-1]:
            if key_part not in config:
                config[key_part] = {}
            config = config[key_part]
        
        config[keys[-1]] = value
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        models = self.config_data.get('models', {}).get('available_models', [])
        
        for model in models:
            if model.get('name') == model_name:
                return ModelConfig(
                    name=model['name'],
                    provider=model['provider'],
                    api_key_env=model['api_key_env'],
                    max_tokens=model.get('max_tokens', 2000),
                    temperature=model.get('temperature', 0.7),
                    timeout=model.get('timeout', 60),
                    retry_attempts=model.get('retry_attempts', 3)
                )
        
        return None
    
    def get_embedding_config(self) -> EmbeddingConfig:
        """Get embedding configuration"""
        embedding_config = self.config_data.get('embeddings', {})
        
        return EmbeddingConfig(
            model_name=embedding_config.get('primary_model', 'all-MiniLM-L6-v2'),
            dimension=384,  # This would be looked up based on model
            max_sequence_length=256,
            batch_size=embedding_config.get('batch_size', 32),
            normalize=embedding_config.get('normalize', True),
            cache_embeddings=embedding_config.get('cache_embeddings', True)
        )
    
    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration"""
        cache_config = self.config_data.get('cache', {})
        
        return CacheConfig(
            enable_memory_cache=cache_config.get('enable_memory_cache', True),
            enable_disk_cache=cache_config.get('enable_disk_cache', True),
            max_memory_items=cache_config.get('max_memory_items', 1000),
            default_ttl=cache_config.get('default_ttl', 3600),
            cleanup_interval=cache_config.get('cleanup_interval', 3600)
        )
    
    def get_api_key(self, key_name: str) -> Optional[str]:
        """Get API key value"""
        api_keys = self.config_data.get('api_keys', {})
        return api_keys.get(key_name) or os.getenv(key_name)
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        models = self.config_data.get('models', {}).get('available_models', [])
        return [model.get('name') for model in models if model.get('name')]
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        return self.config_data.get('app', {}).get('debug', False)
    
    def get_log_level(self) -> str:
        """Get logging level"""
        return self.config_data.get('app', {}).get('log_level', 'INFO')
    
    def save_configuration(self, filepath: Optional[str] = None):
        """Save current configuration to file"""
        try:
            output_file = filepath or self.config_file
            
            with open(output_file, 'w') as f:
                json.dump(self.config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
    
    def reload_configuration(self):
        """Reload configuration from file"""
        self._load_configuration()
        self._validate_configuration()
        logger.info("Configuration reloaded")
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get complete configuration dictionary"""
        return self.config_data.copy()
    
    def export_config_template(self, filepath: str):
        """Export configuration template"""
        try:
            template = {
                "// Configuration Template": "Edit values as needed",
                **self.default_config
            }
            
            with open(filepath, 'w') as f:
                json.dump(template, f, indent=2)
            
            logger.info(f"Configuration template exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export config template: {str(e)}")
    
    def validate_runtime_config(self) -> Dict[str, List[str]]:
        """Validate runtime configuration and return issues"""
        issues = {
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Check API keys
        models = self.get_available_models()
        for model_name in models:
            model_config = self.get_model_config(model_name)
            if model_config:
                api_key = self.get_api_key(model_config.api_key_env)
                if not api_key:
                    issues['warnings'].append(f"API key missing for {model_name}")
        
        # Check file permissions
        cache_dir = self.get('cache.cache_directory', '.cache')
        if not os.access(cache_dir, os.W_OK):
            try:
                os.makedirs(cache_dir, exist_ok=True)
                issues['info'].append(f"Created cache directory: {cache_dir}")
            except Exception:
                issues['errors'].append(f"Cannot create cache directory: {cache_dir}")
        
        return issues

