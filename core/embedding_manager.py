import os
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
import hashlib
import pickle
from datetime import datetime
import json

# Embedding model imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from utils.config import Config
from utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Advanced embedding manager with multiple model support and caching"""
    
    def __init__(self, config: Config):
        self.config = config
        self.cache_manager = CacheManager()
        
        # Available embedding models
        self.available_models = {
            'sentence_transformers': {
                'all-MiniLM-L6-v2': {
                    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                    'dimension': 384,
                    'max_seq_length': 256,
                    'best_for': ['general', 'similarity', 'clustering']
                },
                'multi-qa-MiniLM-L6-cos-v1': {
                    'model_name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
                    'dimension': 384,
                    'max_seq_length': 512,
                    'best_for': ['question_answering', 'retrieval']
                },
                'all-mpnet-base-v2': {
                    'model_name': 'sentence-transformers/all-mpnet-base-v2',
                    'dimension': 768,
                    'max_seq_length': 384,
                    'best_for': ['high_quality', 'semantic_search']
                }
            },
            'openai': {
                'text-embedding-ada-002': {
                    'model_name': 'text-embedding-ada-002',
                    'dimension': 1536,
                    'max_tokens': 8191,
                    'best_for': ['general', 'high_quality']
                },
                'text-embedding-3-small': {
                    'model_name': 'text-embedding-3-small',
                    'dimension': 1536,
                    'max_tokens': 8191,
                    'best_for': ['cost_effective', 'general']
                },
                'text-embedding-3-large': {
                    'model_name': 'text-embedding-3-large',
                    'dimension': 3072,
                    'max_tokens': 8191,
                    'best_for': ['highest_quality', 'complex_tasks']
                }
            },
            'custom': {
                'industrial-bert': {
                    'model_name': 'custom/industrial-bert-base',
                    'dimension': 768,
                    'max_seq_length': 512,
                    'best_for': ['industrial', 'domain_specific']
                }
            }
        }
        
        # Loaded models cache
        self.loaded_models = {}
        
        # Default model configuration
        self.default_config = {
            'primary_model': 'all-MiniLM-L6-v2',
            'fallback_model': 'simple_hash',
            'batch_size': 32,
            'normalize_embeddings': True,
            'cache_embeddings': True
        }
        
        # Performance statistics
        self.embedding_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'model_usage': {},
            'average_embedding_time': 0,
            'total_texts_embedded': 0
        }
        
        # Initialize primary model
        self._initialize_primary_model()
    
    def _initialize_primary_model(self):
        """Initialize the primary embedding model"""
        try:
            primary_model = self.default_config['primary_model']
            
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                model_info = self.available_models['sentence_transformers'].get(primary_model)
                if model_info:
                    self.loaded_models[primary_model] = SentenceTransformer(
                        model_info['model_name']
                    )
                    logger.info(f"Primary embedding model '{primary_model}' loaded successfully")
                    return
            
            # Fallback to simple embeddings
            logger.warning("Advanced embedding models not available, using simple hash-based embeddings")
            self.loaded_models['simple_hash'] = SimpleHashEmbedding()
            
        except Exception as e:
            logger.error(f"Primary model initialization failed: {str(e)}")
            self.loaded_models['simple_hash'] = SimpleHashEmbedding()
    
    def generate_embeddings(
        self, 
        texts: List[str], 
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            start_time = datetime.now()
            self.embedding_stats['total_requests'] += 1
            
            # Use default model if none specified
            if model_name is None:
                model_name = self.default_config['primary_model']
            
            # Check cache first
            cached_embeddings = self._check_embedding_cache(texts, model_name)
            if cached_embeddings:
                self.embedding_stats['cache_hits'] += 1
                return cached_embeddings
            
            # Generate embeddings
            embeddings = self._generate_embeddings_batch(
                texts, model_name, batch_size or self.default_config['batch_size']
            )
            
            # Normalize if requested
            if normalize and self.default_config['normalize_embeddings']:
                embeddings = self._normalize_embeddings(embeddings)
            
            # Cache results
            if self.default_config['cache_embeddings']:
                self._cache_embeddings(texts, embeddings, model_name)
            
            # Update statistics
            embedding_time = (datetime.now() - start_time).total_seconds()
            self._update_embedding_stats(model_name, embedding_time, len(texts))
            
            self.embedding_stats['total_texts_embedded'] += len(texts)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            # Fallback to simple embeddings
            return self._generate_simple_embeddings(texts)
    
    def _generate_embeddings_batch(
        self, 
        texts: List[str], 
        model_name: str, 
        batch_size: int
    ) -> List[List[float]]:
        """Generate embeddings in batches for memory efficiency"""
        all_embeddings = []
        
        try:
            # Load model if not already loaded
            if model_name not in self.loaded_models:
                self._load_model(model_name)
            
            model = self.loaded_models.get(model_name)
            if not model:
                raise ValueError(f"Model {model_name} not available")
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                if hasattr(model, 'encode'):
                    # Sentence Transformers model
                    batch_embeddings = model.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    ).tolist()
                elif hasattr(model, 'embed_documents'):
                    # LangChain style model
                    batch_embeddings = model.embed_documents(batch)
                else:
                    # Custom model interface
                    batch_embeddings = model.generate_embeddings(batch)
                
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {str(e)}")
            raise
    
    def _load_model(self, model_name: str):
        """Load an embedding model on demand"""
        try:
            # Check if it's a sentence transformers model
            if model_name in self.available_models['sentence_transformers']:
                if SENTENCE_TRANSFORMERS_AVAILABLE:
                    model_info = self.available_models['sentence_transformers'][model_name]
                    self.loaded_models[model_name] = SentenceTransformer(
                        model_info['model_name']
                    )
                    logger.info(f"Loaded SentenceTransformers model: {model_name}")
                else:
                    raise ImportError("SentenceTransformers not available")
            
            # Check if it's an OpenAI model
            elif model_name in self.available_models['openai']:
                if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
                    self.loaded_models[model_name] = OpenAIEmbedding(model_name)
                    logger.info(f"Loaded OpenAI model: {model_name}")
                else:
                    raise ImportError("OpenAI not available or API key missing")
            
            # Check if it's a custom model
            elif model_name in self.available_models['custom']:
                if TRANSFORMERS_AVAILABLE:
                    model_info = self.available_models['custom'][model_name]
                    self.loaded_models[model_name] = CustomTransformerEmbedding(
                        model_info['model_name']
                    )
                    logger.info(f"Loaded custom model: {model_name}")
                else:
                    raise ImportError("Transformers not available")
            
            else:
                raise ValueError(f"Unknown model: {model_name}")
                
        except Exception as e:
            logger.error(f"Model loading failed for {model_name}: {str(e)}")
            # Fallback to simple embedding
            self.loaded_models[model_name] = SimpleHashEmbedding()
    
    def _check_embedding_cache(
        self, 
        texts: List[str], 
        model_name: str
    ) -> Optional[List[List[float]]]:
        """Check if embeddings are already cached"""
        try:
            # Create cache key from texts and model
            text_hash = hashlib.md5('|'.join(texts).encode()).hexdigest()
            cache_key = f"embeddings_{model_name}_{text_hash}"
            
            cached_data = self.cache_manager.get(cache_key)
            if cached_data:
                return cached_data
            
            return None
            
        except Exception as e:
            logger.error(f"Cache check failed: {str(e)}")
            return None
    
    def _cache_embeddings(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        model_name: str
    ):
        """Cache embeddings for future use"""
        try:
            text_hash = hashlib.md5('|'.join(texts).encode()).hexdigest()
            cache_key = f"embeddings_{model_name}_{text_hash}"
            
            # Cache for 24 hours
            self.cache_manager.set(cache_key, embeddings, ttl=86400)
            
        except Exception as e:
            logger.error(f"Embedding caching failed: {str(e)}")
    
    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize embeddings to unit length"""
        try:
            normalized = []
            for embedding in embeddings:
                arr = np.array(embedding)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    normalized.append((arr / norm).tolist())
                else:
                    normalized.append(embedding)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Embedding normalization failed: {str(e)}")
            return embeddings
    
    def _generate_simple_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate simple hash-based embeddings as fallback"""
        try:
            simple_model = SimpleHashEmbedding()
            return simple_model.generate_embeddings(texts)
            
        except Exception as e:
            logger.error(f"Simple embedding generation failed: {str(e)}")
            # Return zero embeddings as last resort
            return [[0.0] * 384 for _ in texts]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        for provider, models in self.available_models.items():
            if model_name in models:
                info = models[model_name].copy()
                info['provider'] = provider
                info['loaded'] = model_name in self.loaded_models
                return info
        
        return {'error': f'Model {model_name} not found'}
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models by provider"""
        available = {}
        for provider, models in self.available_models.items():
            available[provider] = list(models.keys())
        
        return available
    
    def calculate_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            arr1 = np.array(embedding1)
            arr2 = np.array(embedding2)
            
            dot_product = np.dot(arr1, arr2)
            norm1 = np.linalg.norm(arr1)
            norm2 = np.linalg.norm(arr2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {str(e)}")
            return 0.0
    
    def find_most_similar(
        self, 
        query_embedding: List[float], 
        candidate_embeddings: List[List[float]], 
        top_k: int = 5
    ) -> List[tuple]:
        """Find most similar embeddings to query"""
        try:
            similarities = []
            
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.calculate_similarity(query_embedding, candidate)
                similarities.append((i, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            return []
    
    def _update_embedding_stats(self, model_name: str, embedding_time: float, text_count: int):
        """Update embedding performance statistics"""
        try:
            # Update model usage
            if model_name not in self.embedding_stats['model_usage']:
                self.embedding_stats['model_usage'][model_name] = {
                    'requests': 0,
                    'total_time': 0,
                    'avg_time': 0,
                    'texts_processed': 0
                }
            
            stats = self.embedding_stats['model_usage'][model_name]
            stats['requests'] += 1
            stats['total_time'] += embedding_time
            stats['avg_time'] = stats['total_time'] / stats['requests']
            stats['texts_processed'] += text_count
            
            # Update overall average
            total_requests = self.embedding_stats['total_requests']
            current_avg = self.embedding_stats['average_embedding_time']
            new_avg = ((current_avg * (total_requests - 1)) + embedding_time) / total_requests
            self.embedding_stats['average_embedding_time'] = new_avg
            
        except Exception as e:
            logger.error(f"Stats update failed: {str(e)}")
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding performance statistics"""
        stats = self.embedding_stats.copy()
        
        # Add cache hit rate
        if stats['total_requests'] > 0:
            stats['cache_hit_rate'] = (stats['cache_hits'] / stats['total_requests']) * 100
        else:
            stats['cache_hit_rate'] = 0
        
        return stats
    
    def clear_cache(self):
        """Clear embedding cache"""
        try:
            # This would clear the embedding-specific cache
            # Implementation depends on cache manager
            logger.info("Embedding cache cleared")
            
        except Exception as e:
            logger.error(f"Cache clearing failed: {str(e)}")
    
    def save_model_state(self, filepath: str):
        """Save model state to disk"""
        try:
            state = {
                'loaded_models': list(self.loaded_models.keys()),
                'stats': self.embedding_stats,
                'config': self.default_config
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Model state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Model state saving failed: {str(e)}")


class SimpleHashEmbedding:
    """Simple hash-based embedding as fallback"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate simple hash-based embeddings"""
        embeddings = []
        
        for text in texts:
            # Create deterministic embedding from text hash
            hash_obj = hashlib.md5(text.encode())
            
            # Create multiple hashes to fill dimension
            embedding = []
            for i in range(0, self.dimension, 16):  # MD5 produces 16 bytes
                if i == 0:
                    hash_bytes = hash_obj.digest()
                else:
                    # Create variations by adding salt
                    salted_text = f"{text}_{i}"
                    hash_bytes = hashlib.md5(salted_text.encode()).digest()
                
                # Convert bytes to normalized floats
                for byte in hash_bytes:
                    if len(embedding) < self.dimension:
                        embedding.append((byte / 255.0) - 0.5)  # Center around 0
            
            embeddings.append(embedding[:self.dimension])
        
        return embeddings
    
    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Compatibility method for SentenceTransformers interface"""
        return self.generate_embeddings(texts)


class OpenAIEmbedding:
    """OpenAI embedding model wrapper"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            
            embeddings = []
            for item in response.data:
                embeddings.append(item.embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {str(e)}")
            raise
    
    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Compatibility method for SentenceTransformers interface"""
        return self.generate_embeddings(texts)


class CustomTransformerEmbedding:
    """Custom transformer model wrapper"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using custom transformer model"""
        try:
            embeddings = []
            
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(self.device)
                
                # Generate embedding
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling of last hidden states
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                    embeddings.append(embedding.cpu().numpy().tolist())
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Custom transformer embedding generation failed: {str(e)}")
            raise
    
    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Compatibility method for SentenceTransformers interface"""
        return self.generate_embeddings(texts)

