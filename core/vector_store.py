import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import json
import pickle
import os
from dataclasses import dataclass

# Vector and embedding imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from langchain.schema import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .chunking_strategies import ChunkingManager
from .embedding_manager import EmbeddingManager
from utils.config import Config

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Enhanced search result with metadata"""
    content: str
    metadata: Dict[str, Any]
    score: float
    chunk_type: str
    source: str
    retrieval_method: str

class EnhancedVectorStore:
    """Enhanced vector store with multiple embedding strategies and retrieval methods"""
    
    def __init__(self, config: Config):
        self.config = config
        self.chunking_manager = ChunkingManager(config)
        self.embedding_manager = EmbeddingManager(config)
        
        # Initialize storage backends
        self.faiss_index = None
        self.chroma_client = None
        self.documents = []
        self.metadata = []
        self.embeddings = []
        
        # Performance tracking
        self.retrieval_stats = {
            'total_queries': 0,
            'avg_retrieval_time': 0,
            'cache_hits': 0
        }
        
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize available storage backends"""
        try:
            if CHROMADB_AVAILABLE:
                self.chroma_client = chromadb.Client()
                logger.info("ChromaDB initialized successfully")
            elif FAISS_AVAILABLE:
                logger.info("Using FAISS for vector storage")
            else:
                logger.warning("No vector database available, using fallback storage")
        except Exception as e:
            logger.error(f"Storage initialization error: {str(e)}")
    
    def create_enhanced_store(self, documents: List[LCDocument]) -> bool:
        """Create enhanced vector store with hierarchical chunking"""
        try:
            logger.info(f"Creating enhanced store for {len(documents)} documents")
            
            # Apply chunking strategies
            chunked_docs = self.chunking_manager.apply_hierarchical_chunking(documents)
            
            # Generate embeddings for all chunks
            embeddings = self.embedding_manager.generate_embeddings(
                [doc.page_content for doc in chunked_docs]
            )
            
            # Store documents and embeddings
            self.documents = chunked_docs
            self.embeddings = embeddings
            self.metadata = [doc.metadata for doc in chunked_docs]
            
            # Create FAISS index if available
            if FAISS_AVAILABLE and embeddings:
                dimension = len(embeddings[0])
                self.faiss_index = faiss.IndexFlatIP(dimension)
                embeddings_array = np.array(embeddings).astype('float32')
                self.faiss_index.add(embeddings_array)
                logger.info(f"FAISS index created with {len(embeddings)} vectors")
            
            # Create ChromaDB collection if available
            if CHROMADB_AVAILABLE and self.chroma_client:
                try:
                    collection = self.chroma_client.create_collection("documents")
                    collection.add(
                        embeddings=embeddings,
                        documents=[doc.page_content for doc in chunked_docs],
                        metadatas=self.metadata,
                        ids=[f"doc_{i}" for i in range(len(chunked_docs))]
                    )
                    logger.info("ChromaDB collection created successfully")
                except Exception as e:
                    logger.warning(f"ChromaDB creation failed: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Enhanced store creation failed: {str(e)}")
            return False
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        retrieval_mode: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Enhanced similarity search with multiple retrieval strategies"""
        start_time = datetime.now()
        
        try:
            self.retrieval_stats['total_queries'] += 1
            
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_embeddings([query])[0]
            
            results = []
            
            if retrieval_mode in ["dense", "hybrid"]:
                dense_results = self._dense_retrieval(query, query_embedding, k, filters)
                results.extend(dense_results)
            
            if retrieval_mode in ["sparse", "hybrid"]:
                sparse_results = self._sparse_retrieval(query, k, filters)
                results.extend(sparse_results)
            
            # Rerank and deduplicate
            if retrieval_mode == "hybrid":
                results = self._rerank_results(results, query_embedding, k)
            
            # Update performance stats
            retrieval_time = (datetime.now() - start_time).total_seconds()
            self._update_retrieval_stats(retrieval_time)
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            return []
    
    def _dense_retrieval(
        self, 
        query: str, 
        query_embedding: List[float], 
        k: int, 
        filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Dense retrieval using vector similarity"""
        results = []
        
        try:
            if self.faiss_index and self.embeddings:
                # FAISS search
                query_vector = np.array([query_embedding]).astype('float32')
                scores, indices = self.faiss_index.search(query_vector, min(k * 2, len(self.documents)))
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.documents):
                        doc = self.documents[idx]
                        metadata = self.metadata[idx]
                        
                        # Apply filters if specified
                        if filters and not self._apply_filters(metadata, filters):
                            continue
                        
                        results.append(SearchResult(
                            content=doc.page_content,
                            metadata=metadata,
                            score=float(score),
                            chunk_type=metadata.get('chunk_type', 'standard'),
                            source=metadata.get('source', 'unknown'),
                            retrieval_method='dense'
                        ))
            
            return results
            
        except Exception as e:
            logger.error(f"Dense retrieval failed: {str(e)}")
            return []
    
    def _sparse_retrieval(
        self, 
        query: str, 
        k: int, 
        filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Sparse retrieval using keyword matching"""
        results = []
        
        try:
            query_terms = set(query.lower().split())
            
            for i, doc in enumerate(self.documents):
                doc_terms = set(doc.page_content.lower().split())
                
                # Calculate term overlap score
                overlap = len(query_terms.intersection(doc_terms))
                if overlap > 0:
                    score = overlap / len(query_terms.union(doc_terms))
                    
                    metadata = self.metadata[i]
                    
                    # Apply filters if specified
                    if filters and not self._apply_filters(metadata, filters):
                        continue
                    
                    results.append(SearchResult(
                        content=doc.page_content,
                        metadata=metadata,
                        score=score,
                        chunk_type=metadata.get('chunk_type', 'standard'),
                        source=metadata.get('source', 'unknown'),
                        retrieval_method='sparse'
                    ))
            
            # Sort by score
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {str(e)}")
            return []
    
    def _rerank_results(
        self, 
        results: List[SearchResult], 
        query_embedding: List[float], 
        k: int
    ) -> List[SearchResult]:
        """Rerank and deduplicate results from multiple retrieval methods"""
        try:
            # Remove duplicates based on content similarity
            unique_results = []
            seen_contents = set()
            
            for result in results:
                content_hash = hash(result.content[:100])  # Use first 100 chars for dedup
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    unique_results.append(result)
            
            # Sort by combined score (could implement more sophisticated reranking)
            unique_results.sort(key=lambda x: x.score, reverse=True)
            
            return unique_results[:k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            return results[:k]
    
    def _apply_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply metadata filters to search results"""
        try:
            for key, value in filters.items():
                if key not in metadata:
                    return False
                
                if isinstance(value, list):
                    if metadata[key] not in value:
                        return False
                else:
                    if metadata[key] != value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Filter application failed: {str(e)}")
            return True
    
    def _update_retrieval_stats(self, retrieval_time: float):
        """Update retrieval performance statistics"""
        try:
            total_queries = self.retrieval_stats['total_queries']
            current_avg = self.retrieval_stats['avg_retrieval_time']
            
            # Update moving average
            new_avg = ((current_avg * (total_queries - 1)) + retrieval_time) / total_queries
            self.retrieval_stats['avg_retrieval_time'] = new_avg
            
        except Exception as e:
            logger.error(f"Stats update failed: {str(e)}")
    
    def generate_document_insights(self) -> List[Dict[str, Any]]:
        """Generate automated insights from document corpus"""
        insights = []
        
        try:
            if not self.documents:
                return insights
            
            # Document type distribution
            doc_types = {}
            for metadata in self.metadata:
                doc_type = metadata.get('file_type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            insights.append({
                'category': 'Document Distribution',
                'description': f"Processed {len(self.documents)} chunks from {len(doc_types)} file types",
                'details': doc_types
            })
            
            # Chunk type analysis
            chunk_types = {}
            for metadata in self.metadata:
                chunk_type = metadata.get('chunk_type', 'standard')
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            insights.append({
                'category': 'Chunking Analysis',
                'description': f"Applied {len(chunk_types)} chunking strategies",
                'details': chunk_types
            })
            
            # Content length analysis
            content_lengths = [len(doc.page_content) for doc in self.documents]
            avg_length = np.mean(content_lengths)
            
            insights.append({
                'category': 'Content Analysis',
                'description': f"Average chunk length: {avg_length:.0f} characters",
                'recommendation': "Consider adjusting chunk size if retrieval quality is poor"
            })
            
            return insights
            
        except Exception as e:
            logger.error(f"Insight generation failed: {str(e)}")
            return insights
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval performance statistics"""
        return self.retrieval_stats.copy()
    
    def save_store(self, path: str):
        """Save vector store to disk"""
        try:
            store_data = {
                'documents': self.documents,
                'metadata': self.metadata,
                'embeddings': self.embeddings,
                'stats': self.retrieval_stats
            }
            
            with open(path, 'wb') as f:
                pickle.dump(store_data, f)
            
            logger.info(f"Vector store saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save store: {str(e)}")
    
    def load_store(self, path: str) -> bool:
        """Load vector store from disk"""
        try:
            if not os.path.exists(path):
                return False
            
            with open(path, 'rb') as f:
                store_data = pickle.load(f)
            
            self.documents = store_data['documents']
            self.metadata = store_data['metadata']
            self.embeddings = store_data['embeddings']
            self.retrieval_stats = store_data.get('stats', self.retrieval_stats)
            
            # Rebuild FAISS index
            if FAISS_AVAILABLE and self.embeddings:
                dimension = len(self.embeddings[0])
                self.faiss_index = faiss.IndexFlatIP(dimension)
                embeddings_array = np.array(self.embeddings).astype('float32')
                self.faiss_index.add(embeddings_array)
            
            logger.info(f"Vector store loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load store: {str(e)}")
            return False
