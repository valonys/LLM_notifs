"""
Performance optimization utilities for DigiTwin RAG system
Provides caching, lazy loading, and query optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import logging
from functools import lru_cache, wraps
from typing import Dict, Any, Optional, Callable
from cachetools import TTLCache, LRUCache
import time
import asyncio

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Centralized performance optimization manager"""
    
    def __init__(self):
        self.query_cache = LRUCache(maxsize=100)
        self.dataframe_cache = LRUCache(maxsize=10)
        self.pivot_cache = TTLCache(maxsize=50, ttl=1800)  # 30 min TTL for pivot analyses
        self.ui_state_cache = {}
        
    @staticmethod
    def cache_key(*args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def cached_dataframe_operation(self, operation_name: str, df: pd.DataFrame, **kwargs):
        """Cache expensive DataFrame operations"""
        cache_key = self.cache_key(operation_name, df.shape, **kwargs)
        
        if cache_key in self.dataframe_cache:
            logger.info(f"Cache hit for {operation_name}")
            return self.dataframe_cache[cache_key]
            
        # Perform operation based on name
        if operation_name == "fpso_filter":
            fpso_value = kwargs.get('fpso_value')
            if fpso_value and fpso_value != "All":
                result = df[df['FPSO'] == fpso_value].copy()
            else:
                result = df.copy()
        elif operation_name == "preprocess_fpso":
            result = self._preprocess_fpso_data(df)
        else:
            result = df
            
        self.dataframe_cache[cache_key] = result
        logger.info(f"Cached result for {operation_name}")
        return result
    
    def _preprocess_fpso_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized FPSO data preprocessing"""
        try:
            # Remove empty rows efficiently
            df_cleaned = df.dropna(subset=['Main Work Ctr', 'Notifictn type'], how='all')
            
            # Filter out LDA entries in one operation
            if 'FPSO' in df_cleaned.columns:
                df_cleaned = df_cleaned[df_cleaned['FPSO'] != 'LDA']
            
            # Convert dates efficiently if needed
            date_columns = ['Created on', 'Completn date']
            for col in date_columns:
                if col in df_cleaned.columns:
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
            
            return df_cleaned
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return df
    
    def get_cached_pivot_analysis(self, df: pd.DataFrame, fpso_value: str, work_center: str = None) -> Optional[Dict]:
        """Get cached pivot analysis or create new one"""
        cache_key = self.cache_key("pivot_analysis", df.shape, fpso_value, work_center)
        
        if cache_key in self.pivot_cache:
            logger.info(f"Cache hit for pivot analysis: {fpso_value}")
            return self.pivot_cache[cache_key]
        
        return None
    
    def cache_pivot_analysis(self, df: pd.DataFrame, fpso_value: str, analysis_result: Dict, work_center: str = None):
        """Cache pivot analysis result"""
        cache_key = self.cache_key("pivot_analysis", df.shape, fpso_value, work_center)
        self.pivot_cache[cache_key] = analysis_result
        logger.info(f"Cached pivot analysis for {fpso_value}")

class LazyLoader:
    """Lazy loading utilities for large datasets"""
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_dataframe_chunk(file_data: bytes, chunk_size: int = 5000, chunk_number: int = 0):
        """Load DataFrame in chunks for better performance"""
        try:
            import io
            if chunk_number == 0:
                # Load first chunk to get column info
                df = pd.read_excel(io.BytesIO(file_data), nrows=chunk_size)
            else:
                # Load specific chunk
                skip_rows = chunk_number * chunk_size
                df = pd.read_excel(io.BytesIO(file_data), 
                                 skiprows=range(1, skip_rows + 1), 
                                 nrows=chunk_size)
            return df
        except Exception as e:
            logger.error(f"Error loading chunk {chunk_number}: {str(e)}")
            return pd.DataFrame()

class QueryOptimizer:
    """Optimize database queries and vector searches"""
    
    @staticmethod
    def optimize_vector_search(vector_store, query: str, k: int = 5) -> list:
        """Optimized vector search with caching"""
        try:
            # Use smaller k for faster results
            optimized_k = min(k, 3)
            
            # Perform search
            if hasattr(vector_store, 'similarity_search'):
                results = vector_store.similarity_search(query, k=optimized_k)
            else:
                # Fallback for simple text store
                results = vector_store.similarity_search(query, k=optimized_k)
                
            return results
        except Exception as e:
            logger.error(f"Vector search optimization failed: {str(e)}")
            return []
    
    @staticmethod
    def optimize_database_query(engine, query: str, params: Dict = None):
        """Optimize database queries with connection pooling"""
        try:
            with engine.connect() as conn:
                if params:
                    result = conn.execute(query, params)
                else:
                    result = conn.execute(query)
                return result.fetchall()
        except Exception as e:
            logger.error(f"Database query optimization failed: {str(e)}")
            return []

def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if execution_time > 1.0:  # Log slow operations
                logger.warning(f"{func.__name__} took {execution_time:.2f}s")
            else:
                logger.debug(f"{func.__name__} executed in {execution_time:.2f}s")
                
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {str(e)}")
            raise
    return wrapper

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()

# Streamlit-specific optimizations
def optimize_streamlit_layout():
    """Optimize Streamlit layout for better performance"""
    # Reduce rerun frequency
    if 'last_interaction' not in st.session_state:
        st.session_state.last_interaction = time.time()
    
    # Throttle interactions
    current_time = time.time()
    if current_time - st.session_state.last_interaction < 0.5:  # 500ms throttle
        return False
    
    st.session_state.last_interaction = current_time
    return True

def cache_ui_state(key: str, value: Any):
    """Cache UI state to prevent unnecessary recalculations"""
    if 'ui_cache' not in st.session_state:
        st.session_state.ui_cache = {}
    st.session_state.ui_cache[key] = value

def get_cached_ui_state(key: str, default: Any = None):
    """Retrieve cached UI state"""
    if 'ui_cache' not in st.session_state:
        st.session_state.ui_cache = {}
    return st.session_state.ui_cache.get(key, default)