import os
import json
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class CacheManager:
    """Advanced caching system with multiple storage backends and TTL support"""
    
    def __init__(self, cache_dir: str = ".cache", max_memory_items: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory cache for frequently accessed items
        self.memory_cache = {}
        self.memory_access_times = {}
        self.max_memory_items = max_memory_items
        
        # Thread lock for thread safety
        self.lock = threading.RLock()
        
        # Database for persistent cache
        self.db_path = self.cache_dir / "cache.db"
        self._init_database()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'expired_items': 0
        }
        
        # Cleanup settings
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(hours=1)
        
        logger.info(f"Cache manager initialized with directory: {self.cache_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for persistent caching"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_items (
                        key TEXT PRIMARY KEY,
                        value BLOB,
                        created_at TIMESTAMP,
                        expires_at TIMESTAMP,
                        access_count INTEGER DEFAULT 0,
                        last_accessed TIMESTAMP
                    )
                """)
                
                # Create index for faster lookups
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_expires_at 
                    ON cache_items(expires_at)
                """)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache"""
        with self.lock:
            try:
                # Check memory cache first
                if key in self.memory_cache:
                    item = self.memory_cache[key]
                    
                    # Check if expired
                    if item['expires_at'] and datetime.now() > item['expires_at']:
                        del self.memory_cache[key]
                        if key in self.memory_access_times:
                            del self.memory_access_times[key]
                        self.stats['expired_items'] += 1
                    else:
                        # Update access time
                        self.memory_access_times[key] = datetime.now()
                        self.stats['hits'] += 1
                        self.stats['memory_hits'] += 1
                        return item['value']
                
                # Check persistent cache
                disk_item = self._get_from_disk(key)
                if disk_item:
                    # Move to memory cache if space available
                    if len(self.memory_cache) < self.max_memory_items:
                        self.memory_cache[key] = disk_item
                        self.memory_access_times[key] = datetime.now()
                    
                    self.stats['hits'] += 1
                    self.stats['disk_hits'] += 1
                    return disk_item['value']
                
                self.stats['misses'] += 1
                return None
                
            except Exception as e:
                logger.error(f"Cache get failed for key '{key}': {str(e)}")
                self.stats['misses'] += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store item in cache with optional TTL in seconds"""
        with self.lock:
            try:
                expires_at = None
                if ttl:
                    expires_at = datetime.now() + timedelta(seconds=ttl)
                
                item = {
                    'value': value,
                    'created_at': datetime.now(),
                    'expires_at': expires_at
                }
                
                # Store in memory cache if space available
                if len(self.memory_cache) < self.max_memory_items:
                    self.memory_cache[key] = item
                    self.memory_access_times[key] = datetime.now()
                else:
                    # Evict least recently used item
                    self._evict_lru_memory()
                    self.memory_cache[key] = item
                    self.memory_access_times[key] = datetime.now()
                
                # Store in persistent cache
                self._set_to_disk(key, item)
                
                self.stats['sets'] += 1
                
                # Periodic cleanup
                if datetime.now() - self.last_cleanup > self.cleanup_interval:
                    self._cleanup_expired()
                
                return True
                
            except Exception as e:
                logger.error(f"Cache set failed for key '{key}': {str(e)}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        with self.lock:
            try:
                deleted = False
                
                # Remove from memory cache
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    if key in self.memory_access_times:
                        del self.memory_access_times[key]
                    deleted = True
                
                # Remove from persistent cache
                if self._delete_from_disk(key):
                    deleted = True
                
                if deleted:
                    self.stats['deletes'] += 1
                
                return deleted
                
            except Exception as e:
                logger.error(f"Cache delete failed for key '{key}': {str(e)}")
                return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        return self.get(key) is not None
    
    def clear(self) -> bool:
        """Clear all cache items"""
        with self.lock:
            try:
                # Clear memory cache
                self.memory_cache.clear()
                self.memory_access_times.clear()
                
                # Clear persistent cache
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute("DELETE FROM cache_items")
                    conn.commit()
                
                logger.info("All cache items cleared")
                return True
                
            except Exception as e:
                logger.error(f"Cache clear failed: {str(e)}")
                return False
    
    def _get_from_disk(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve item from persistent cache"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    "SELECT value, created_at, expires_at FROM cache_items WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                value_blob, created_at_str, expires_at_str = row
                
                # Parse timestamps
                created_at = datetime.fromisoformat(created_at_str)
                expires_at = None
                if expires_at_str:
                    expires_at = datetime.fromisoformat(expires_at_str)
                    
                    # Check if expired
                    if datetime.now() > expires_at:
                        self._delete_from_disk(key)
                        self.stats['expired_items'] += 1
                        return None
                
                # Deserialize value
                value = pickle.loads(value_blob)
                
                # Update access statistics
                conn.execute(
                    "UPDATE cache_items SET access_count = access_count + 1, last_accessed = ? WHERE key = ?",
                    (datetime.now().isoformat(), key)
                )
                conn.commit()
                
                return {
                    'value': value,
                    'created_at': created_at,
                    'expires_at': expires_at
                }
                
        except Exception as e:
            logger.error(f"Disk cache get failed for key '{key}': {str(e)}")
            return None
    
    def _set_to_disk(self, key: str, item: Dict[str, Any]) -> bool:
        """Store item in persistent cache"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Serialize value
                value_blob = pickle.dumps(item['value'])
                
                # Prepare timestamps
                created_at_str = item['created_at'].isoformat()
                expires_at_str = item['expires_at'].isoformat() if item['expires_at'] else None
                
                # Insert or replace
                conn.execute(
                    """INSERT OR REPLACE INTO cache_items 
                       (key, value, created_at, expires_at, last_accessed) 
                       VALUES (?, ?, ?, ?, ?)""",
                    (key, value_blob, created_at_str, expires_at_str, datetime.now().isoformat())
                )
                conn.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"Disk cache set failed for key '{key}': {str(e)}")
            return False
    
    def _delete_from_disk(self, key: str) -> bool:
        """Delete item from persistent cache"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("DELETE FROM cache_items WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Disk cache delete failed for key '{key}': {str(e)}")
            return False
    
    def _evict_lru_memory(self):
        """Evict least recently used item from memory cache"""
        if not self.memory_access_times:
            return
        
        # Find least recently used key
        lru_key = min(self.memory_access_times, key=self.memory_access_times.get)
        
        # Remove from memory cache
        if lru_key in self.memory_cache:
            del self.memory_cache[lru_key]
        del self.memory_access_times[lru_key]
    
    def _cleanup_expired(self):
        """Clean up expired items from both memory and disk cache"""
        try:
            current_time = datetime.now()
            
            # Clean memory cache
            expired_keys = []
            for key, item in self.memory_cache.items():
                if item['expires_at'] and current_time > item['expires_at']:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
                if key in self.memory_access_times:
                    del self.memory_access_times[key]
                self.stats['expired_items'] += 1
            
            # Clean disk cache
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    "DELETE FROM cache_items WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (current_time.isoformat(),)
                )
                conn.commit()
                self.stats['expired_items'] += cursor.rowcount
            
            self.last_cleanup = current_time
            
            if expired_keys or cursor.rowcount > 0:
                logger.info(f"Cleaned up {len(expired_keys)} memory items and {cursor.rowcount} disk items")
                
        except Exception as e:
            logger.error(f"Cache cleanup failed: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        with self.lock:
            stats = self.stats.copy()
            
            # Add calculated metrics
            total_requests = stats['hits'] + stats['misses']
            if total_requests > 0:
                stats['hit_rate'] = (stats['hits'] / total_requests) * 100
                stats['memory_hit_rate'] = (stats['memory_hits'] / total_requests) * 100
                stats['disk_hit_rate'] = (stats['disk_hits'] / total_requests) * 100
            else:
                stats['hit_rate'] = 0
                stats['memory_hit_rate'] = 0
                stats['disk_hit_rate'] = 0
            
            # Add cache sizes
            stats['memory_cache_size'] = len(self.memory_cache)
            stats['memory_cache_limit'] = self.max_memory_items
            
            # Get disk cache size
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM cache_items")
                    stats['disk_cache_size'] = cursor.fetchone()[0]
            except Exception:
                stats['disk_cache_size'] = 0
            
            return stats
    
    def get_cache_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a cached item"""
        try:
            # Check memory first
            if key in self.memory_cache:
                item = self.memory_cache[key]
                return {
                    'location': 'memory',
                    'created_at': item['created_at'],
                    'expires_at': item['expires_at'],
                    'last_accessed': self.memory_access_times.get(key),
                    'size_bytes': len(pickle.dumps(item['value']))
                }
            
            # Check disk
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    "SELECT created_at, expires_at, access_count, last_accessed, LENGTH(value) FROM cache_items WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    created_at_str, expires_at_str, access_count, last_accessed_str, size_bytes = row
                    return {
                        'location': 'disk',
                        'created_at': datetime.fromisoformat(created_at_str),
                        'expires_at': datetime.fromisoformat(expires_at_str) if expires_at_str else None,
                        'access_count': access_count,
                        'last_accessed': datetime.fromisoformat(last_accessed_str) if last_accessed_str else None,
                        'size_bytes': size_bytes
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Cache info retrieval failed for key '{key}': {str(e)}")
            return None
    
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List all cache keys, optionally filtered by pattern"""
        try:
            keys = set()
            
            # Add memory cache keys
            keys.update(self.memory_cache.keys())
            
            # Add disk cache keys
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT key FROM cache_items")
                keys.update(row[0] for row in cursor.fetchall())
            
            key_list = list(keys)
            
            # Apply pattern filter if provided
            if pattern:
                import fnmatch
                key_list = [key for key in key_list if fnmatch.fnmatch(key, pattern)]
            
            return sorted(key_list)
            
        except Exception as e:
            logger.error(f"Cache key listing failed: {str(e)}")
            return []
    
    def export_cache(self, filepath: str) -> bool:
        """Export cache contents to file"""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'stats': self.get_stats(),
                'items': {}
            }
            
            # Export memory cache
            for key, item in self.memory_cache.items():
                export_data['items'][key] = {
                    'value': item['value'],
                    'created_at': item['created_at'].isoformat(),
                    'expires_at': item['expires_at'].isoformat() if item['expires_at'] else None,
                    'location': 'memory'
                }
            
            # Export disk cache
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT key, value, created_at, expires_at FROM cache_items")
                for key, value_blob, created_at_str, expires_at_str in cursor.fetchall():
                    if key not in export_data['items']:  # Don't overwrite memory items
                        try:
                            value = pickle.loads(value_blob)
                            export_data['items'][key] = {
                                'value': value,
                                'created_at': created_at_str,
                                'expires_at': expires_at_str,
                                'location': 'disk'
                            }
                        except Exception:
                            continue  # Skip items that can't be deserialized
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Cache exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Cache export failed: {str(e)}")
            return False

