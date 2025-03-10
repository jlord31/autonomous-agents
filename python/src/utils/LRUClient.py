from functools import lru_cache
import time
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

class TimedLRUCache:
    """LRU Cache with expiration times"""
    
    def __init__(self, maxsize=100):
        self.cache = {}
        self.expiry = {}
        self.maxsize = maxsize
        self.access_times = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value if it exists and isn't expired"""
        if key not in self.cache:
            return None
            
        # Check expiration
        expires_at = self.expiry.get(key, 0)
        if expires_at > 0 and time.time() > expires_at:
            self._remove(key)
            return None
            
        # Update access time
        self.access_times[key] = time.time()
        return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Store a value with expiration time"""
        # If cache is full, remove least recently used item
        if len(self.cache) >= self.maxsize and key not in self.cache:
            self._remove_lru()
            
        self.cache[key] = value
        self.access_times[key] = time.time()
        
        if ttl > 0:
            self.expiry[key] = time.time() + ttl
    
    def _remove(self, key: str) -> None:
        """Remove an item from all dictionaries"""
        if key in self.cache:
            del self.cache[key]
        if key in self.expiry:
            del self.expiry[key]
        if key in self.access_times:
            del self.access_times[key]
    
    def _remove_lru(self) -> None:
        """Remove the least recently used item"""
        if not self.access_times:
            return
            
        # Find key with oldest access time
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        self._remove(lru_key)
        logger.info(f"Removed LRU item: {lru_key}")
    
    def cleanup_expired(self) -> List[str]:
        """Remove all expired items"""
        now = time.time()
        expired = [k for k, v in self.expiry.items() if v > 0 and now > v]
        
        for key in expired:
            self._remove(key)
            
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired items")
        
        return expired

# Create singleton instance
cache_store = TimedLRUCache(maxsize=1000)