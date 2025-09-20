#!/usr/bin/env python3
"""
Caching module for the Human/Machine Sentence Classifier.

This module provides intelligent caching for API responses to reduce costs and improve performance.
It uses both in-memory and optional file-based caching with TTL (Time To Live) support.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict

try:
    from cachetools import TTLCache
    from cachetools.keys import hashkey
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    ttl: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    @property
    def age_seconds(self) -> float:
        """Get the age of the cache entry in seconds."""
        return time.time() - self.created_at

class ResponseCache:
    """Intelligent caching system for API responses."""

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,  # 1 hour default
        cache_dir: Optional[Union[str, Path]] = None,
        enable_file_cache: bool = True
    ):
        """
        Initialize the cache system.

        Args:
            max_size: Maximum number of items to keep in memory cache
            ttl_seconds: Time to live for cache entries in seconds
            cache_dir: Directory for file-based caching (optional)
            enable_file_cache: Whether to enable file-based caching
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_file_cache = enable_file_cache

        # In-memory cache
        if CACHETOOLS_AVAILABLE:
            self.memory_cache = TTLCache(maxsize=max_size, ttl=ttl_seconds)
        else:
            # Fallback to simple dict if cachetools not available
            self.memory_cache = {}
            self._cleanup_old_entries()

        # File-based cache
        if enable_file_cache and cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_size = 0

    def _generate_key(self, text: str, provider: str, model: str) -> str:
        """Generate a cache key from the input parameters."""
        # Create a hash of the input to use as cache key
        key_data = f"{text.strip()}:{provider}:{model}"
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()

    def _get_cache_file_path(self, key: str) -> Optional[Path]:
        """Get the file path for a cache key."""
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{key}.json"

    def _load_from_file(self, key: str) -> Optional[CacheEntry]:
        """Load a cache entry from file."""
        if not self.cache_dir:
            return None

        cache_file = self._get_cache_file_path(key)
        if not cache_file or not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            entry = CacheEntry(
                key=data['key'],
                value=data['value'],
                created_at=data['created_at'],
                ttl=data.get('ttl'),
                metadata=data.get('metadata')
            )

            if entry.is_expired:
                # Remove expired file
                cache_file.unlink(missing_ok=True)
                return None

            return entry

        except (IOError, json.JSONDecodeError, KeyError) as e:
            logging.debug(f"Failed to load cache file {cache_file}: {e}")
            return None

    def _save_to_file(self, entry: CacheEntry) -> bool:
        """Save a cache entry to file."""
        if not self.cache_dir:
            return False

        cache_file = self._get_cache_file_path(entry.key)
        if not cache_file:
            return False

        try:
            data = asdict(entry)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except (IOError, OSError) as e:
            logging.debug(f"Failed to save cache file {cache_file}: {e}")
            return False

    def _cleanup_old_entries(self):
        """Clean up old entries from simple dict cache."""
        if CACHETOOLS_AVAILABLE:
            return  # TTLCache handles this automatically

        current_time = time.time()
        keys_to_remove = []

        for key, entry in self.memory_cache.items():
            if isinstance(entry, CacheEntry) and entry.is_expired:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.memory_cache[key]

    def get(self, text: str, provider: str, model: str) -> Optional[Any]:
        """
        Get a cached response for the given parameters.

        Args:
            text: The text being classified
            provider: The AI provider ('gemini' or 'openrouter')
            model: The model name

        Returns:
            Cached response or None if not found/expired
        """
        key = self._generate_key(text, provider, model)

        # Try memory cache first
        if CACHETOOLS_AVAILABLE:
            cached = self.memory_cache.get(key)
        else:
            cached = self.memory_cache.get(key)
            if cached and cached.is_expired:
                del self.memory_cache[key]
                cached = None

        if cached:
            self.cache_hits += 1
            logging.debug(f"Cache hit for key {key}")
            return cached.value

        # Try file cache
        file_entry = self._load_from_file(key)
        if file_entry:
            # Store in memory cache for faster future access
            if CACHETOOLS_AVAILABLE:
                self.memory_cache[key] = file_entry
            else:
                self.memory_cache[key] = file_entry

            self.cache_hits += 1
            logging.debug(f"File cache hit for key {key}")
            return file_entry.value

        self.cache_misses += 1
        logging.debug(f"Cache miss for key {key}")
        return None

    def put(self, text: str, provider: str, model: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Cache a response for the given parameters.

        Args:
            text: The text being classified
            provider: The AI provider ('gemini' or 'openrouter')
            model: The model name
            value: The response value to cache
            metadata: Optional metadata to store with the cache entry

        Returns:
            True if successfully cached, False otherwise
        """
        key = self._generate_key(text, provider, model)

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl=self.ttl_seconds,
            metadata=metadata
        )

        # Store in memory cache
        if CACHETOOLS_AVAILABLE:
            self.memory_cache[key] = entry
        else:
            self.memory_cache[key] = entry
            self._cleanup_old_entries()

        # Store in file cache if enabled
        if self.enable_file_cache:
            self._save_to_file(entry)

        self.cache_size = len(self.memory_cache)
        logging.debug(f"Cached response for key {key}")
        return True

    def clear(self) -> int:
        """Clear all cache entries and return the number of entries cleared."""
        memory_count = len(self.memory_cache)

        # Clear memory cache
        if CACHETOOLS_AVAILABLE:
            self.memory_cache.clear()
        else:
            self.memory_cache.clear()

        # Clear file cache
        file_count = 0
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                    file_count += 1
                except OSError:
                    pass

        total_cleared = memory_count + file_count
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_size = 0

        logging.info(f"Cleared {total_cleared} cache entries ({memory_count} memory, {file_count} file)")
        return total_cleared

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': hit_rate,
            'cache_size': self.cache_size,
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'file_cache_enabled': self.enable_file_cache,
            'cache_dir': str(self.cache_dir) if self.cache_dir else None
        }

# Global cache instance
_cache_instance: Optional[ResponseCache] = None

def get_cache() -> ResponseCache:
    """Get the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        # Create cache with reasonable defaults
        cache_dir = Path.home() / '.cache' / 'sentence-classifier'
        _cache_instance = ResponseCache(
            max_size=1000,
            ttl_seconds=3600,  # 1 hour
            cache_dir=cache_dir,
            enable_file_cache=True
        )
    return _cache_instance

def set_cache(cache: ResponseCache) -> None:
    """Set the global cache instance."""
    global _cache_instance
    _cache_instance = cache

if __name__ == "__main__":
    # Test the cache functionality
    cache = get_cache()

    # Test caching
    test_text = "This is a test sentence."
    test_provider = "gemini"
    test_model = "gemini-2.5-pro"
    test_value = "1"

    # Should be a miss
    result = cache.get(test_text, test_provider, test_model)
    print(f"First get: {result}")  # Should be None

    # Cache the value
    cache.put(test_text, test_provider, test_model, test_value)

    # Should be a hit
    result = cache.get(test_text, test_provider, test_model)
    print(f"Second get: {result}")  # Should be "1"

    # Print stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")

    # Clear cache
    cleared = cache.clear()
    print(f"Cleared {cleared} entries")