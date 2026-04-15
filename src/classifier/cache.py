from typing import Optional

# Global cache instance
_cache_instance: Optional['ResponseCache'] = None

def get_cache():
    """Get the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        # Create cache with reasonable defaults
        cache_dir = __import__('pathlib').Path.home() / '.cache' / 'sentence-classifier'
        cache_dir.mkdir(parents=True, exist_ok=True)
        _cache_instance = __import__('classifier.cache').cache.ResponseCache(
            max_size=1000,
            ttl_seconds=3600,  # 1 hour
            cache_dir=cache_dir,
            enable_file_cache=True
        )
    return _cache_instance

def set_cache(cache):
    """Set the global cache instance."""
    global _cache_instance
    _cache_instance = cache