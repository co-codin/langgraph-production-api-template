import hashlib
import time
from typing import Optional


class ResponseCache:
    """
    In-memory response cache with TTL (time-to-live).

    In production, replace this with Redis for:
    - Persistence across restarts
    - Shared cache across multiple instances
    - Built-in TTL management
    """

    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._cache: dict[str, dict] = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str) -> str:
        """Normalize the query and hash it so keys are short and uniform."""
        normalized = " ".join(query.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, query: str) -> Optional[str]:
        """Return the cached response, or None on a miss or expired entry."""
        key = self._make_key(query)
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        if time.time() - entry["created_at"] > self.ttl:
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        return entry["response"]

    def set(self, query: str, response: str) -> None:
        """Store a response for this query."""
        key = self._make_key(query)
        self._cache[key] = {"response": response, "created_at": time.time()}

    def clear(self) -> None:
        """Remove every entry and reset counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "cached_entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{(self._hits / total) if total else 0.0:.2%}",
            "ttl_seconds": self.ttl,
        }
