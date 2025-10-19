# Utils Module

## Overview
Shared utility functions used across multiple modules. Includes caching, text processing, and common helpers.

## Responsibilities
- **Caching**: Redis-based caching for expensive operations
- **Text Utilities**: Text normalization, tokenization, similarity
- **Common Helpers**: Retry logic, rate limiting, validation

## Key Files
- `cache.py`: Caching utilities (Redis)
- `text.py`: Text processing utilities

## Development Guidelines

### ⚠️ MANDATORY: Test-First Development

**YOU MUST WRITE TESTS BEFORE IMPLEMENTATION**

**TDD for Utilities:**
```bash
# 1. Write test FIRST (RED)
cat > tests/utils/test_cache_decorator.py << 'EOF'
from src.utils.cache import cached

@cached(ttl=60)
async def expensive_operation(x: int) -> int:
    return x * 2

@pytest.mark.asyncio
async def test_cached_decorator_caches_result():
    result1 = await expensive_operation(5)
    result2 = await expensive_operation(5)

    assert result1 == result2 == 10
    # Second call should be from cache (instant)
EOF

# 2. Run and watch fail (RED)
uv run pytest tests/utils/test_cache_decorator.py -v

# 3. Implement decorator (GREEN)
# Add @cached decorator to src/utils/cache.py
```

**Why Test-First for Utils?**
- **Reusability**: Well-tested utilities used everywhere
- **Edge cases**: Utilities handle many input types
- **Documentation**: Tests show usage examples

### Working on Caching (`cache.py`)
**What you can do in parallel:**
- Add cache backends (Redis, in-memory, DynamoDB)
- Add cache decorators for easy function caching
- Add cache invalidation strategies (TTL, LRU)
- Add cache monitoring and statistics

**What requires coordination:**
- Changing cache key format (breaks existing cache)
- Modifying cache TTL defaults (affects behavior)
- Changing serialization format (breaks compatibility)

**Testing requirements:**
```bash
# Test caching logic
uv run pytest tests/utils/test_cache.py

# Test with Redis
docker run -d -p 6379:6379 redis
export REDIS_URL="redis://localhost:6379"
uv run pytest tests/utils/test_cache.py::test_redis_cache
```

**Code style:**
```python
# GOOD: Generic cache interface
from abc import ABC, abstractmethod
from typing import Any

class Cache(ABC):
    """Base cache interface."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass

# GOOD: Redis implementation
import redis.asyncio as redis
import json

class RedisCache(Cache):
    """Redis-based cache."""

    def __init__(self, url: str):
        self.client = redis.from_url(url)

    async def get(self, key: str) -> Any | None:
        """Get value from Redis."""
        value = await self.client.get(key)
        if value is None:
            return None
        return json.loads(value)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in Redis with optional TTL."""
        serialized = json.dumps(value)
        if ttl:
            await self.client.setex(key, ttl, serialized)
        else:
            await self.client.set(key, serialized)

    async def delete(self, key: str) -> None:
        """Delete value from Redis."""
        await self.client.delete(key)

# GOOD: Cache decorator
from functools import wraps

def cached(ttl: int = 3600):
    """Decorator to cache function results.

    Args:
        ttl: Time to live in seconds

    Example:
        @cached(ttl=3600)
        async def expensive_operation(param: str) -> str:
            # ... expensive computation
            return result
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and args
            cache_key = f"{func.__name__}:{args}:{kwargs}"

            # Check cache
            cache = get_cache()
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Compute and cache
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl=ttl)
            return result

        return wrapper
    return decorator

# GOOD: Cache with namespace
class NamespacedCache:
    """Cache with automatic key namespacing."""

    def __init__(self, cache: Cache, namespace: str):
        self.cache = cache
        self.namespace = namespace

    def _make_key(self, key: str) -> str:
        """Prepend namespace to key."""
        return f"{self.namespace}:{key}"

    async def get(self, key: str) -> Any | None:
        return await self.cache.get(self._make_key(key))

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        await self.cache.set(self._make_key(key), value, ttl)
```

### Working on Text Utils (`text.py`)
**What you can do in parallel:**
- Add text normalization functions (whitespace, Unicode)
- Add tokenization utilities (word, sentence, token counting)
- Add similarity functions (Jaccard, edit distance)
- Add text quality checks (language detection, readability)

**What requires coordination:**
- Changing normalization behavior (affects search quality)
- Modifying tokenization (impacts chunking)
- Changing similarity thresholds (affects deduplication)

**Testing requirements:**
```bash
# Test text utilities
uv run pytest tests/utils/test_text.py

# Test with various languages
uv run pytest tests/utils/test_text.py::test_unicode_handling
```

**Code style:**
```python
# GOOD: Text normalization
import re
import unicodedata

def normalize_text(text: str) -> str:
    """Normalize text for processing.

    - Normalizes Unicode (NFKC)
    - Removes excess whitespace
    - Converts to lowercase

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)

    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Lowercase
    text = text.lower()

    return text

# GOOD: Token counting (for chunking)
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken.

    Args:
        text: Input text
        model: Model name for tokenizer

    Returns:
        Number of tokens
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# GOOD: Sentence splitting
import re

def split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    # Simple sentence splitting (handles common cases)
    # For production, consider using spaCy or NLTK
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# GOOD: Text similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def text_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score (0-1)
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return float(cosine_similarity(vectors[0], vectors[1])[0][0])

# GOOD: Text truncation
def truncate_text(text: str, max_chars: int = 280, suffix: str = "...") -> str:
    """Truncate text to maximum characters.

    Args:
        text: Input text
        max_chars: Maximum characters
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text

    # Truncate at word boundary
    truncated = text[:max_chars - len(suffix)]
    last_space = truncated.rfind(' ')
    if last_space > 0:
        truncated = truncated[:last_space]

    return truncated + suffix

# GOOD: Text quality checks
def is_quality_text(text: str, min_words: int = 10) -> bool:
    """Check if text meets quality threshold.

    Args:
        text: Input text
        min_words: Minimum word count

    Returns:
        True if text is quality
    """
    # Remove whitespace
    text = text.strip()

    # Check length
    words = text.split()
    if len(words) < min_words:
        return False

    # Check for excessive special characters
    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    if alpha_ratio < 0.5:
        return False

    return True
```

## Interface Contract

### Caching
```python
from src.utils.cache import get_cache, cached

# Direct cache usage
cache = get_cache()
await cache.set("key", "value", ttl=3600)
value = await cache.get("key")

# Decorator usage
@cached(ttl=3600)
async def expensive_function(param: str) -> str:
    # ... computation
    return result
```

### Text Processing
```python
from src.utils.text import (
    normalize_text,
    count_tokens,
    split_sentences,
    text_similarity,
    truncate_text
)

# Normalize
normalized = normalize_text("  Hello   World!  ")  # "hello world!"

# Count tokens
num_tokens = count_tokens("This is a test")  # 4

# Split sentences
sentences = split_sentences("First. Second! Third?")  # ["First.", "Second!", "Third?"]

# Similarity
score = text_similarity("hello world", "hello there")  # ~0.5

# Truncate
short = truncate_text("Very long text...", max_chars=20)
```

## Dependencies
**External:**
- `redis`: Redis client for caching
- `tiktoken`: Token counting
- `scikit-learn`: Text similarity

**Internal:**
- None (utils are foundation layer)

## Performance Considerations
- **Caching**: ~1ms for Redis get/set (network latency)
- **Token Counting**: ~1ms per 1000 tokens
- **Text Similarity**: ~10ms for 1000-word documents
- **Normalization**: ~0.1ms per 1000 characters

### Optimization Tips
```python
# GOOD: Batch cache operations
async def get_many(cache: Cache, keys: list[str]) -> dict[str, Any]:
    """Get multiple cache values in parallel."""
    results = await asyncio.gather(*[cache.get(key) for key in keys])
    return dict(zip(keys, results))

# GOOD: Cache tokenizer instances
from functools import lru_cache

@lru_cache(maxsize=10)
def get_tokenizer(model: str):
    """Get cached tokenizer."""
    return tiktoken.encoding_for_model(model)

# GOOD: Lazy initialization
class LazyCache:
    """Cache with lazy Redis connection."""

    def __init__(self, url: str):
        self.url = url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = redis.from_url(self.url)
        return self._client
```

## Common Pitfalls
- **DON'T** cache mutable objects without serialization
- **DON'T** use cache keys without namespacing (collisions)
- **DON'T** forget to handle cache misses
- **DON'T** cache forever (use appropriate TTLs)
- **DO** validate cached data before using
- **DO** handle cache backend failures gracefully
- **DO** monitor cache hit rates

## Testing Checklist
- [ ] Cache set/get/delete work correctly
- [ ] Cache TTL expires correctly
- [ ] Cache handles serialization of complex objects
- [ ] Text normalization is idempotent
- [ ] Token counting matches expected values
- [ ] Text similarity correlates with semantic similarity
- [ ] Truncation preserves word boundaries

## Local Development Setup
```bash
# Start Redis for cache testing
docker run -d -p 6379:6379 redis

# Set Redis URL
export REDIS_URL="redis://localhost:6379"

# Install dependencies
uv sync

# Run tests
uv run pytest tests/utils/ -v
```

## Usage Examples

### Caching Expensive Operations
```python
from src.utils.cache import cached
from src.generation.embeddings import embed_text

# Cache embeddings
@cached(ttl=86400)  # 24 hours
async def embed_text_cached(text: str) -> list[float]:
    """Generate embedding with caching."""
    return embed_text(text)

# Use cached version
embedding = await embed_text_cached("AI safety")  # Computed
embedding = await embed_text_cached("AI safety")  # Cached (instant)
```

### Text Processing Pipeline
```python
from src.utils.text import normalize_text, split_sentences, is_quality_text

def process_document(text: str) -> list[str]:
    """Process document into quality sentences."""
    # Normalize
    text = normalize_text(text)

    # Split into sentences
    sentences = split_sentences(text)

    # Filter quality sentences
    quality_sentences = [
        sent for sent in sentences
        if is_quality_text(sent, min_words=5)
    ]

    return quality_sentences
```

### Deduplication with Similarity
```python
from src.utils.text import text_similarity

def deduplicate_texts(texts: list[str], threshold: float = 0.85) -> list[str]:
    """Remove near-duplicate texts."""
    unique = []

    for text in texts:
        is_duplicate = False
        for existing in unique:
            if text_similarity(text, existing) > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(text)

    return unique
```

## Contact for Coordination
When modifying utilities:
1. Ensure backward compatibility (many modules depend on utils)
2. Document breaking changes clearly
3. Update all dependent modules simultaneously
4. Test with all modules that use the utility
