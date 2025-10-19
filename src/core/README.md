# Core Module

## Overview
Foundation module containing shared configuration, data models, and core types used across the entire system.

## Responsibilities
- **Configuration Management**: Centralized settings via pydantic-settings
- **Data Models**: Pydantic models for type safety and validation
- **Shared Types**: Enums and common interfaces

## Key Files
- `config.py`: Application configuration with environment variable loading
- `models.py`: All data models and types used throughout the system

## Development Guidelines

### Working on Configuration (`config.py`)
**What you can do in parallel:**
- Add new configuration sections (e.g., `CacheConfig`, `MonitoringConfig`)
- Add new environment variables with defaults
- Add validation logic for configuration values

**What requires coordination:**
- Renaming existing config fields (impacts all modules)
- Changing default values (discuss impact on existing deployments)
- Modifying the `get_config()` singleton pattern

**Testing requirements:**
```bash
# Test configuration loading
uv run pytest tests/core/test_config.py

# Verify environment variable parsing
uv run pytest tests/core/test_config.py::test_env_var_loading
```

**Code style:**
```python
# GOOD: Use pydantic Field with description
class NewConfig(BaseModel):
    timeout: int = Field(default=30, description="Request timeout in seconds")

# GOOD: Use validator for complex logic
@validator("timeout")
def validate_timeout(cls, v):
    if v < 1 or v > 300:
        raise ValueError("timeout must be between 1 and 300")
    return v
```

### Working on Data Models (`models.py`)
**What you can do in parallel:**
- Add new models (e.g., `CacheEntry`, `MetricsSnapshot`)
- Add optional fields to existing models (use `field_name: type | None = None`)
- Add new enums for extensibility

**What requires coordination:**
- Modifying required fields on existing models (breaks compatibility)
- Renaming fields (requires migration across all modules)
- Changing field types (impacts serialization/deserialization)

**Testing requirements:**
```bash
# Test model validation
uv run pytest tests/core/test_models.py

# Test model serialization
uv run pytest tests/core/test_models.py::test_model_serialization
```

**Code style:**
```python
# GOOD: Use descriptive field names and types
class SearchResult(BaseModel):
    source_id: str  # Unique identifier
    content: str
    score: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)

# GOOD: Add validators for business logic
@validator("score")
def validate_score(cls, v):
    if v < 0:
        raise ValueError("score must be non-negative")
    return v

# GOOD: Use enums for constrained choices
class Status(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
```

## Interface Contract

### Configuration Access Pattern
```python
from src.core.config import get_config

config = get_config()  # Singleton - always returns same instance
# Access nested configs
db_url = config.supabase.database_url
api_key = config.openai.api_key
```

### Model Usage Pattern
```python
from src.core.models import SearchResult, EvidenceFact

# Create and validate
result = SearchResult(
    source_id="doc_123",
    content="AI systems require alignment...",
    score=0.95,
    source_type="internal"
)

# Serialize/deserialize
json_str = result.model_dump_json()
restored = SearchResult.model_validate_json(json_str)
```

## Dependencies
**External:**
- `pydantic`: Model validation and settings management
- `pydantic-settings`: Environment variable loading

**Internal:**
- None (this is the foundation module)

## Performance Considerations
- `get_config()` is cached - call freely without performance concerns
- Pydantic validation adds ~microseconds per model creation
- Use `model_construct()` for trusted data to bypass validation

## Common Pitfalls
- **DON'T** modify config after initialization (not thread-safe)
- **DON'T** add mutable defaults without `Field(default_factory=...)`
- **DON'T** use relative imports (use `from src.core.models import ...`)
- **DO** add type hints to all fields
- **DO** use `| None` for optional fields instead of `Optional[]`
- **DO** document complex fields with Field(description="...")

## Testing Checklist
- [ ] Configuration loads from environment variables correctly
- [ ] Configuration validates invalid values (negative numbers, empty strings, etc.)
- [ ] Models serialize/deserialize without data loss
- [ ] Model validators catch invalid data
- [ ] Enums cover all expected cases
- [ ] Default values are sensible for all configs

## Contact for Coordination
When making breaking changes to core models or configuration:
1. Review impact across all modules (`src/*/`)
2. Check usages: `rg "YourModel" src/`
3. Create migration plan for database models
4. Update CLAUDE.md if changing architectural patterns
