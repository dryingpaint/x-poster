# Parallel Development Guide

Guide for multiple engineers/agents working on this project in parallel without conflicts.

## Quick Start

1. **Read your module's README**: `src/<module>/README.md`
2. **Check dependencies**: Review dependency graph below
3. **Follow TDD**: Write tests FIRST, then implement
4. **Don't break interfaces**: Add optional fields, coordinate breaking changes

### Setup
```bash
uv sync
cp .env.example .env  # Fill in API keys
uv run pytest tests/<your_module>/ -v
```

## Test-First Development (TDD)

**🔴 RED → 🟢 GREEN → 🔵 REFACTOR**

1. **RED**: Write failing test in `tests/<module>/test_<feature>.py`
2. **GREEN**: Implement minimum code to pass in `src/<module>/`
3. **REFACTOR**: Clean up while keeping tests green

**Why?** Tests define contracts before implementation, enabling safe parallel work.

```bash
# For each feature
uv run pytest tests/<module>/test_<feature>.py -v  # RED: watch it fail
# Implement in src/<module>/
uv run pytest tests/<module>/test_<feature>.py -v  # GREEN: watch it pass
uv run pytest tests/<module>/ -v                   # REFACTOR: all still pass
```

## Module Overview

| Module | Purpose | Independence | README |
|--------|---------|--------------|--------|
| **core** | Config & models | ⭐⭐⭐ Foundation | [README](src/core/README.md) |
| **db** | Database ops | ⭐⭐ Low | [README](src/db/README.md) |
| **utils** | Utilities | ⭐⭐ Low | [README](src/utils/README.md) |
| **ingestion** | Document processing | ⭐⭐⭐ High | [README](src/ingestion/README.md) |
| **retrieval** | Search & reranking | ⭐⭐⭐ High | [README](src/retrieval/README.md) |
| **generation** | LLM operations | ⭐⭐⭐ High | [README](src/generation/README.md) |
| **orchestrator** | Pipeline coordination | ⭐ Very Low | [README](src/orchestrator/README.md) |

**Independence:** ⭐⭐⭐ High (parallel safe) | ⭐⭐ Low (coordinate changes) | ⭐ Very Low (coordinate all)

### Dependency Graph
```
Core (models, config)
  ↓
  ├─→ DB, Utils, Ingestion, Retrieval, Generation
  └─────────────────────────┬─────────────────────
                             ↓
                       Orchestrator
```

- **Core changes affect everyone**: Coordinate carefully
- **Middle layers independent**: Work in parallel
- **Orchestrator last**: After middle layers stable

## Coordination Rules

### 1. Interface Contracts Are Sacred
**Don't modify without discussion:**
- Function signatures in public APIs
- Data model fields (`src/core/models.py`)
- Database schema (requires migration)
- Config fields (`src/core/config.py`)

```python
# ❌ BAD: Breaking change
class SearchResult(BaseModel):
    new_required_field: str  # Breaks all code

# ✅ GOOD: Backward compatible
class SearchResult(BaseModel):
    new_optional_field: str | None = None
```

### 2. Add, Don't Modify
- Add new functions instead of changing existing ones
- Add optional parameters with defaults
- Deprecate gradually, don't delete immediately

### 3. Breaking Changes Protocol
1. Announce in team channel with affected modules
2. Wait for acknowledgment
3. Update all affected code in single PR
4. Update tests
5. Document in CHANGELOG

### 4. Test Isolation
- Test modules independently before integration
- Use mocks for dependencies
- Don't rely on external services in unit tests

### 5. Code Review Requirements
- **Core/DB schema**: 2 reviewers
- **Other modules**: 1 reviewer
- **Checklist**: Tests, docs, no breaking changes, performance assessed

## Testing Strategy

### Unit Tests (fast, isolated)
```bash
uv run pytest tests/<module>/ -v                      # Module
uv run pytest tests/<module>/test_file.py -v          # File
uv run pytest tests/<module>/test_file.py::test_fn -v # Function
```

### Integration Tests (slower, real dependencies)
```bash
uv run pytest tests/ -m integration -v
```

### E2E Tests (slowest, full system)
```bash
uv run pytest tests/orchestrator/test_pipeline.py::test_full_pipeline -v
```

### Coverage Targets
- **Core**: 95%+ | **DB**: 90%+ | **Other**: 80%+
```bash
uv run pytest --cov=src --cov-report=html tests/
```

## Git Workflow

### Branch Naming
- `feature/<module>-<description>`
- `bugfix/<module>-<description>`
- `refactor/<module>-<description>`

### Commit Messages
```
<module>: <short description>

- Detailed change 1
- Detailed change 2

Closes #123
```

### PR Checklist
- [ ] Tests added/updated
- [ ] All tests pass locally
- [ ] Documentation updated
- [ ] Code follows style guide (ruff, black, mypy)
- [ ] No breaking changes (or coordinated)
- [ ] Performance impact assessed

## Performance Budgets

Don't regress without discussion:

| Module | Operation | Budget | Current |
|--------|-----------|--------|---------|
| db | Vector search | <500ms | ~300ms |
| retrieval | Web search | <2s | ~1.5s |
| retrieval | Reranking (20 docs) | <500ms | ~400ms |
| generation | Evidence pack | <3s | ~2.5s |
| generation | Tweet generation | <5s | ~4s |
| orchestrator | Full pipeline | <15s | ~12s |

```bash
uv run pytest tests/<module>/test_performance.py -v
```

## Deployment

### Order
1. Core + Utils (foundation)
2. Database migrations (schema)
3. Middle layers (ingestion, retrieval, generation) - parallel
4. Orchestrator (integration)

### Feature Flags (for risky changes)
```python
# src/core/config.py
class Config(BaseModel):
    enable_new_feature: bool = False

# src/<module>/<file>.py
if config.enable_new_feature:
    return new_implementation()
else:
    return old_implementation()
```

**Deployment**: Deploy OFF → Test → Enable 10% → Monitor → 100% → Remove old code

## Common Issues

### Merge Conflicts
```bash
git fetch origin main
git rebase origin/main
git add . && git rebase --continue
```

### Test Failures After Merge
```bash
uv run pytest tests/ -v --tb=short  # Identify broken module
git log tests/<module>/              # Check recent changes
git revert <commit>                  # Revert if critical
```

### Interface Mismatches
```bash
uv run mypy src/                     # Type checking catches these
git log src/core/models.py           # Find interface changes
```

### Performance Regressions
```bash
python -m cProfile src/<module>/...  # Profile slow code
git checkout <previous-commit>       # Compare with baseline
```

## Development Tools

```bash
uv run mypy src/            # Type checking
uv run ruff check src/      # Linting
uv run black src/           # Formatting
uv run pytest tests/ -v     # Testing
uv run pytest --cov=src tests/  # Coverage
```

## Resources

- **Module READMEs**: `src/<module>/README.md`
- **Project Overview**: [CLAUDE.md](CLAUDE.md)
- **Quick Start**: [README.md](README.md)

## Questions?

- **Interface changes**: Check module README, ask in team channel
- **Testing strategy**: See module README
- **Performance impact**: Run benchmarks
- **Deployment**: Follow order above

**When in doubt, communicate early!**
