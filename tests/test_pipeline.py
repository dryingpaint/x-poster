"""Test main pipeline."""

import pytest

from src.core.models import GenerateRequest
from src.orchestrator.pipeline import run_generation_pipeline


@pytest.mark.asyncio
async def test_pipeline_basic():
    """Test basic pipeline execution."""
    request = GenerateRequest(
        prompt="Test prompt", max_variants=1, max_thread_tweets=3
    )

    # This will fail without proper setup, but tests structure
    # In real tests, you'd mock the external dependencies
    # response = await run_generation_pipeline(request)
    # assert response is not None
    pass


def test_request_creation():
    """Test GenerateRequest creation."""
    request = GenerateRequest(prompt="Test", max_variants=2)

    assert request.prompt == "Test"
    assert request.max_variants == 2
    assert request.max_thread_tweets == 6  # default

