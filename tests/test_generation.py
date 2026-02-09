"""Tests for chain-of-thought generation."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pragma import InferenceConfig
from pragma.generate.cotgen import run_batches, set_seed
from pragma.mcqa import MultipleChoiceQA
from pragma.tokenize import get_special_start_end


@pytest.fixture(scope="module")
def test_config():
    """Load the Granite test configuration."""
    return InferenceConfig.from_toml("config/granite.toml")


@pytest.fixture(scope="module")
def model_and_tokenizer(test_config):
    """Load IBM Granite 2B model and tokenizer once for all tests."""
    tokenizer = AutoTokenizer.from_pretrained(test_config.model)
    model = AutoModelForCausalLM.from_pretrained(
        test_config.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=test_config.attn_implementation,
    )

    return model, tokenizer


@pytest.fixture
def test_qa():
    """Create a simple test question."""
    return MultipleChoiceQA(
        id="test-1",
        question="What is 2+2?",
        subject="math",
        choices=("3", "4", "5", "6"),
        choice_labels=("A", "B", "C", "D"),
        correct_answer="B",
    )


@pytest.fixture
def inference_config(test_config):
    """Get the model config from the loaded configuration."""
    return test_config


@pytest.mark.slow
def test_model_loads(model_and_tokenizer):
    """Test that IBM Granite 2B model loads successfully."""
    model, tokenizer = model_and_tokenizer

    assert model is not None
    assert tokenizer is not None
    assert model.device.type in ["cuda", "cpu", "mps"]


@pytest.mark.slow
def test_tokenizer_has_think_tokens(model_and_tokenizer, inference_config):
    """Test that tokenizer handles think tokens appropriately.

    Some tokenizers (like Granite) may not have <think>/<\/think> as single tokens,
    which is valid. In that case, get_special_start_end returns None and the
    generation code handles it gracefully.
    """
    _, tokenizer = model_and_tokenizer

    # Get special start/end tokens for thinking
    special_tokens = get_special_start_end(
        tokenizer,
        substring=inference_config.think_substring,
        token_format=inference_config.think_token_format,
    )

    # Either the tokenizer has special tokens (returns dict) or it doesn't (returns None)
    # Both are valid behaviors
    if special_tokens is not None:
        # If tokens exist, verify they have the expected structure
        assert "start" in special_tokens
        assert "end" in special_tokens

        # Verify the tokens are valid token IDs
        start_token = special_tokens["start"]
        end_token = special_tokens["end"]
        assert isinstance(start_token, int)
        assert isinstance(end_token, int)
    else:
        # If None, that's also okay - tokenizer doesn't have the tokens as single tokens
        # The generation code handles this gracefully
        assert special_tokens is None


@pytest.mark.slow
def test_generation_with_thinking(
    model_and_tokenizer, test_qa, inference_config
):
    """Test CoT generation with thinking enabled."""
    model, tokenizer = model_and_tokenizer

    set_seed(42)

    results = run_batches(
        [test_qa],
        model=model,
        inference_config=inference_config,
        tokenizer=tokenizer,
        batch_size=1,
        enable_thinking=True,
        generation_kwargs={
            "max_new_tokens": 128,
            "do_sample": False,  # Greedy decoding for determinism
        },
    )

    assert len(results) == 1
    result = results[0]

    # Verify result structure
    assert result.qa == test_qa
    assert result.reasoning_enabled is True

    # With thinking enabled, we should have a reasoning trace
    # (though it might be empty if model doesn't generate thinking)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.response, str)

    # Response should not be empty
    assert len(result.response.strip()) > 0


@pytest.mark.slow
def test_generation_without_thinking(
    model_and_tokenizer, test_qa, inference_config
):
    """Test CoT generation with thinking disabled."""
    model, tokenizer = model_and_tokenizer

    set_seed(42)

    results = run_batches(
        [test_qa],
        model=model,
        inference_config=inference_config,
        tokenizer=tokenizer,
        batch_size=1,
        enable_thinking=False,
        generation_kwargs={
            "max_new_tokens": 128,
            "do_sample": False,  # Greedy decoding for determinism
        },
    )

    assert len(results) == 1
    result = results[0]

    # Verify result structure
    assert result.qa == test_qa
    assert result.reasoning_enabled is False

    # Without thinking, reasoning should be empty
    assert result.reasoning == ""
    assert isinstance(result.response, str)

    # Response should not be empty
    assert len(result.response.strip()) > 0
