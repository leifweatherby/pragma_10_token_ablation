# LLM Judge Setup Guide

## Overview

The LLM judge provides more accurate evaluation than regex-based extraction by using a language model to understand and evaluate responses. This is especially important for:
- Complex reasoning tasks
- Open-ended responses
- Cases where answer format varies
- Better handling of edge cases

---

## Quick Start

### 1. Set up API Key

**For OpenAI (recommended for speed/cost):**
```bash
export OPENAI_API_KEY="sk-..."
```

**For Anthropic:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**For local models:**
No API key needed if running locally.

### 2. Update Config File

Edit your experiment config (e.g., `config/qwen3-0.6B-arc-easy.toml`):

```toml
[ablations]
# Change from:
judge = "simple-regex"
base_url = ""
api_key = "NONE"

# To (for OpenAI):
judge = "gpt-4o-mini"  # Fast and cheap
base_url = "https://api.openai.com/v1"
api_key = "OPENAI_API_KEY"

# Or (for Claude):
judge = "claude-3-5-haiku-20241022"  # Fast and accurate
base_url = "https://api.anthropic.com/v1"
api_key = "ANTHROPIC_API_KEY"
```

### 3. Run Experiment

```bash
pixi run python -m pragma config/your-config.toml --runtype ablations
```

---

## Supported Models

### OpenAI Models
- **gpt-4o-mini** (recommended) - Fast, cheap, accurate
- **gpt-4o** - More capable but slower/expensive
- **gpt-3.5-turbo** - Fastest but less reliable

### Anthropic Models
- **claude-3-5-haiku-20241022** (recommended) - Fast and accurate
- **claude-3-5-sonnet-20241022** - Most capable

### Local Models
Any OpenAI-compatible endpoint:
```toml
judge = "llama-3-8b"  # Model name doesn't matter for local
base_url = "http://localhost:8000/v1"
api_key = "NONE"  # Or whatever your local server requires
```

---

## Configuration Examples

### Example 1: OpenAI GPT-4o-mini (Recommended)

```toml
[ablations]
judge = "gpt-4o-mini"
base_url = "https://api.openai.com/v1"
api_key = "OPENAI_API_KEY"
metric = "fighting_words"
ngram_range = [1, 1]
strategy = "mask"
penalty_weight = 1.0
cutoff = 0.98
top_k = 10
max_steps = 20
```

**Cost estimate:** ~$0.01-0.02 per 100 questions (very cheap)

### Example 2: Claude Haiku (Most Accurate)

```toml
[ablations]
judge = "claude-3-5-haiku-20241022"
base_url = "https://api.anthropic.com/v1"
api_key = "ANTHROPIC_API_KEY"
metric = "fighting_words"
ngram_range = [1, 1]
strategy = "mask"
penalty_weight = 1.0
cutoff = 0.98
top_k = 10
max_steps = 20
```

**Cost estimate:** ~$0.05-0.10 per 100 questions

### Example 3: Local Model (Free)

```toml
[ablations]
judge = "local-model"
base_url = "http://localhost:8000/v1"
api_key = "NONE"  # No API key needed
metric = "fighting_words"
ngram_range = [1, 1]
strategy = "mask"
penalty_weight = 1.0
cutoff = 0.98
top_k = 10
max_steps = 20
```

**Setup local server:**
```bash
# Using vLLM
vllm serve meta-llama/Llama-3-8B-Instruct --port 8000

# Using Ollama
ollama serve
```

---

## How the LLM Judge Works

The LLM judge evaluates responses using structured output with tool calling:

1. **Receives:** Question, answer choices, model's response
2. **Analyzes:** Uses reasoning to understand which answer the model chose
3. **Returns:** Structured verdict with extracted answer and correctness

**Advantages over regex:**
- Understands context and reasoning
- Handles varied response formats
- Can extract answers from complex explanations
- More robust to edge cases

**Example evaluation:**

```python
# Input
Question: "What is photosynthesis?"
Choices: A) Respiration B) Plant food production C) Cell division
Response: "I think the answer is B, because plants use sunlight..."

# Regex judge: Might extract "B" ✓
# LLM judge: Understands "plants use sunlight to make food" → B ✓✓
```

---

## Cost Optimization

### Use GPT-4o-mini for Most Tasks
- **Cost:** $0.150 per 1M input tokens, $0.600 per 1M output tokens
- **Speed:** ~100-200ms per evaluation
- **Accuracy:** >95% for multiple choice

### Batch Processing
The judge automatically batches requests to minimize API calls:
```python
# Configured in judge.py
max_calls_per_minute = 50  # Rate limiting
```

### Token Usage Estimates

Per evaluation (~300 tokens total):
- System prompt: ~100 tokens
- Question + choices: ~100 tokens
- Model response: ~50 tokens
- Tool call: ~50 tokens

**For 100 questions:** ~30,000 tokens = **$0.005** with GPT-4o-mini

---

## Troubleshooting

### Error: "No API key set"
```bash
# Make sure you've exported the key
export OPENAI_API_KEY="sk-..."

# Verify it's set
echo $OPENAI_API_KEY
```

### Error: "Rate limit exceeded"
Reduce `max_calls_per_minute` in the code or wait and retry.

### Error: "Invalid model"
Check that the model name matches the API provider:
- OpenAI: `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`
- Anthropic: `claude-3-5-haiku-20241022`, `claude-3-5-sonnet-20241022`

### Slow evaluation
- Switch to `gpt-4o-mini` for speed
- Check your internet connection
- Consider local model for offline use

---

## Migration from Simple Judge

To convert existing configs:

**Before (simple-regex):**
```toml
[ablations]
judge = "simple-regex"
base_url = ""
api_key = "NONE"
```

**After (LLM judge):**
```toml
[ablations]
judge = "gpt-4o-mini"
base_url = "https://api.openai.com/v1"
api_key = "OPENAI_API_KEY"
```

Then set the environment variable:
```bash
export OPENAI_API_KEY="your-key-here"
```

---

## Pre-configured Templates

We've created template configs for common setups:

1. **`config/qwen3-0.6B-arc-easy-llm-judge.toml`** - OpenAI GPT-4o-mini
2. Copy and modify for your experiments

---

## Testing Your Setup

Quick test:
```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Run small test
pixi run python -m pragma config/qwen3-0.6B-arc-easy-llm-judge.toml --runtype ablations
```

You should see:
```
INFO - Loading Qwen/Qwen3-0.6B
INFO - Using 100 examples from allenai/ai2_arc
INFO - Generating CoT with ablations
INFO - Judging 100 CoT-OFF responses
INFO - Batch judging complete: XX/100 correct
```

---

## Best Practices

1. **Start with GPT-4o-mini** - Best balance of speed/cost/accuracy
2. **Use specific model versions** - Avoids unexpected changes
3. **Monitor costs** - Check OpenAI dashboard regularly
4. **Test with small datasets first** - Use `split = "test[:10]"` for testing
5. **Keep API keys secure** - Never commit them to git

---

## References

- OpenAI API: https://platform.openai.com/docs/
- Anthropic API: https://docs.anthropic.com/
- LLM Judge implementation: `src_pragma/pragma/eval/judge.py`

---

*Last updated: 2026-02-16*
