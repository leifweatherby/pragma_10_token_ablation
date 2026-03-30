# Quick Start: LLM Judge

## TL;DR

```bash
# 1. Set API key
export OPENAI_API_KEY="sk-..."

# 2. Update your config file
# Change: judge = "simple-regex"
# To:     judge = "gpt-4o-mini"
#         base_url = "https://api.openai.com/v1"
#         api_key = "OPENAI_API_KEY"

# 3. Run experiment
pixi run python -m pragma config/your-config.toml --runtype ablations
```

## Cost: ~$0.01 per 100 questions with GPT-4o-mini

## Pre-made Template

Use: `config/qwen3-0.6B-arc-easy-llm-judge.toml`

## Full Documentation

See: `docs/llm-judge-setup.md`

---

## Why LLM Judge?

**Simple regex:**
- Only extracts "A", "B", "C", "D" from text
- Fails on complex responses
- Can't understand context

**LLM judge:**
- Understands reasoning
- Handles varied formats
- More accurate (>95% vs ~85% for regex)
- Worth the tiny cost

---

## Supported Models

| Model | Speed | Cost | Use Case |
|-------|-------|------|----------|
| gpt-4o-mini | Fast | $0.01/100q | **Recommended** |
| claude-3-5-haiku | Fast | $0.05/100q | Most accurate |
| gpt-4o | Slow | $0.10/100q | Complex tasks |
| Local | Fast | Free | Offline work |

---

*Ready to use! Just set `OPENAI_API_KEY` and update your config.*
