Run Configuration
=================

We use a configuration layer to manage various settings for a chain-of-thought
run. These settings are stored as TOML files in `config/`.

Current reference configuration
-------------------------------

To stand up a new run, copy the block below and paste it into a new file:

```toml
[model]
name = "HuggingFaceTB/SmolLM3-3B"  # Hugging Face model ID
think_substring = "think"
think_token_format = "<{tok}>"

[dataset]
name = "allenai/ai2_arc"
config = "ARC-Challenge"
split = "train[:128]"  # Use slice syntax for subsets

[io]
output = "data/smollm3-3B"

[runtime]
batch_size = 32
seed = 357
attn_implementation = "sdpa"

[ablations]
judge = "claude-haiku-4-5"
base_url = "https://api.anthropic.com/v1/"
api_key = "ANTHROPIC_API_KEY"  # Environment variable name
metric = "fighting_words"
ngram_range = [1, 1]  # [min_n, max_n]
strategy = "mask"  # "mask" or "penalize"
penalty_weight = 1.0
cutoff = 0.95
top_k = ""  # Empty string or integer
max_steps = 100

[runs.think]
enable_thinking = true

[runs.think.generation]
do_sample = true
max_new_tokens = 16384
temperature = 0.6
top_p = 0.95
top_k = 20
min_p = 0.0

[runs.no_think]
enable_thinking = false

[runs.no_think.generation]
do_sample = true
max_new_tokens = 4096
temperature = 0.6
top_p = 0.95
top_k = 20
```

Settings
--------

| Section                   | Variable            | Description                                                                                                          |
|---------------------------|---------------------|----------------------------------------------------------------------------------------------------------------------|
| model                     | name                | Identifier for the base model to run (e.g., a Hugging Face repository like "Qwen/Qwen3-0.6B")                        |
| model                     | think_substring     | Substring used to mark or detect "thinking" segments/tokens in prompts or outputs (e.g., "think")                    |
| model                     | think_token_format  | Format string for wrapping the thinking marker; `{tok}` is replaced by `think_substring` (e.g., "<{tok}>")           |
| dataset                   | name                | Dataset identifier (e.g., a Hugging Face dataset like "allenai/ai2_arc")                                             |
| dataset                   | config              | Dataset configuration or subset name (e.g., "ARC-Challenge")                                                         |
| dataset                   | split               | Dataset split and optional slice using HF syntax (e.g., "train[:8]" = first 8 examples of the train split)           |
| io                        | output              | Path to the output directory where results (e.g., generations or metrics) are written (e.g., "data/qwen3-0.6B")      |
| runtime                   | batch_size          | Number of examples processed per batch during inference or evaluation                                                |
| runtime                   | seed                | Random seed for reproducible sampling and shuffling                                                                  |
| runtime                   | attn_implementation | Attention backend selection (e.g., "sdpa" for PyTorch scaled dot-product attention)                                  |
| ablations                 | judge               | Name/ID of the judging model used to evaluate outputs (e.g., "claude-sonnet-4-5")                                    |
| ablations                 | base_url            | Base URL for the judge model's API (e.g., Anthropic endpoint)                                                        |
| ablations                 | api_key             | Name of the environment variable containing the API key, used to authenticate requests to the judge API              |
| ablations                 | metric              | Evaluation metric applied during ablation analysis (e.g., "fighting_words")                                          |
| ablations                 | ngram_range         | Range of n-gram sizes considered by the metric, in [min_n, max_n] form (e.g., [1, 3] for unigrams to trigrams)       |
| ablations                 | strategy            | Masking strategy; "mask" ablates completely, whereas "penalize" reduces n-gram probability by a fixed amount         |
| ablations                 | penalty_weight      | Probability reduction factor when strategy is "penalize"; must be 0 < rate <= 1.0                                    |
| ablations                 | cutoff              | Score/quantile threshold used by the metric to flag significant differences; higher values are stricter (e.g., 0.98) |
| ablations                 | top_k               | If not `""`, ignore `cutoff` and select top-k most distinctive n-gram values ('values' is key: no tie-breaking here) |
| ablations                 | max_steps           | Maximum number of examples to process during the ablation run                                                        |
| runs.think                | enable_thinking     | Enables "thinking" mode (e.g., structured reasoning segments marked by the think token/format                        |
| runs.no_think             | enable_thinking     | Disables "thinking" mode; generate answers without special reasoning segments                                        |
| runs.<...>.generation     | do_sample           | If true, use sampling-based decoding (non-greedy)                                                                    |
| runs.<...>.generation     | max_new_tokens      | Maximum number of tokens to generate beyond the prompt                                                               |
| runs.<...>.generation     | temperature         | Sampling temperature; higher values increase randomness                                                              |
| runs.<...>.generation     | top_p               | Nucleus sampling parameter; sample from the smallest set of tokens whose cumulative probability ≥ top_p              |
| runs.<...>.generation     | top_k               | Top-k sampling parameter; restricts sampling to the k most probable tokens                                           |
| runs.<...>.generation     | min_p               | Minimum probability threshold for candidate tokens; values below this are filtered (0.0 disables the filter)         |
