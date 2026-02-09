# Architecture Overview

This document explains the key components and how they work together.

## High-Level Architecture

The codebase is organized into three main areas:

1. **Configuration & Data Models** (top level)
2. **Generation** (generating CoT reasoning traces)
3. **Tokenization & N-gram Analysis** (extracting and analyzing patterns)

---

## Key Libraries Used

### Transformers (HuggingFace)
- `AutoModelForCausalLM`: Loads pre-trained language models for text generation
- `AutoTokenizer`: Converts text to token IDs and back
- `LogitsProcessor`: Base class for modifying model predictions during generation

### PyTorch
- Used for all tensor operations and model inference
- `torch.utils.data.DataLoader`: Batches data for efficient processing

### Datasets (HuggingFace)
- `load_dataset`: Loads benchmark datasets (like ARC-Challenge, MMLU)

---

## Core Components

### 1. Configuration System (`config.py`)

Loads TOML configuration files that specify model, dataset, runs, and generation parameters.

**Key classes:**
- `LoaderMixin`: Provides `from_toml()` class method for loading configs
- `RunConfig`: Settings for a single run (thinking on/off, generation params)
- `InferenceConfig`: Top-level config containing model, dataset, and multiple runs

### 2. Data Models (`mcqa.py`)

**`MultipleChoiceQA`**: Represents a QA example with question, choices, correct answer, and prompt formatting methods.

**`QAWithResponse`**: Wraps a QA pair with the model's output (reasoning trace, final response, thinking enabled flag).

### 3. Generation Pipeline (`generate/`)

**`__main__.py`** - Entry point:
1. Loads config from TOML file
2. Loads model and tokenizer
3. Loads dataset examples
4. For each run configuration, generates responses
5. Saves results to JSON

**`cotgen.py`** - Core generation logic:
- `run_batches()`: Main inference loop using PyTorch DataLoader for batching
- `collate_fn()`: Prepares batches by tokenizing prompts

**`processors.py`** - Logit masking during generation:
- `NGramMaskingLogitsProcessor`: Prevents the model from generating specific n-grams during reasoning by tracking position in sequence and banning tokens that would complete forbidden n-grams

### 4. Tokenization & Reasoning Extraction (`tokenize/`)

**`tokenization.py`**:
- `tokenize()`: Converts prompts to token IDs using `tokenizer.apply_chat_template()` with left-padding for batched inference

**`reasoning.py`**: Extracts reasoning traces from model outputs
- `get_special_start_end()`: Finds token IDs for `<think>` and `</think>`
- `detokenize_traces()`: Separates reasoning from answer

**`ngrams.py`**: N-gram extraction
- `tokenize_to_ngrams()`: Tokenizes text and extracts n-grams as tuples of token IDs
- `decode_ngram()`: Converts token ID tuples back to text

### 5. Distinctiveness Metrics (`metrics.py`)

Statistical methods to find n-grams characteristic of CoT reasoning vs. non-CoT answers. All methods compare two corpora (target vs. comparison) and return scored n-grams.

**`dunning()`**: Log-likelihood ratio test for comparing word frequencies between corpora.

**`fighting_words()`**: Weighted log-odds with Dirichlet prior (Monroe et al. 2008). More robust to corpus size differences.

**`pmi()`**: Pointwise Mutual Information - information-theoretic measure of association.

**`Metric`** class: Wrapper providing unified interface:
```python
metric = Metric("fighting_words")
results = metric(target_ngrams, compare_ngrams)
```

---

## How It All Fits Together

### Typical Workflow:

1. **Configuration**: Define runs in TOML file
   - Run 1: `enable_thinking=true` (generates `<think>reasoning</think>answer`)
   - Run 2: `enable_thinking=false` (generates just answer)

2. **Generation** (`pixi r python -m pragma.generate -c config/testing.toml`):
   - Loads model and dataset
   - For each run, generates responses in batches
   - Saves results to JSON

3. **Analysis**:
   - Load JSON results
   - Extract reasoning traces from thinking=true runs
   - Tokenize into n-grams using `tokenize_to_ngrams()`
   - Compare CoT vs non-CoT using `Metric("fighting_words")`

4. **Masked Generation**:
   - Find distinctive n-grams from analysis
   - Create `NGramMaskingLogitsProcessor` with those n-grams
   - Pass as `logits_processor` to `model.generate()`
   - Measure how model performance changes when forbidden from using certain reasoning patterns

---

## Key Concepts

**Chat Templates**: Models expect specific formatting with system/user/assistant roles.

**Thinking Tokens**: Models support `<think>...</think>` for explicit reasoning before answering.

**Logit Processors**: Modify model predictions during generation by adjusting logits at each step.

**Left Padding**: For batched generation, shorter sequences are padded on the left so all generations start at the same position.
