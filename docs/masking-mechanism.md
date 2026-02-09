# Token Masking Mechanism

This document explains how Pragma prevents the generation of specific n-grams during reasoning by hooking into the model's generation process.

## Overview

Pragma uses a custom `LogitsProcessor` to intercept and modify the model's output probabilities at each generation step. This allows us to effectively "withdraw" vocabulary from the reasoning process without modifying the model itself.

## The Logits Tensor

Each forward pass of the model produces a **logits tensor** with shape `[batch_size, vocab_size]`.

For example, with batch size 4 and a 32,000 token vocabulary:

```
logits shape: [4, 32000]
```

- Each row corresponds to one sequence in the batch
- Each column corresponds to one token in the vocabulary
- Values are **unnormalized log-odds**: higher values mean the model considers that token more likely

```
logits[0] = [-2.1, 0.5, -8.3, 3.7, -1.2, ...]
             ↑     ↑     ↑     ↑     ↑
           token  token token token token
             0     1     2     3     4    ... up to vocab_size-1
```

In this example, token 3 has the highest logit (3.7), so the model considers it most likely to come next.

### Conversion to Probabilities

Softmax converts logits to a probability distribution that sums to 1.0:

```python
probs = softmax(logits)
```

## The Generation Loop

When you call `model.generate()`, HuggingFace runs an autoregressive loop:

```
for each generation step:
    1. Model forward pass → produces logits tensor [batch, vocab_size]
    2. LogitsProcessors modify the logits  ← WE HOOK IN HERE
    3. Softmax converts logits to probabilities
    4. Sample/select next token from probabilities
    5. Append token to sequence, repeat
```

## How Masking Works

The `NGramMaskingLogitsProcessor` (in `src/pragma/generate/processors.py`) implements the `LogitsProcessor` interface. At each generation step, its `__call__` method:

1. **Checks if inside reasoning**: Only masks tokens if we're between `<think>` and `</think>` tags
2. **Finds tokens to ban**: For each n-gram length L, checks if the last L-1 tokens match any banned prefix
3. **Modifies the logits**: Either sets banned tokens to `-inf` (mask) or subtracts a penalty (penalize)

### Why Setting to -∞ Works

```python
softmax([-2.1, 0.5, -inf, 3.7, ...])
#                    ↑
#              e^(-inf) = 0
```

The exponential of negative infinity is zero, so that token gets exactly zero probability. It literally cannot be sampled.

## Visual Example

```
Forward pass for "The cat sat on the ___"

                    ┌─────────────────────────────────┐
   input_ids ──────►│      Transformer Model          │
   [1, 450, 2087]   │   (attention, FFN layers, etc)  │
                    └───────────────┬─────────────────┘
                                    │
                                    ▼
                    logits: [1, 32000] tensor

   Token ID:    0      1      2    ...  5765   ...  31999
   Token:     "<pad>" "the"  "a"  ...  "mat"  ...  "<eos>"
   Logit:     -12.3   -4.2   -3.1 ...   4.8   ...   -8.7
                                         ↑
                                   highest logit
                                   = most likely next token

   After softmax:  0%     0%    1%  ...  47%   ...    0%
```

## N-gram Prefix Matching

To ban an n-gram like `(42, 99)` (a bigram of token IDs):

1. The processor precomputes: `prefix_to_next_by_len[2] = {(42,): {99}}`
2. At generation step N, if the sequence ends with token 42:
   - `_get_masked_tokens` finds prefix `(42,)` matches
   - Returns `{99}` as the token to ban
3. `_apply_masking` sets `scores[:, 99] = -inf`
4. After softmax, token 99 has probability ≈ 0
5. The model cannot generate 99, so the bigram `(42, 99)` can never form

## Integration Point

In `src/pragma/generate/cotgen.py`, the processor is passed to `model.generate()`:

```python
outputs = model.generate(
    **batch,
    **generation_kwargs,
    logits_processor=logits_processor  # ← Our NGramMaskingLogitsProcessor
)
```

## Masking Strategies

Two strategies are available:

| Strategy | Effect | Use Case |
|----------|--------|----------|
| `mask` | Sets logits to `-inf` (zero probability) | Complete suppression |
| `penalize` | Subtracts penalty from logits | Soft discouragement |

The penalty is computed as `-log(penalty_weight)`, so a `penalty_weight` of 0.5 reduces the token's probability by half relative to other tokens.

## Key Insight

There is no special "n-gram vocabulary" API in the model. We intercept the probability distribution at each step and zero out tokens that would complete banned sequences. The model doesn't "know" certain vocabulary is withdrawn—it simply finds those tokens have zero probability when it tries to use them.
