"""Tokenization utilities for n-grams."""


def tokenize_to_ngrams(texts, tokenizer, ngram_range=(1, 1)):
    """Tokenize texts and extract n-grams.

    Parameters
    ----------
    texts : Iterable[str]
        The texts
    tokenizer : AutoTokenizer
        The LLM tokenizer
    ngram_range = int or tuple[int, int]
        N-gram range; use a single number for only one kind of n-grams

    Parameters
    -----------
    list[list[tuple[int]]]
        Document-wise n-grams

    Raises
    ------
    ValueError
        If the requested n-gram range is invalid
    """
    if isinstance(ngram_range, int):
        min_n, max_n = ngram_range, ngram_range + 1
    else:
        min_n, max_n = ngram_range

    if min_n < 1 or max_n < min_n:
        raise ValueError(f"Invalid ngram_range: {ngram_range}")

    all_texts = []
    for text in texts:
        text_ngrams = []
        if not text:
            # Preserve empties
            all_texts.append(text_ngrams)
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        for n in range(min_n, max_n + 1):
            if n == 1:
                ngrams = [(tok,) for tok in tokens]
            else:
                ngrams = [
                    tuple(tokens[i : i + n])
                    for i in range(len(tokens) - n + 1)
                ]
            text_ngrams.extend(ngrams)

        all_texts.append(text_ngrams)

    return all_texts


def decode_ngram(ngram, tokenizer):
    """Decode an n-gram tuple back to text.

    Parameters
    ----------
    ngram : tuple[int, ...]
        The n-gram
    tokenizer : AutoTokenizer
        The LLM tokenizer

    Returns
    -------
    str
        Decoded n-gram
    """
    return tokenizer.decode(list(ngram))
