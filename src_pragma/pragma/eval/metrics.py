"""Metrics functions."""

from collections import Counter

import numpy as np


def _align_counts(vocab, counter):
    """Return counts aligned to vocab order.

    Parameters
    ----------
    vocab : list[tuple[int, ...]]
        Vocabulary of n-grams
    counter : Counter
        N-gram counts

    Returns
    -------
    np.array
        Counts in vocab order
    """
    return np.fromiter(
        (counter[ng] for ng in vocab), dtype=np.float64, count=len(vocab)
    )


def _count_ngrams(corpus):
    """Count n-grams in a corpus of samples, where each sample is a list of
    n-gram tuples.

    Parameters
    ----------
    corpus : list[list[tuple[int, ...]]]
        The corpus of n-grams

    Returns
    -------
    Counter
        N-gram counts
    """
    counter = Counter()
    for doc in corpus:
        counter.update(doc)

    return counter


def _count_ngrams_presence(corpus):
    """Count one-hot document frequency (each sample contributes at most 1 per
    n-gram.

    Parameters
    ----------
    corpus : list[list[tuple[int, ...]]]
        The corpus of n-grams

    Returns
    -------
    Counter
        N-gram counts
    """
    counter = Counter()
    for doc in corpus:
        counter.update(set(doc))

    return counter


def dunning(target, compare, eps=1e-10):
    """Calculate signed log-likelihood ratio.

    Score sign is determined by the difference in proportions p_t - p_c.
    Z-score is approximated as sign * sqrt(LLR) (chi-square with df=1).

    Parameters
    ----------
    target : list[list[tuple[int, ...]]]
        N-grams of token IDs in target class
    compare : list[list[tuple[int, ...]]]
        N-grams of token IDs in comparison class
    eps : float
        Numerical stability value

    Returns
    -------
    list[tuple[tuple[int, ...], float, float, int, int]]
        N-grams, score, z-score, target count, and comparison count
    """
    tgt_counts = _count_ngrams(target)
    cmp_counts = _count_ngrams(compare)

    # Build vocabulary
    vocab = list(set(tgt_counts.keys()) | set(cmp_counts.keys()))
    V = len(vocab)
    if V == 0:
        return []

    # N-gram counts per class
    y_t = _align_counts(vocab, tgt_counts)
    y_c = _align_counts(vocab, cmp_counts)

    # Total n-grams for each class, then the entire corpus
    n_target = y_t.sum()
    n_compare = y_c.sum()
    T = n_target + n_compare

    a = y_t
    b = y_c
    c = np.maximum(n_target - a, 0)
    d = np.maximum(n_compare - b, 0)

    # Calculate probabilities
    p = (a + b) / (T + eps)
    p1 = a / (n_target + eps)
    p2 = b / (n_compare + eps)

    # Expected counts under null hypothesis
    Ea = n_target * p
    Eb = n_compare * p
    Ec = n_target * (1 - p)
    Ed = n_compare * (1 - p)

    # Log-likelihood ratio (2 * sum O * log(O/E))
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = []
        for O, E in ((a, Ea), (b, Eb), (c, Ec), (d, Ed)):
            term = np.where(O > 0, O * np.log((O + eps) / (E + eps)), 0.0)
            terms.append(term)

        llr_vals = 2.0 * sum(terms)

    sign = np.sign(p1 - p2)
    score = sign * llr_vals
    z = sign * np.sqrt(np.maximum(llr_vals, 0))

    # Order by LLR and package it all up
    order = np.argsort(-score)

    result = []
    for i in order:
        ng = vocab[i]
        result.append((ng, score[i], z[i], a[i], b[i]))

    return result


def fighting_words(
    target,
    compare,
    alpha_total=100.0,
    use_presence=True,
    normalize_lengths=True,
):
    """Calculate weighted log-odds with informative Dirichlet prior (aka "fighting words").

    Parameters
    ----------
    target : list[list[tuple[int, ...]]]
        N-grams of token IDs in target class
    compare : list[list[tuple[int, ...]]]
        N-grams of token IDs in comparison class
    alpha_total : float
        Total prior mass; larger values increase smoothing
    use_presence : bool
        If true, count eacn n-gram once per document to reduce
        length/repetition sensitivity. If false, use frequency counts
    normalize_lengths : bool
        If true, subtract log(n_target/n_compare) to normalize for class length

    Returns
    -------
    list[tuple[tuple[int, ...], float, float, int, int]]
        N-grams, score, z-score, target count, and comparison count
    """
    if use_presence:
        tgt_counts = _count_ngrams_presence(target)
        cmp_counts = _count_ngrams_presence(compare)
    else:
        tgt_counts = _count_ngrams(target)
        cmp_counts = _count_ngrams(compare)

    # Build vocabulary
    vocab = list(set(tgt_counts.keys()) | set(cmp_counts.keys()))
    V = len(vocab)
    if V == 0:
        return []

    # N-gram counts per class
    y_t = _align_counts(vocab, tgt_counts)
    y_c = _align_counts(vocab, cmp_counts)

    # Total n-grams for each class
    n_target = y_t.sum()
    n_compare = y_c.sum()

    # Informative prior: alpha_i proportional to background frequency
    bg = y_t + y_c
    bg_sum = bg.sum()
    if bg_sum > 0:
        alpha_i = alpha_total * (bg / bg_sum)
    else:
        # Fallback: uniform prior
        alpha_i = np.full(V, alpha_total / max(V, 1), dtype=np.float64)

    alpha_0 = alpha_total

    # Log-odds with prior
    l_t = np.log((y_t + alpha_i) / (n_target + alpha_0 - (y_t + alpha_i)))
    l_c = np.log((y_c + alpha_i) / (n_compare + alpha_0 - (y_c + alpha_i)))
    delta = l_t - l_c

    # Optinally normalize log-odds by length
    if normalize_lengths and n_target > 0 and n_compare > 0:
        delta = delta - np.log(n_target / n_compare)

    # Calculate Z-scores
    var = 1.0 / (y_t + alpha_i) + 1.0 / (y_c + alpha_i)
    z = delta / np.sqrt(var)

    # Order by log-odds and package it all up
    order = np.argsort(-delta)

    result = []
    for i in order:
        ng = vocab[i]
        result.append((ng, delta[i], z[i], y_t[i], y_c[i]))

    return result


def pmi(target, compare, smoothing=1.0, normalized=True, base=np.e):
    """Calculate pointwise mutual information between n-gram and the target
    class.

    PMI(ng, target) = log p(ng | target) - log p(ng)

    Parameters
    ----------
    target : list[list[tuple[int, ...]]]
        N-grams of token IDs in target class
    compare : list[list[tuple[int, ...]]]
        N-grams of token IDs in comparison class
    smoothing : float
        Additive smoothing to avoid zero division
    normalized : bool
        If True, return normalized PMI: PMI / -log p(ng, target)
    base : float
        Logarithm base for PMI (e.g., np.e or 2)

    Returns
    -------
    list[tuple[tuple[int, ...], float, int, int]]
        N-grams, pmi, target count, and comparison count
    """
    tgt_counts = _count_ngrams(target)
    cmp_counts = _count_ngrams(compare)

    # Build vocabulary
    vocab = list(set(tgt_counts.keys()) | set(cmp_counts.keys()))
    V = len(vocab)
    if V == 0:
        return []

    # N-gram counts per class
    y_t = _align_counts(vocab, tgt_counts)
    y_c = _align_counts(vocab, cmp_counts)

    # Total n-grams for each class, then the entire corpus
    n_target = y_t.sum()
    n_compare = y_c.sum()
    T = n_target + n_compare

    # Smoothed probabilities: p(ng, target), p(ng), p(target | total)
    p_ng_t = (y_t + smoothing) / (n_target + smoothing * V)
    p_ng = (y_t + y_c + smoothing) / (T + smoothing * V)
    p_ng_and_target = (y_t + smoothing) / (T + smoothing * V)

    # Calculate PMI
    with np.errstate(divide="ignore"):
        pmi_vals = np.log(p_ng_t / p_ng) / np.log(base)

    # Are we normalizing?
    if normalized:
        denom = -np.log(p_ng_and_target) / np.log(base)
        denom = np.where(denom == 0, np.inf, denom)
        pmi_vals = pmi_vals / denom

    # Order by PMI and package it all up
    order = np.argsort(-pmi_vals)

    result = []
    for i in order:
        if not np.isfinite(pmi_vals[i]):
            continue

        ng = vocab[i]
        result.append((ng, pmi_vals[i], y_t[i], y_c[i]))

    return result


class Metric:
    """A wrapper class for distinctiveness metrics."""

    available = {
        "dunning": dunning,
        "fighting_words": fighting_words,
        "pmi": pmi,
    }

    def __init__(self, metric="fighting_words"):
        """Initialize the metric.

        Select "dunning", "fighting_words", or "pmi" for `metric`.

        Parameters
        ----------
        metric : str
            Name of the metric

        Raises
        ------
        NotImplementedError
            If the requested metric isn't implemented
        """
        self.metric = metric
        self.func = self.available.get(metric, None)
        if self.func is None:
            raise NotImplementedError(f"{metric} unavailable")

    def __call__(self, target, compare, **kwargs):
        """Calculate the metric.

        Parameters
        ----------
        target : list[list[tuple[int, ...]]]
            N-grams of token IDs in target class
        compare : list[list[tuple[int, ...]]]
            N-grams of token IDs in comparison class
        kwargs : dict
            Keyword arguments for the metric

        Returns
        -------
        list[tuple[tuple[int, ...], float | int, ...]]
            N-gram, metric score, z-score, class counts
        """
        return self.func(target, compare, **kwargs)

    @property
    def __doc__(self):
        """Return the docstring of the active metric function."""
        return self.func.__doc__

    @classmethod
    def list_available(cls):
        """List available metrics.

        Returns
        -------
        list[str, ...]
            Available metrics
        """
        return list(cls.available.keys())
