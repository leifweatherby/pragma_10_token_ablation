"""Functions for running distinctiveness metrics."""

import logging

import torch

from pragma.results import DistinctivenessResult
from pragma.tokenize import tokenize_to_ngrams

from .metrics import Metric

logger = logging.getLogger(__name__)


def run_distinctiveness(
    cot_off, cot_on, tokenizer, metric=None, ngram_range=(1, 3)
):
    """Run the distinctiveness metric.

    Parameters
    ----------
    cot_off : list[QAWithResponse]
        CoT-OFF responses
    cot_on : list[QAWithResponse]
        CoT-ON responses
    tokenizer : AutoTokenizer
        The LLM tokenizer
    metric : Metric or None
        Distinctiveness metric; defaults to fighting_words
    ngram_range : tuple[int, int]
        N-gram range

    Returns
    -------
    list[tuple]
        Metric scores for ngrams; score is always index 1
    """
    if metric is None:
        metric = Metric()
        logger.warning("no metric passed; defaulting to %s", metric.metric)

    ngrams_off = tokenize_to_ngrams(
        [qa.response for qa in cot_off], tokenizer, ngram_range
    )
    ngrams_on = tokenize_to_ngrams(
        [qa.reasoning for qa in cot_on], tokenizer, ngram_range
    )

    distinctive = metric(ngrams_on, ngrams_off)

    return distinctive


def process_distinctiveness(distinct_scores, cutoff_quantile=0.95, top_k=None):
    """Process distinctiveness scores and create structured results.

    Note that the top-k filtering doesn't break ties. The requested k value
    determines a cutoff score (the k-th highest distinctive value), and
    `above_cutoff=True` is set for any n-gram with a score greater than or
    equal to this cutoff. This may result in selecting more than k n-grams when
    multiple n-grams share the same k-th highest score.

    Parameters
    ----------
    distinct_scores : list[tuple]
        Distinctiveness results (ngram, score, ...)
    cutoff_quantile : float or None
        Quantile for cutoff threshold; if None and top_k is also None, defaults
        to 0.95
    top_k : int or None
        Number of top n-grams to select; if provided, overrides cutoff_quantile

    Returns
    -------
    tuple[list[DistinctivenessResult], list[tuple], float, dict]
        Distinctiveness results along with the n-grams above our cutoff, the
        cutoff value, and scoring stats
    """
    if len(distinct_scores) == 0:
        return (
            [],
            [],
            0.0,
            {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
            },
        )

    scores = torch.tensor([score for _, score, *_ in distinct_scores])

    # Determine cutoff
    if top_k is not None:
        k = min(top_k, len(scores))
        cutoff = scores.topk(k).values[-1].item()
        logger.debug("Using top-%d values to ablate n-grams", k)
    else:
        if cutoff_quantile is None:
            cutoff_quantile = 0.95
        cutoff = scores.quantile(cutoff_quantile).item()
        logger.debug("Using a quantile cutoff to ablate n-grams")

    distinctiveness_results = []
    ngrams_above_cutoff = []

    for rank, (ngram, score, *_) in enumerate(distinct_scores):
        # Get the cutoff and ensure we're working with Python native datatypes,
        # not ones from PyTorch, which don't serialize
        above_cutoff = bool(score >= cutoff)
        distinctiveness_results.append(
            DistinctivenessResult(
                ngram=ngram,
                score=float(score),
                rank=rank,
                above_cutoff=above_cutoff,
            )
        )
        if above_cutoff:
            ngrams_above_cutoff.append(ngram)

    score_stats = {
        "min": scores.min().item(),
        "max": scores.max().item(),
        "mean": scores.mean().item(),
        "median": scores.median().item(),
    }

    return distinctiveness_results, ngrams_above_cutoff, cutoff, score_stats
