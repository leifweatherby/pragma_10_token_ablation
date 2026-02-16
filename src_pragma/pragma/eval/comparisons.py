"""Utilities for comparing CoT-OFF and CoT-ON output."""


def cot_is_better(cot_off_verdicts, cot_on_verdicts, min_improvement=0):
    """Check whether CoT-ON performance is better than CoT-OFF.

    If min_improvement = 0, any positive change means CoT-ON is better. When
    this argument is 1, we require 2 net correct answers; 5 would require 6,
    etc.

    Parameters
    ----------
    cot_off_verdicts : list[LLMJudgeVerdict]
        Verdicts from CoT-OFF
    cot_on_verdicts : list[LLMJudgeVerdict]
        Verdicts from CoT-ON
    min_improvement : int
        Minimum net improvement in correct answers required

    Returns
    -------
    bool
        Whether CoT-ON performance has net improvement over CoT-OFF

    Raises
    ------
    ValueError
        If number of verdicts don't match
    AssertionError
        If the indexing across CoT-OFF/ON verdicts is misaligned
    """
    if len(cot_off_verdicts) != len(cot_on_verdicts):
        raise ValueError("Mismatched number of verdicts for CoT-OFF/ON")

    # Align verdicts to sort order
    cot_off_verdicts = sorted(cot_off_verdicts, key=hash)
    cot_on_verdicts = sorted(cot_on_verdicts, key=hash)

    improvements = 0
    regressions = 0

    for off, on in zip(cot_off_verdicts, cot_on_verdicts):
        assert hash(off) == hash(on), "IDs for QA pairs don't match"

        # Did CoT-ON improve or get worse?
        if on.correct and not off.correct:
            improvements += 1
        elif off.correct and not on.correct:
            regressions += 1

    net_improvement = improvements - regressions

    return net_improvement > min_improvement
