"""Tokenization utilities for reasoning traces."""

import logging

import torch

logger = logging.getLogger(__name__)


def get_special_start_end(
    tokenizer, substring="think", token_format="<{tok}>"
):
    """Return special start/end tokens if they exist.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        The LLM tokenizer
    substring : str
        String content of the special token
    token_format : str
        Format for the tokenizer

    Returns
    -------
    tuple[int, int] or None
        The special start/end tokens
    """
    start = token_format.format(tok=substring)
    end = token_format.format(tok=f"/{substring}")

    start_id = tokenizer.convert_tokens_to_ids(start)
    end_id = tokenizer.convert_tokens_to_ids(end)

    unk = getattr(tokenizer, "unk_token_id", None)
    if (
        start_id is None
        or end_id is None
        or (unk is not None and (start_id == unk or end_id == unk))
    ):
        logger.warning("No single-token markers for %s/%s found", start, end)
        return None

    return start_id, end_id


def strip_prompt(input_ids, output_ids):
    """Strip prompt tokens from a prompt/generation pair.

    Parameters
    ----------
    input_ids : torch.Tensor
        Prompt IDs
    output_ids : torch.Tensor
        Prompt and generated IDs

    Returns
    -------
    torch.Tensor
        Just the generated IDs
    """
    L = input_ids.shape[0]
    if output_ids.shape[0] >= L and torch.equal(output_ids[:L], input_ids):
        return output_ids[L:]

    return output_ids


def _split_think_text(text, start_tag="<think>", end_tag="</think>"):
    """Split text into (trace, answer).

    Parameters
    ----------
    text : str
        The text to split
    start_tag : str
        Start of thinking tag
    end_tag : str
        End of thinking tag

    Returns
    -------
    tuple[str, str]
        The trace and answer
    """
    s = text.find(start_tag)
    e = text.find(end_tag, s + 1 if s != -1 else 0)

    if s != -1 and e != -1 and e > s:
        trace = text[s + len(start_tag) : e]
        answer = text[e + len(end_tag) :]
    elif s != -1 and (e == -1 or e < s):
        # Only opening tag
        trace = text[s + len(start_tag) :]
        answer = ""
    elif s == -1 and e != -1:
        # Only closing tag
        trace = text[:e]
        answer = text[e + len(end_tag) :]
    else:
        # No tags
        trace = ""
        answer = text

    return trace, answer


def detokenize_traces(
    input_ids,
    output_ids,
    tokenizer,
    enable_thinking=False,
    special_tokens=None,
):
    """Detokenize model outputs into reasoning traces and answers.

    Parameters
    ----------
    input_ids : torch.Tensor
        Prompt IDs
    output_ids : torch.Tensor
        Prompt and generated IDs
    tokenizer : AutoTokenizer
        The LLM tokenizer
    enable_thinking : bool
        Whether to enable thinking mode
    special_tokens : tuple[int, int] or None
        Special tokens like start/end of reasoning

    Returns
    -------
    tuple[list[str], list[str]]
        Batch of reasoning traces and answers
    """
    gens = [
        strip_prompt(input_ids[i], output_ids[i])
        for i in range(output_ids.shape[0])
    ]

    # No reasoning requested
    if not enable_thinking:
        return [], tokenizer.batch_decode(gens, skip_special_tokens=True)

    # Fallback to string parsing if no markers provided
    if special_tokens is None:
        decoded = tokenizer.batch_decode(gens, skip_special_tokens=False)

        traces, answers = [], []
        for text in decoded:
            trace, answer = _split_think_text(text)

            if trace:
                trace_dec = tokenizer.decode(
                    tokenizer.encode(trace, add_special_tokens=False),
                    skip_special_tokens=True,
                )
            else:
                trace_dec = ""

            if answer:
                answer_dec = tokenizer.decode(
                    tokenizer.encode(answer, add_special_tokens=False),
                    skip_special_tokens=True,
                )
            else:
                answer_dec = ""

            traces.append(trace_dec)
            answers.append(answer_dec)

        return traces, answers

    # Token-based extraction
    start_id, end_id = special_tokens
    traces, answers = [], []

    for seq in gens:
        starts = (seq == start_id).nonzero().flatten()
        ends = (seq == end_id).nonzero().flatten()

        if len(starts) > 0:
            s = starts[0].item()
            ends_after = ends[ends > s]
            if len(ends_after) > 0:
                e = ends_after[0].item()
                trace_ids = seq[s + 1 : e]
                answer_ids = seq[e + 1 :]
            else:
                # Only opening tag
                trace_ids = seq[s + 1 :]
                answer_ids = torch.empty(0, dtype=seq.dtype, device=seq.device)

        elif len(ends) > 0:
            # Only closing tag
            e = ends[0].item()
            trace_ids = seq[:e]
            answer_ids = seq[e + 1 :]

        else:
            # No tags
            trace_ids = torch.empty(0, dtype=seq.dtype, device=seq.device)
            answer_ids = seq

        trace = tokenizer.decode(trace_ids, skip_special_tokens=True)
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
        traces.append(trace)
        answers.append(answer)

    return traces, answers
