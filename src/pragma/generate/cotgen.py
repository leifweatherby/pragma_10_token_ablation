"""Core CoT generation code."""

import logging
import random

import torch
from torch.utils.data import DataLoader
from transformers import LogitsProcessorList

from pragma.mcqa import QAWithResponse
from pragma.tokenize import detokenize_traces, get_special_start_end, tokenize

logger = logging.getLogger(__name__)


def collate_fn(
    tokenizer,
    enable_thinking=True,
    device=None,
):
    """Create a collate function for DataLoader batching.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        The LLM tokenizer
    enable_thinking : bool
        Whether to enable thinking mode
    device : str or torch.device or None
        Model device

    Returns
    -------
    callable
        The collate function
    """

    def collate(batch):
        """Collate batch.

        Parameters
        ----------
        batch : list[MultipleChoiceQA]
            The batch

        Returns
        -------
        dict
            Tokenized batch
        """
        tokenized = tokenize(
            [qa.to_prompt() for qa in batch],
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
            device=device,
        )
        tokenized["qa"] = batch

        return tokenized

    return collate


def set_seed(seed):
    """Set a seed for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_batches(
    examples,
    model,
    inference_config,
    tokenizer,
    batch_size=32,
    enable_thinking=True,
    generation_kwargs=None,
    logits_processors=None,
    shuffle=False,
    num_workers=0,
):
    """Run batched inference.

    Parameters
    ----------
    examples : list[MultipleChoiceQA]
        List of QA examples to process
    model : AutoModelForCausalLM
        The LLM
    inference_config : InferenceConfig
        Config such as think_substring, think_token_format
    tokenizer : AutoTokenizer
        The LLM tokenizer
    batch_size : int
        Batch size for inference
    enable_thinking : bool
        Whether to enable thinking mode
    generation_kwargs : dict or None
        Additional keywords to pass to model.generate()
    logits_processors : list or None
        Logits processors to pass to model.generate()
    shuffle : bool
        Whether to shuffle examples
    num_workers : int
        DataLoader workers

    Returns
    -------
    list[QAWithResponse]
        QA pairs with responses

    Raises
    ------
    torch.cuda.OutOfMemoryError
        If the GPU runs out of memory
    """
    model.eval()
    if generation_kwargs is None:
        generation_kwargs = {}

    if logits_processors is None:
        logits_processor = LogitsProcessorList([])
    else:
        logits_processor = LogitsProcessorList(logits_processors)

    special_tokens = (
        get_special_start_end(
            tokenizer,
            substring=inference_config.think_substring,
            token_format=inference_config.think_token_format,
        )
        if enable_thinking
        else None
    )

    loader = DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn(
            tokenizer,
            enable_thinking=enable_thinking,
            device=model.device,
        ),
    )

    n_batch = len(loader)
    results = []

    for i, batch in enumerate(loader):
        # Pop out the QA pairs
        qa_batch = batch.pop("qa")

        # Send the inputs the model
        try:
            outputs = model.generate(
                **batch, **generation_kwargs, logits_processor=logits_processor
            )
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM at batch %d", i + 1)
            raise

        # Detokenize traces and answers
        traces, answers = detokenize_traces(
            batch["input_ids"],
            outputs,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
            special_tokens=special_tokens,
        )

        if not traces:
            traces = [""] * len(answers)

        # Create new QAWithResponse objects
        for qa, trace, answer in zip(qa_batch, traces, answers):
            results.append(
                QAWithResponse(
                    qa=qa,
                    reasoning=trace,
                    response=answer,
                    reasoning_enabled=enable_thinking,
                )
            )

        if (i + 1) % 10 == 0 or (i + 1) == n_batch:
            logger.info("Completed batch %d/%d", i + 1, n_batch)

    return results
