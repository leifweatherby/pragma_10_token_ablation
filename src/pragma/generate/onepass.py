"""Code for performing a single pass of CoT-OFF/CoT-ON outputs."""

import logging

from pragma.results import OnePassResult

from .cotgen import run_batches, set_seed

logger = logging.getLogger(__name__)


def run_onepass(examples, model, tokenizer, inference_config):
    """Run a single pass of CoT-ON/CoT-OFF.

    Parameters
    ----------
    examples : list[MultipleChoiceQA]
        List of QA examples to process
    model : AutoModelForCausalLM
        The LLM
    tokenizer : AutoTokenizer
        The LLM tokenizer
    inference_config : InferenceConfig
        Inference configuration

    Returns
    -------
    OnePassResult
        QA pair responses with metadata
    """
    set_seed(inference_config.seed)
    results_by_thinking = {True: [], False: []}

    for run in inference_config.runs:
        logger.info(
            "Generating for enable_thinking=%s...", run.enable_thinking
        )

        results = run_batches(
            examples,
            model=model,
            inference_config=inference_config,
            tokenizer=tokenizer,
            batch_size=inference_config.batch_size,
            enable_thinking=run.enable_thinking,
            generation_kwargs=run.generation_kwargs,
        )

        results_by_thinking[run.enable_thinking].extend(results)

    results = OnePassResult(
        model=inference_config.model,
        seed=inference_config.seed,
        cot_off=tuple(results_by_thinking[False]),
        cot_on=tuple(results_by_thinking[True]),
    )

    return results
