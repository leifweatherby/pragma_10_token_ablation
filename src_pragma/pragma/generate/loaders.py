"""Model and data loading functions."""

import logging
import random

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from pragma.mcqa import MultipleChoiceQA

logger = logging.getLogger(__name__)


def load_model(inference_config):
    """Load a model and its tokenizer.

    Parameters
    ----------
    inference_config : InferenceConfig
        Inference configuration

    Returns
    -------
    tuple[AutoTokenizer, AutoModelForCausalLM]
        The LLM tokenizer and LLM
    """
    logger.info("Loading %s", inference_config.model)

    tokenizer = AutoTokenizer.from_pretrained(inference_config.model)
    model = AutoModelForCausalLM.from_pretrained(
        inference_config.model,
        dtype="auto",
        device_map="auto",
        attn_implementation=inference_config.attn_implementation,
    )

    return tokenizer, model


def load_benchmark(inference_config):
    """Load Hugging Face benchmark examples using InferenceConfig metadata.

    Parameters
    ----------
    inference_config : InferenceConfig
        Configuration information with dataset name, config, and split

    Returns
    -------
    list[MultipleChoiceQA]
        Benchmark examples

    Raises
    ------
    ValueError
        If there isn't a loader function for the requested dataset config
    """
    loader_key = (
        inference_config.dataset_name,
        inference_config.dataset_config,
    )
    LOADER_MAP = {
        ("allenai/ai2_arc", "*"): _load_arc,
        ("tasksource/bigbench", "*"): _load_bigbench,
        ("cais/mmlu", "*"): _load_mmlu,
        ("allenai/openbookqa", "main"): _load_openbookqa,
    }

    loader = LOADER_MAP.get(loader_key)
    if not loader:
        try:
            # Fallback: flexible match for benchmarks like MMLU and BIG-Bench
            loader_key = (inference_config.dataset_name, "*")
            loader = LOADER_MAP.get(loader_key)
        except Exception:
            raise ValueError(
                f"No loader for dataset: {inference_config.dataset_name}"
            )

    examples = loader(
        config=inference_config.dataset_config, split=inference_config.split
    )

    logger.info(
        "Using %d examples from %s",
        len(examples),
        inference_config.dataset_name,
    )

    return examples


def _generate_choice_labels(choices):
    """Generate dynamic choice labels.

    Supports up to 702 choices using single letters, then double letters. Cf.
    the format of BIG-Bench.

    Parameters
    ----------
    choices : Iterable
        Possible choices

    Returns
    -------
    tuple[str]
        Choice labels
    """
    num_choices = len(choices)

    labels = []
    for i in range(num_choices):
        if i < 26:
            # Single letters: A-Z
            labels.append(chr(65 + i))
        else:
            # Double letters: AA, BB, ...
            label = ""
            n = i - 26
            while True:
                label = chr(65 + (n % 26)) + label
                n //= 26
                if n == 0:
                    break

            labels.append(label)

    return tuple(labels)


def _load_arc(config="ARC-Challenge", split="test"):
    """Load ARC-* benchmark.

    Parameters
    ----------
    config : str
        Dataset config, e.g. "ARC-Challenge"
    split : str
        The split to load

    Returns
    -------
    list[MultipleChoiceQA]
        Benchmark examples
    """
    ds = load_dataset("allenai/ai2_arc", config, split=split)
    return [
        MultipleChoiceQA(
            id=ex["id"],
            question=ex["question"],
            subject=ex.get("subject", ""),
            choices=tuple(ex["choices"]["text"]),
            choice_labels=tuple(ex["choices"]["label"]),
            correct_answer=ex["answerKey"].upper(),
        )
        for ex in ds
    ]


def _load_bigbench(config="", split="test"):
    """Load BIG-Bench benchmark.

    Parameters
    ----------
    config : str
        Dataset config, e.g. "analogical_similarity"
    split : str
        The split to load

    Returns
    -------
    list[MultipleChoiceQA]
        Benchmark examples
    """
    ds = load_dataset("tasksource/bigbench", config, split=split)

    results = []
    for ex in ds:
        mc_targets = ex.get("multiple_choice_targets", [])
        target = ex["targets"][0]

        # If multiple choice targets exist, use them
        if mc_targets and len(mc_targets) > 1:
            choices = mc_targets
        else:
            if config not in ("word_sorting", "word_unscrambling"):
                logger.warning(
                    "Using free-form strings with an unrecognized config: %s",
                    config,
                )

            # Free-form format: generate choices by scrambling target string
            words = target.split()
            if len(words) < 2:
                continue

            # Set the correct answer, then shuffle
            choices = [target]
            attempts = 0
            while len(choices) < 4 and attempts < 50:
                shuffled = words.copy()
                random.shuffle(shuffled)

                shuffled_str = " ".join(shuffled)
                if shuffled_str not in choices:
                    choices.append(shuffled_str)

                attempts += 1

            if len(choices) < 4:
                continue

            # Shuffle everything one last time so that the first answer isn't
            # always the correct one
            random.shuffle(choices)

        labels = _generate_choice_labels(choices)
        correct_idx = choices.index(target)

        results.append(
            MultipleChoiceQA(
                id=ex["idx"],
                question=ex["inputs"],
                subject=config,
                choices=tuple(choices),
                choice_labels=labels,
                correct_answer=labels[correct_idx],
            )
        )

    return results


def _load_mmlu(config="all", split="test"):
    """Load MMLU benchmark.

    Parameters
    ----------
    config : str
        Dataset config, e.g. "all"
    split : str
        The split to load

    Returns
    -------
    list[MultipleChoiceQA]
        Benchmark examples
    """
    ds = load_dataset("cais/mmlu", config, split=split)

    results = []
    for i, ex in enumerate(ds):
        labels = _generate_choice_labels(ex["choices"])
        correct_idx = ex["answer"]

        results.append(
            MultipleChoiceQA(
                id=f"mmlu_{i}",
                question=ex["question"],
                subject=ex.get("subject", ""),
                choices=tuple(ex["choices"]),
                choice_labels=labels,
                correct_answer=labels[correct_idx],
            )
        )

    return results


def _load_openbookqa(config="main", split="test"):
    """Load OpenBookQA benchmark.

    Parameters
    ----------
    config : str
        Dataset config, e.g. "main"
    split : str
        The split to load

    Returns
    -------
    list[MultipleChoiceQA]
        Benchmark examples
    """
    ds = load_dataset("openbookqa", config, split=split)
    return [
        MultipleChoiceQA(
            id=ex["id"],
            question=ex["question_stem"],
            subject=ex.get("subject", ""),
            choices=tuple(ex["choices"]["text"]),
            choice_labels=tuple(ex["choices"]["label"]),
            correct_answer=ex["answerKey"].upper(),
        )
        for ex in ds
    ]
