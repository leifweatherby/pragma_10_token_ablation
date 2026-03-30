"""General tokenization utilities."""

import torch
import torch.nn.functional as F


def tokenize(
    prompts,
    tokenizer,
    enable_thinking=False,
    device=None,
):
    """Tokenize prompts with optional thinking mode and left-padding.

    Parameters
    ----------
    prompts : str or list[str]
        Prompts to tokenize
    tokenizer : AutoTokenizer
        The LLM tokenizer
    enable_thinking : bool
        Whether to enable thinking mode
    device : str or torch.device or None
        Model device

    Returns
    -------
    dict[str, torch.Tensor]
        Input IDs and attention masks

    Raises
    ------
    ValueError
        If the tokenizer doesn't have a PAD or EOS token
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no PAD or EOS token")

        tokenizer.pad_token = tokenizer.eos_token

    batch_msg = [[{"role": "user", "content": prompt}] for prompt in prompts]

    batch_ids = []
    for msg in batch_msg:
        result = tokenizer.apply_chat_template(
            msg,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            return_tensors="pt",
        )
        # Handle both tensor and dict returns from apply_chat_template
        if isinstance(result, dict):
            input_ids = result['input_ids']
        elif hasattr(result, 'input_ids'):
            # Handle dict-like objects (BatchEncoding, etc.)
            input_ids = result.input_ids
        else:
            input_ids = result
        batch_ids.append(input_ids.squeeze(0))

    # Left-pad to max sequence length in batch
    max_len = max(ids.shape[0] for ids in batch_ids)
    padded_ids = []
    for ids in batch_ids:
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            padded = F.pad(ids, (pad_len, 0), value=tokenizer.pad_token_id)
        else:
            padded = ids
        padded_ids.append(padded)

    input_ids = torch.stack(padded_ids)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

    return {"input_ids": input_ids, "attention_mask": attention_mask}
