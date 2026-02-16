"""Custom LogitsProcessors."""

import torch
from transformers import LogitsProcessor


class NGramMaskingLogitsProcessor(LogitsProcessor):
    """Masks specified n-grams during reasoning generation segments.

    If `special_tokens` is None, the processor will mask n-grams for the entire
    input sequence.

    Reference implementation of NoRepeatNGramLogitsProcessor: https://github.com/huggingface/transformers/blob/307c5238546ba1675daabc46050c63ffde25f8e6/src/transformers/generation/logits_process.py#L1075
    """

    def __init__(
        self, ngrams, special_tokens=None, strategy="mask", penalty_weight=1.0
    ):
        """Initialize the processor.

        Strategy options:
            - "mask": n-gram logits are masked to -inf, thereby preventing the
              LLM from using them
            - "penalize": n-gram logits are penalized to reduce probability by
              `penalty_weight` factor. The probability of penalized n-grams will be
              multiplied by this factor (e.g., 0.5 = reduce to 50% of original
              probability)

        Parameters
        ----------
        ngrams : list[tuple[int, ...]]
            The n-grams to mask
        special_tokens : tuple[int, int] or None
            The special start/end tokens for reasoning
        strategy : str
            Masking strategy: "mask" or "penalize"
        penalty_weight : float
            Probability reduction factor when strategy is "penalize"

        Raises
        ------
        ValueError
            If the strategy isn't "mask" or "penalize" or if the penalty weight
            isn't 0 < weight <= 1.0
        """
        self.ngrams = ngrams
        self.special_tokens = special_tokens
        self.strategy = strategy
        self.penalty_weight = penalty_weight

        if self.strategy not in ("mask", "penalize"):
            raise ValueError("Invalid strategy, use 'mask' or 'penalize'")

        if self.strategy == "penalize" and not (0 < self.penalty_weight <= 1):
            raise ValueError("Penalty weight must be in range (0, 1])")

        # Precompute penalty: reducing probability by factor `penalty_weight` is
        # achieved by subracting -log(penalty_weight) from a logit
        if self.strategy == "penalize":
            self.penalty = -torch.log(torch.tensor(self.penalty_weight)).item()

        # Precompute a prefix: next-token mappings by n-gram length
        # For each n-gram of length L, map its prefix (first L-1) to the next
        # token (L-th). For L=1, prefix is the empty tuple (), meaning the
        # single token should be supressed unconditionally
        self.prefix_to_next_by_len = {}
        for ng in self.ngrams:
            L = len(ng)
            prefix = tuple(ng[:-1]) if L > 1 else tuple()
            next_tok = ng[-1]

            if L not in self.prefix_to_next_by_len:
                self.prefix_to_next_by_len[L] = {}

            if prefix not in self.prefix_to_next_by_len[L]:
                self.prefix_to_next_by_len[L][prefix] = set()

            self.prefix_to_next_by_len[L][prefix].add(next_tok)

    def _inside_reasoning(self, seq):
        """Determine if we are inside a reasoning segment.

        Strategy: find last occurence of start token, ensure there is no end
        token occuring after that start.

        Parameters
        ----------
        seq : torch.Tensor
            The sequence

        Returns
        -------
        bool
            Whether we're in a reasoning segment
        """
        # If we have no special tokens, apply to the entire sequence
        if self.special_tokens is None:
            return True

        seq = seq.view(-1)
        start_id, end_id = self.special_tokens

        # Get all start positions for reasoning
        start_indices = (seq == start_id).nonzero(as_tuple=False).view(-1)
        if start_indices.numel() == 0:
            return False

        # Check if any end token appears after that last start. We're in
        # reasoning only if there is no end after the last start
        last_start = int(start_indices[-1].item())
        end_after = (seq[last_start + 1 :] == end_id).any().item()

        return not end_after

    def _get_masked_tokens(self, seq, cur_len):
        """Get the set of tokens that should be masked based on n-gram
        prefixes.

        Parameters
        ----------
        seq : torch.Tensor
            The current sequence
        cur_len : int
            Current sequence length

        Returns
        -------
        set[int]
            Set of token IDs to ban
        """
        masked_tokens = set()

        for L, prefix_map in self.prefix_to_next_by_len.items():
            prefix_len = L - 1

            # For L=1, prefix_len=0, so we use empty tuple
            # For L>1, check if we have enough context
            if prefix_len > 0 and cur_len < prefix_len:
                continue

            if prefix_len > 0:
                prefix = tuple(seq[cur_len - prefix_len : cur_len].tolist())
            else:
                prefix = ()

            if prefix in prefix_map:
                masked_tokens.update(prefix_map[prefix])

        return masked_tokens

    def _apply_masking(self, scores, batch_idx, masked_tokens, vocab_size):
        """Apply the masking strategy to banned tokens.

        Parameters
        ----------
        scores : torch.Tensor
            The scores tensor to modify in-place
        batch_idx : int
            The batch index to apply masking to
        masked_tokens : set[int]
            Token IDs to mask
        vocab_size : int
            Size of vocabulary
        """
        # Ensure our banned tokens are valid vocabulary entries
        valid = [t for t in masked_tokens if 0 <= t < vocab_size]
        if not valid:
            return

        # Apply masking according to our strategy
        if self.strategy == "mask":
            scores[batch_idx, valid] = -float("inf")
        elif self.strategy == "penalize":
            scores[batch_idx, valid] -= self.penalty

    def __call__(self, input_ids, scores):
        """Apply the masking to a sequence.

        At each step, if we're inside reasoning, ban next tokens that would
        complete n-grams.

        For each ngram length L:
            - If L == 1: ban outright
            - If L >= 2: if the last L-1 tokens match an n-gram prefix, ban the
              corresponding next token

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs for a sequence, shape [B, S]
        scores : torch.Tensor, shape [B, V]
            Token logits

        Returns
        -------
        torch.Tensor
            Masked logits
        """
        batch_size = scores.size(0)
        vocab_size = scores.size(1)
        cur_len = input_ids.size(-1)

        scores_processed = scores.clone()

        for i in range(batch_size):
            seq = input_ids[i].view(-1)
            if not self._inside_reasoning(seq):
                # Don't mask outside reasoning steps
                continue

            # If we've found tokens to mask, apply masking strategy
            masked_tokens = self._get_masked_tokens(seq, cur_len)
            if masked_tokens:
                self._apply_masking(
                    scores_processed, i, masked_tokens, vocab_size
                )

        return scores_processed
