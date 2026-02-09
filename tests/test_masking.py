"""Tests for NGramMaskingLogitsProcessor."""

import pytest
import torch

from pragma.generate.processors import NGramMaskingLogitsProcessor


class TestNGramMaskingLogitsProcessor:
    """Test suite for NGramMaskingLogitsProcessor."""

    def test_init_valid_block_strategy(self):
        """Test initialization with valid 'block' strategy."""
        ngrams = [(1, 2), (3, 4, 5)]
        processor = NGramMaskingLogitsProcessor(
            ngrams=ngrams, strategy="mask"
        )
        assert processor.strategy == "mask"
        assert processor.ngrams == ngrams

    def test_init_valid_penalty_strategy(self):
        """Test initialization with valid 'penalize' strategy."""
        ngrams = [(1, 2)]
        processor = NGramMaskingLogitsProcessor(
            ngrams=ngrams, strategy="penalize", penalty_weight=0.5
        )
        assert processor.strategy == "penalize"
        assert processor.penalty_weight == 0.5

        # Penalty should be -log(0.5) ≈ 0.693
        assert pytest.approx(processor.penalty, 0.01) == 0.693

    def test_init_invalid_strategy_raises_error(self):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            NGramMaskingLogitsProcessor(ngrams=[(1, 2)], strategy="invalid")

    def test_init_invalid_penalty_weight_raises_error(self):
        """Test that invalid reduction rate raises ValueError."""
        with pytest.raises(ValueError, match="Penalty weight must be in range"):
            NGramMaskingLogitsProcessor(
                ngrams=[(1, 2)], strategy="penalize", penalty_weight=0.0
            )

        with pytest.raises(ValueError, match="Penalty weight must be in range"):
            NGramMaskingLogitsProcessor(
                ngrams=[(1, 2)], strategy="penalize", penalty_weight=1.5
            )

    def test_prefix_to_next_mapping_unigram(self):
        """Test that unigrams are mapped correctly."""
        ngrams = [(5,), (10,)]
        processor = NGramMaskingLogitsProcessor(ngrams=ngrams)

        assert 1 in processor.prefix_to_next_by_len
        assert () in processor.prefix_to_next_by_len[1]
        assert processor.prefix_to_next_by_len[1][()] == {5, 10}

    def test_prefix_to_next_mapping_bigram(self):
        """Test that bigrams are mapped correctly."""
        ngrams = [(1, 2), (1, 3)]
        processor = NGramMaskingLogitsProcessor(ngrams=ngrams)

        assert 2 in processor.prefix_to_next_by_len
        assert (1,) in processor.prefix_to_next_by_len[2]
        assert processor.prefix_to_next_by_len[2][(1,)] == {2, 3}

    def test_prefix_to_next_mapping_trigram(self):
        """Test that trigrams are mapped correctly."""
        ngrams = [(1, 2, 3), (1, 2, 4)]
        processor = NGramMaskingLogitsProcessor(ngrams=ngrams)

        assert 3 in processor.prefix_to_next_by_len
        assert (1, 2) in processor.prefix_to_next_by_len[3]
        assert processor.prefix_to_next_by_len[3][(1, 2)] == {3, 4}

    def test_inside_reasoning_no_special_tokens(self):
        """Test _inside_reasoning when no special tokens are specified."""
        processor = NGramMaskingLogitsProcessor(
            ngrams=[(1, 2)], special_tokens=None
        )
        seq = torch.tensor([1, 2, 3, 4, 5])
        assert processor._inside_reasoning(seq) is True

    def test_inside_reasoning_with_start_no_end(self):
        """Test _inside_reasoning when inside a reasoning segment."""
        START_TOKEN = 100
        END_TOKEN = 101
        processor = NGramMaskingLogitsProcessor(
            ngrams=[(1, 2)], special_tokens=(START_TOKEN, END_TOKEN)
        )
        seq = torch.tensor([1, 2, START_TOKEN, 3, 4, 5])
        assert processor._inside_reasoning(seq) is True

    def test_inside_reasoning_with_start_and_end(self):
        """Test _inside_reasoning when reasoning segment is closed."""
        START_TOKEN = 100
        END_TOKEN = 101
        processor = NGramMaskingLogitsProcessor(
            ngrams=[(1, 2)], special_tokens=(START_TOKEN, END_TOKEN)
        )
        seq = torch.tensor([1, 2, START_TOKEN, 3, 4, END_TOKEN, 5])
        assert processor._inside_reasoning(seq) is False

    def test_inside_reasoning_no_start_token(self):
        """Test _inside_reasoning when no start token is present."""
        START_TOKEN = 100
        END_TOKEN = 101
        processor = NGramMaskingLogitsProcessor(
            ngrams=[(1, 2)], special_tokens=(START_TOKEN, END_TOKEN)
        )
        seq = torch.tensor([1, 2, 3, 4, 5])
        assert processor._inside_reasoning(seq) is False

    def test_mask_strategy_masking_unigram(self):
        """Test 'mask' masking strategy with unigrams."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])

        processor = NGramMaskingLogitsProcessor(
            ngrams=[(50,)], strategy="mask"
        )

        scores = torch.zeros(1, 100)
        scores[0, 50] = 5.0

        output_scores = processor(input_ids, scores)

        assert output_scores[0, 50] == -float("inf")
        assert output_scores[0, 49] == 0.0

    def test_mask_strategy_masking_bigram(self):
        """Test 'mask' masking strategy with bigrams."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])

        processor = NGramMaskingLogitsProcessor(
            ngrams=[(9, 10)], strategy="mask"
        )

        scores = torch.zeros(1, 100)
        scores[0, 10] = 5.0

        output_scores = processor(input_ids, scores)

        assert output_scores[0, 10] == -float("inf")

    def test_mask_strategy_masking_trigram(self):
        """Test 'mask' masking strategy with trigrams."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])

        processor = NGramMaskingLogitsProcessor(
            ngrams=[(8, 9, 10)], strategy="mask"
        )

        scores = torch.zeros(1, 100)
        scores[0, 10] = 5.0

        output_scores = processor(input_ids, scores)

        assert output_scores[0, 10] == -float("inf")

    def test_penalize_strategy_masking_penalizes_logits(self):
        """Test 'penalize' strategy penalizes logits by correct amount."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])

        processor = NGramMaskingLogitsProcessor(
            ngrams=[(9, 10)], strategy="penalize", penalty_weight=0.5
        )

        scores = torch.zeros(1, 100)
        scores[0, 10] = 5.0

        output_scores = processor(input_ids, scores)

        expected_score = 5.0 - processor.penalty
        assert (
            pytest.approx(output_scores[0, 10].item(), 0.01) == expected_score
        )

    def test_no_masking_outside_reasoning(self):
        """Test that no masking occurs outside reasoning segments."""
        START_TOKEN = 100
        END_TOKEN = 101

        input_ids = torch.tensor([[START_TOKEN, 1, 2, END_TOKEN, 3]])

        processor = NGramMaskingLogitsProcessor(
            ngrams=[(3, 4)],
            special_tokens=(START_TOKEN, END_TOKEN),
            strategy="mask",
        )

        scores = torch.zeros(1, 100)
        scores[0, 4] = 5.0

        output_scores = processor(input_ids, scores)

        assert output_scores[0, 4] == 5.0

    def test_masking_inside_reasoning(self):
        """Test that masking occurs inside reasoning segments."""
        START_TOKEN = 100
        END_TOKEN = 101

        input_ids = torch.tensor([[START_TOKEN, 1, 2, 3]])

        processor = NGramMaskingLogitsProcessor(
            ngrams=[(3, 4)],
            special_tokens=(START_TOKEN, END_TOKEN),
            strategy="mask",
        )

        scores = torch.zeros(1, 100)
        scores[0, 4] = 5.0

        output_scores = processor(input_ids, scores)

        assert output_scores[0, 4] == -float("inf")

    def test_batch_processing(self):
        """Test that processor handles batched inputs correctly."""
        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9], [11, 12, 13, 14, 15, 16, 17, 18, 19]]
        )

        processor = NGramMaskingLogitsProcessor(
            ngrams=[(9, 10), (19, 20)], strategy="mask"
        )

        scores = torch.zeros(2, 100)
        scores[0, 10] = 5.0
        scores[1, 20] = 5.0

        output_scores = processor(input_ids, scores)

        assert output_scores[0, 10] == -float("inf")
        assert output_scores[1, 20] == -float("inf")

    def test_no_masking_when_prefix_doesnt_match(self):
        """Test that tokens are not masked when prefix doesn't match."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])

        processor = NGramMaskingLogitsProcessor(
            ngrams=[(7, 10)], strategy="mask"
        )

        scores = torch.zeros(1, 100)
        scores[0, 10] = 5.0

        output_scores = processor(input_ids, scores)

        assert output_scores[0, 10] == 5.0

    def test_multiple_ngrams_same_next_token(self):
        """Test handling multiple n-grams that ban the same next token."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])

        processor = NGramMaskingLogitsProcessor(
            ngrams=[(9, 10), (8, 9, 10)], strategy="mask"
        )

        scores = torch.zeros(1, 100)
        scores[0, 10] = 5.0

        output_scores = processor(input_ids, scores)

        assert output_scores[0, 10] == -float("inf")

    def test_vocab_boundary_checking(self):
        """Test that tokens outside vocab range are not processed."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        processor = NGramMaskingLogitsProcessor(
            ngrams=[(5, 500)], strategy="mask"
        )

        scores = torch.zeros(1, 100)

        output_scores = processor(input_ids, scores)

        assert torch.all(output_scores == scores)

    def test_empty_ngrams_list(self):
        """Test processor with empty n-grams list."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        processor = NGramMaskingLogitsProcessor(ngrams=[])

        scores = torch.zeros(1, 100)
        original_scores = scores.clone()

        output_scores = processor(input_ids, scores)

        assert torch.all(output_scores == original_scores)

    def test_short_sequence_with_long_ngram(self):
        """Test that long n-grams don't cause issues with short sequences."""
        input_ids = torch.tensor([[1, 2]])

        processor = NGramMaskingLogitsProcessor(
            ngrams=[(1, 2, 3, 4, 5)], strategy="mask"
        )

        scores = torch.zeros(1, 100)
        scores[0, 3] = 5.0

        output_scores = processor(input_ids, scores)

        assert output_scores[0, 3] == 5.0
