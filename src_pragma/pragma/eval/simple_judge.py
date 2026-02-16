"""Simple rule-based judge for MCQA tasks."""

import logging
import re

from pragma.mcqa import LLMJudgeVerdict

logger = logging.getLogger(__name__)


class SimpleJudge:
    """Rule-based judge that extracts answers using regex patterns."""

    NO_RESPONSE = "NO_RESPONSE"

    def __init__(self):
        """Initialize the simple judge."""
        self.model = "simple-regex"

    def _extract_answer(self, text, valid_answers):
        """Extract answer from response text using regex patterns.

        Parameters
        ----------
        text : str
            The model's response text
        valid_answers : list[str]
            Valid answer labels (e.g., ['A', 'B', 'C', 'D'])

        Returns
        -------
        str
            Extracted answer or NO_RESPONSE
        """
        if not text:
            return self.NO_RESPONSE

        text_lower = text.lower()

        # Patterns to match, in order of specificity
        patterns = [
            r'(?:the\s+)?(?:correct\s+)?answer\s+is\s*[:\s]*([A-D])',
            r'(?:the\s+)?answer\s*[:\s]+([A-D])',
            r'(?:choose|select|pick)\s+([A-D])',
            r'([A-D])\s+is\s+(?:the\s+)?(?:correct|right)\s+answer',
            r'(?:therefore|thus|so),?\s+([A-D])',
            r'would\s+be\s+([A-D])\)',
            r'correct\s+answer\s+(?:would\s+be|is)\s+([A-D])',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).upper()
                if answer in valid_answers:
                    return answer

        # Fallback: look for last mentioned answer letter near end
        last_500 = text[-500:] if len(text) > 500 else text
        for answer in valid_answers:
            if answer == self.NO_RESPONSE:
                continue
            # Look for "A)" or "(A)" patterns
            if re.search(rf'\b{answer}\)', last_500):
                return answer

        return self.NO_RESPONSE

    def judge(self, response):
        """Evaluate a single QA response.

        Parameters
        ----------
        response : QAWithResponse
            The question/answer with model response

        Returns
        -------
        LLMJudgeVerdict
            The judgment verdict
        """
        valid_answers = list(
            set(response.qa.choice_labels) | {self.NO_RESPONSE}
        )

        # Get the response text (could be in reasoning or response field)
        text = response.reasoning or response.response or ""

        # Extract answer
        extracted = self._extract_answer(text, valid_answers)

        # Check correctness
        correct_answer = response.qa.correct_answer.strip().upper()
        correct = extracted == correct_answer

        return LLMJudgeVerdict(
            response=response,
            judge=self.model,
            correct=correct,
            extracted_answer=extracted,
            explanation=f"Extracted '{extracted}' from response",
            raw_response=None,
            error=None,
        )

    def judge_batch(self, batch, max_workers=None, log_interval=25, retry_failed=False):
        """Evaluate a batch of responses.

        Parameters
        ----------
        batch : list[QAWithResponse]
            The responses to evaluate
        max_workers : int
            Ignored (no parallelism needed)
        log_interval : int
            Log progress every N completions
        retry_failed : bool
            Ignored (no retries needed)

        Returns
        -------
        list[LLMJudgeVerdict]
            The verdicts for the batch
        """
        verdicts = []
        for i, response in enumerate(batch):
            verdict = self.judge(response)
            verdicts.append(verdict)

            if (i + 1) % log_interval == 0 or (i + 1) == len(batch):
                logger.debug("Judged: %d/%d", i + 1, len(batch))

        num_correct = sum(1 for v in verdicts if v.correct)
        logger.info(
            "Batch judging complete: %d/%d correct (%.1f%%)",
            num_correct,
            len(batch),
            num_correct / len(batch) * 100 if batch else 0,
        )

        return verdicts
