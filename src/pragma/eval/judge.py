"""Handles API calls to an LLM provider for autoevaluating CoT responses."""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import APIConnectionError, APIError, OpenAI

from pragma.mcqa import LLMJudgeVerdict

from .templates import LLMJudgePrompt

# Turn off INFO logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You are a strict judge. Return the result only by calling the tool. "
    "Do not output any prose."
)


class RateLimiter:
    """Rate limit API calls."""

    def __init__(self, max_calls_per_minute=40):
        """Initialize the RateLimiter.

        Parameters
        ----------
        max_calls_per_minute : int
            Maximum number of calls per minute
        """
        self.max_calls = max_calls_per_minute
        self.calls = []
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """Block if rate limit would be exceeded."""
        with self.lock:
            # Drop calls older than 60 seconds
            now = time.time()
            self.calls = [t for t in self.calls if now - t < 60]

            if len(self.calls) >= self.max_calls:
                sleep_time = 60 - (now - self.calls[0])
                if sleep_time > 0:
                    logger.debug(
                        "Rate limit reached, sleeping %.1fs", sleep_time
                    )
                    time.sleep(sleep_time)
                    self.calls = []

            self.calls.append(time.time())


class LLMJudge:
    """Judge that evaluates question/answer responses using an OpenAI-compatible API."""

    NO_RESPONSE = "NO_RESPONSE"

    def __init__(
        self,
        model,
        base_url,
        api_key,
        temperature=0.0,
        system_prompt=None,
        force_tool=True,
        max_calls_per_minute=50,
    ):
        """Initialize the judge.

        Parameters
        ----------
        model : str
            Model identifier
        base_url :
            Base URL for the endpoint
        api_key : str
            API key for authentication
        temperature : float
            Sampling temperature; 0.0 is deterministic judging
        system_prompt : str or None
            System prompt for the judge
        force_tool : bool
            If True, force the tool to be called; else allow auto selection
        max_calls_per_minute : int
            Maximum API calls per minute for rate limiting
        """
        # Client and rate limiter
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.rate_limiter = RateLimiter(max_calls_per_minute)

        # Generation info
        self.model = model
        self.temperature = temperature
        self.prompt_template = LLMJudgePrompt()

        # Templates and tools
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.force_tool = force_tool

    def _build_judgment_tool(self, valid_answers):
        """Build judgment tool dynamically based on valid answers.

        Parameters
        ----------
        valid_answers : list[str]
            Valid answer labels
        """
        return {
            "type": "function",
            "function": {
                "name": "submit_judgment",
                "description": "Return the extracted answer and explanation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "enum": valid_answers,
                        },
                        "explanation": {"type": "string"},
                    },
                    "required": ["answer", "explanation"],
                },
            },
        }

    def judge(self, response):
        """
        Evaluate a single QA response.

        Returns
        -------
        LLMJudgeVerdict
            The LLM's verdict
        """
        # Extract valid answers
        valid_answers = list(
            set(response.qa.choice_labels) | {self.NO_RESPONSE}
        )
        valid_answers.sort()

        try:
            self.rate_limiter.wait_if_needed()

            # Build tools and prompt with dynamic valid answers
            tools = [self._build_judgment_tool(valid_answers)]
            tool_choice = (
                {"type": "function", "function": {"name": "submit_judgment"}}
                if self.force_tool
                else "auto"
            )

            prompt = self.prompt_template.format(response, valid_answers)
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tools=tools,
                tool_choice=tool_choice,
                temperature=self.temperature,
            )

            msg = completion.choices[0].message
            answer, expl, parse_error = self._parse_judgment_from_message(
                msg, valid_answers
            )

            # Do a deterministic correctness check
            correct_answer = response.qa.correct_answer.strip().upper()
            extracted_norm = answer.strip().upper()
            correct = extracted_norm == correct_answer
            raw_response = msg.content

            return LLMJudgeVerdict(
                response=response,
                judge=self.model,
                correct=correct,
                extracted_answer=answer,
                explanation=expl,
                raw_response=raw_response,
                error=parse_error,
            )

        except (APIError, APIConnectionError) as e:
            logger.error("API error for question %s: %s", response.qa.id, e)
            return LLMJudgeVerdict(
                response=response,
                judge=self.model,
                correct=False,
                extracted_answer="",
                explanation="",
                raw_response=None,
                error=f"API error: {str(e)}",
            )

        except json.JSONDecodeError as e:
            logger.error(
                "JSON parsing error for question %s: %s", response.qa.id, e
            )
            return LLMJudgeVerdict(
                response=response,
                judge=self.model,
                correct=False,
                extracted_answer="",
                explanation="",
                raw_response=None,
                error=f"JSON decode error: {str(e)}",
            )

        except Exception as e:
            logger.exception(
                "Unexpected error for question %s", response.qa.id
            )
            return LLMJudgeVerdict(
                response=response,
                judge=self.model,
                correct=False,
                extracted_answer="",
                explanation="",
                raw_response=None,
                error=f"Unexpected error: {str(e)}",
            )

    def judge_batch(
        self, batch, max_workers=5, log_interval=25, retry_failed=True
    ):
        """Evaluate a batch of responses.

        Parameters
        ----------
        batch : list[QAWithResponse]
            The responses to evaluate
        max_workers : int
            Maximum concurrent API requests
        log_interval : int
            Log progress every N completions
        retry_failed : bool
            Whether to retry failed judgments once more

        Returns
        -------
        list[LLMJudgeVerdict]
            The LLM's verdict for the batch
        """
        if not batch:
            return []

        logger.debug(
            "Judging batch of %d responses with %d workers",
            len(batch),
            max_workers,
        )

        verdicts = [None] * len(batch)
        failed_indices = []
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.judge, response): idx
                for idx, response in enumerate(batch)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    verdict = future.result()
                    verdicts[idx] = verdict
                    if verdict.error:
                        failed_indices.append(idx)

                except Exception as e:
                    logger.error("Exception for response %d: %s", idx, e)
                    verdicts[idx] = LLMJudgeVerdict(
                        response=batch[idx],
                        judge=self.model,
                        correct=False,
                        extracted_answer="",
                        explanation="",
                        raw_response=None,
                        error=f"Batch error: {str(e)}",
                    )
                    failed_indices.append(idx)

                completed += 1
                if completed % log_interval == 0 or completed == len(batch):
                    logger.debug("Judged: %d/%d", completed, len(batch))

        # Retry failed judgments
        if retry_failed and failed_indices:
            logger.info("Retrying %d failed judgments", len(failed_indices))
            for idx in failed_indices:
                try:
                    verdicts[idx] = self.judge(batch[idx])
                except Exception as e:
                    logger.error("Retry failed for response %d: %s", idx, e)

        num_errors = sum(1 for v in verdicts if v.error)
        logger.info(
            "Batch judging complete: %d successful, %d errors",
            len(batch) - num_errors,
            num_errors,
        )

        return verdicts

    def _parse_judgment_from_message(self, msg, valid_answers):
        """Extract answer an explanation from a message object.

        Parameters
        ----------
        msg : ChatCompletionMessage
            The chat completion message
        valid_answers : list[str]
            Valid answer labels

        Returns
        -------
        tuple[str, str, str|None]
            Extracted answer, explanation, and parse_error
        """
        answer = ""
        expl = ""
        parse_error = None

        try:
            if getattr(msg, "tool_calls", None):
                tc = msg.tool_calls[0]
                args_str = (
                    tc.function.arguments
                    if hasattr(tc, "function")
                    else tc.arguments
                )
                args = (
                    json.loads(args_str)
                    if isinstance(args_str, str)
                    else args_str
                )
                answer = args.get("answer", "").strip().upper()
                expl = args.get("explanation", "").strip()

            else:
                logger.warning(
                    "Judge didn't call tool, falling back to content parsing"
                )

                content = msg.content or ""
                try:
                    data = json.loads(content)
                    answer = data.get("answer", "").strip().upper()
                    expl = data.get("explanation", "").strip()

                except Exception:
                    answer = self.prompt_template.extract_answer(content)
                    expl = self.prompt_template.extract_explanation(content)

            if answer not in valid_answers:
                parse_error = f"Invalid or missing answer: '{answer}'"
                logger.exception(parse_error)

        except Exception as e:
            parse_error = f"Parsing error: {e}"
            logger.exception(parse_error)

        return answer, expl, parse_error
