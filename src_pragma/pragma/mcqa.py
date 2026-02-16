"""Multiple choice question/answer pair dataclasses."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MultipleChoiceQA:
    """Multiple choice question/answer pair.

    For use with datasets like ARC-Challenge or MMLU.

    Parameters
    ----------
    id : str
        Dataset id for the question/answer pair
    question : str
        Question
    subject : str
        Subject of the question
    choices : tuple[str, ...]
        Multiple-choice answers
    choice_labels : tuple[str, ...]
        Multiple-choice answer labels (e.g., 'a', 'b', 'c', 'd')
    correct_answer : str
        Correct answer (provided as label)
    """

    id: str
    question: str
    subject: str
    choices: tuple[str, ...]
    choice_labels: tuple[str, ...]
    correct_answer: str

    def __hash__(self):
        """Hashing function for sorting."""
        return hash(self.id)

    def __post_init__(self):
        """Validate data after initialization.

        Raises
        ------
        ValueError
            - If there's a length mismatch between choices and choice labels
            - If the (normalized) answer isn't in (normalized) labels
        """
        # Validate matching lengths
        if len(self.choices) != len(self.choice_labels):
            raise ValueError("Number of choices and choice labels don't match")

        # Normalize and validate correct answer
        normalized_answer = self.correct_answer.strip().upper()
        normalized_labels = [
            label.strip().upper() for label in self.choice_labels
        ]

        if normalized_answer not in normalized_labels:
            raise ValueError("Correct answer not in choice labels")

    @classmethod
    def from_dict(cls, data):
        """Reconstruct from dictionary.

        Parameters
        ----------
        data : dict
            Serialized MultipleChoiceQA

        Returns
        -------
        MultipleChoiceQA
            Reconstructed instance
        """
        return cls(
            id=data["id"],
            question=data["question"],
            subject=data["subject"],
            choices=tuple(data["choices"]),
            choice_labels=tuple(data["choices_labels"]),
            correct_answer=data["correct_answer"],
        )

    def to_prompt(self):
        """Format the question into a prompt with numbered choices.

        Returns
        -------
        str
            The formatted question with choices
        """
        if not self.choices:
            return self.question

        formatted = "\n".join(
            f"{label}) {choice}"
            for label, choice in zip(self.choice_labels, self.choices)
        )

        return f"{self.question}\n\n{formatted}".strip()


@dataclass(frozen=True)
class QAWithResponse:
    """Multiple choice question/answer pair with LLM reasoning/response.

    Parameters
    ----------
    qa : MultipleChoiceQA
        A question/answer pair
    reasoning : str
        The LLM's reasoning trace
    response : str
        The LLM's response
    reasoning_enabled : bool
        If true, this is a CoT-ON response
    """

    qa: MultipleChoiceQA
    reasoning: str = ""
    response: str = ""
    reasoning_enabled: bool = False

    def __hash__(self):
        """Hashing function for sorting."""
        return hash(self.qa.id)

    @classmethod
    def from_dict(cls, data):
        """Reconstruct from dictionary.

        Parameters
        ----------
        data : dict
            Serialized QAWithResponse

        Returns
        -------
        QAWithResponse
            Reconstructed instance
        """
        qa = MultipleChoiceQA(
            id=data["id"],
            question=data["question"],
            subject=data["subject"],
            choices=tuple(data["choices"]),
            choice_labels=tuple(data["choices_labels"]),
            correct_answer=data["correct_answer"],
        )

        return cls(
            qa=qa,
            reasoning=data.get("reasoning", ""),
            response=data.get("response", ""),
            reasoning_enabled=data.get("reasoning_enabled", False),
        )

    def to_dict(self):
        """Convert to dictionary for serialization.

        Returns
        -------
        dict
            Dictionary representation
        """
        return {
            "id": self.qa.id,
            "question": self.qa.question,
            "subject": self.qa.subject,
            "choices": list(self.qa.choices),
            "choices_labels": list(self.qa.choice_labels),
            "correct_answer": self.qa.correct_answer,
            "prompt": self.qa.to_prompt(),
            "reasoning": self.reasoning,
            "response": self.response,
            "reasoning_enabled": self.reasoning_enabled,
        }


@dataclass(frozen=True)
class LLMJudgeVerdict:
    """Verdict from an LLM judge.

    Parameters
    ----------
    response : QAWithResponse
        The QA pair with an LLM's response
    judge : str
        Name of the LLM that evaluated the response
    correct : bool
        If true, the response is correct
    extracted_answer : str
        Extracted answer span from the LLM response
    explanation : str
        The judge's explanation for why the response is correct or incorrect
    raw_response : str | None
        All response text from the LLM; used when extracting the answer span
        fails
    error : str | None
        Error message if the verdict fails
    """

    response: QAWithResponse
    judge: str
    correct: bool
    extracted_answer: str
    explanation: str
    raw_response: str | None = None
    error: str | None = None

    def __hash__(self):
        """Hashing function for sorting."""
        return hash(self.response.qa.id)

    @classmethod
    def from_dict(cls, data):
        """Reconstruct from dictionary.

        Parameters
        ----------
        data : dict
            Serialized LLMJudgeVerdict

        Returns
        -------
        LLMJudgeVerdict
            Reconstructed instance
        """
        response = QAWithResponse.from_dict(data)

        return cls(
            response=response,
            judge=data["judge"],
            correct=data["correct"],
            extracted_answer=data["extracted_answer"],
            explanation=data["explanation"],
            raw_response=data.get("raw_response"),
            error=data.get("error"),
        )

    def to_dict(self):
        """Convert to dictionary for serialization.

        Returns
        -------
        dict
            Dictionary representation
        """
        return {
            "id": self.response.qa.id,
            "question": self.response.qa.question,
            "subject": self.response.qa.subject,
            "choices": list(self.response.qa.choices),
            "choices_labels": list(self.response.qa.choice_labels),
            "correct_answer": self.response.qa.correct_answer,
            "prompt": self.response.qa.to_prompt(),
            "reasoning": self.response.reasoning,
            "response": self.response.response,
            "reasoning_enabled": self.response.reasoning_enabled,
            "judge": self.judge,
            "correct": self.correct,
            "extracted_answer": self.extracted_answer,
            "explanation": self.explanation,
            "raw_response": self.raw_response,
            "error": self.error,
        }
