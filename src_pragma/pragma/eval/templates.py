"""Prompt templates for autoevaluation."""

import json
import re


class LLMJudgePrompt:
    """Standard prompt template for judging responses to multiple choice
    questions.
    """

    TEMPLATE = """You are evaluating a model's response to a multiple choice question.

Question: {question}

Choices:
{choices}

Correct Answer: {correct_answer}

Model's Response:
{model_response}

Your task:
1. Extract the chosen letter ({valid_labels}) from the model's response
2. Determine if it matches the correct answer. If the model's response is 
   {no_response}, the answer is always incorrect
3. Provide a brief explanation

Return the result by calling the tool submit_judgment with fields:

+ answer: one of {valid_labels}
+ explanation: short text

Do not output any additional text; use the tool only.
"""

    NO_RESPONSE = "NO_RESPONSE"

    def format(self, response, valid_answers):
        """Build the judging prompt.

        Parameters
        ----------
        response : QAWithResponse
            The response to evaluate
        valid_answers : list[str]
            Valid answer labels for this question

        Returns
        -------
        str
            The formatted prompt
        """
        choices_text = "\n".join(
            f"{label}) {choice}"
            for label, choice in zip(
                response.qa.choice_labels, response.qa.choices
            )
        )

        valid_labels = ", ".join(valid_answers)

        return self.TEMPLATE.format(
            question=response.qa.question,
            choices=choices_text,
            correct_answer=response.qa.correct_answer,
            model_response=response.response or self.NO_RESPONSE,
            valid_labels=valid_labels,
            no_response=self.NO_RESPONSE,
        )

    def extract_answer(self, raw_response):
        """Fallback parser for non-tool outputs: try JSON, then A-D.

        Parameters
        ----------
        raw_response : str
            The raw judge response

        Returns
        -------
        str
            Extracted answer (A/B/D/C) or empty string if not found
        """
        try:
            match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                ans = data.get("answer", "")
                return ans.upper().strip()

        except Exception:
            pass

        match = re.search(
            r"EXTRACTED_ANSWER:\s*([A-Z]+)", raw_response, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        return ""

    def extract_explanation(self, raw_response):
        """Fallback parser for non-tool outputs: try JSON, else return raw
        content.

        Parameters
        ----------
        raw_response : str
            The raw judge response

        Returns
        -------
        str
            Extracted explanation
        """
        try:
            match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                expl = data.get("explanation", "")
                return expl.strip()

        except Exception:
            pass

        return raw_response.strip()
