"""Ablation runner for managing CoT ablation experiments."""

import json
import logging
from pathlib import Path

from pragma.eval import (
    Metric,
    cot_is_better,
    process_distinctiveness,
    run_distinctiveness,
)
from pragma.results import (
    AblationRunResult,
    AblationStepResult,
    BaselineResult,
)
from pragma.tokenize import get_special_start_end

from .cotgen import run_batches, set_seed
from .processors import NGramMaskingLogitsProcessor

logger = logging.getLogger(__name__)


class AblationRunner:
    """Manages the ablation experiment lifecycle."""

    def __init__(
        self,
        examples,
        model,
        tokenizer,
        inference_config,
        judge,
        output_dir=None,
        resume=False,
    ):
        """Initialize the runner.

        Parameters
        ----------
        examples : list[MultipleChoiceQA]
            QA examples to process
        model : AutoModelForCausalLM
            The LLM
        tokenizer : AutoTokenizer
            The LLM tokenizer
        inference_config : InferenceConfig
            Inference configuration
        judge : LLMJudge
            Response judge
        output_dir : Path or None
            Directory to save incremental results. If None, no saving occurs
        resume : bool
            If True, resume from last completed step in output_dir

        Raises
        ------
        ValueError
            If resume is True but no output_dir is provided
        """
        self.examples = examples
        self.model = model
        self.tokenizer = tokenizer
        self.config = inference_config
        self.judge = judge

        # Extract run configs
        self.cot_off_run = self._get_run_config(enable_thinking=False)
        self.cot_on_run = self._get_run_config(enable_thinking=True)

        # Get special tokens for reasoning
        self.special_tokens = get_special_start_end(
            tokenizer,
            substring=inference_config.think_substring,
            token_format=inference_config.think_token_format,
        )

        # Set up a metric
        self.metric = Metric(inference_config.ablations.metric)

        # State tracking
        self.baseline = None
        self.ablated_ngrams = []
        self.ablation_steps = []
        self.current_step = 0

        # Save state initialization
        self.output_dir = Path(output_dir) if output_dir else None
        self.resume = resume

        # Auto-detect resume point if resuming
        if self.resume:
            if not self.output_dir:
                raise ValueError("Cannot resume without output_dir")

            self._load_state()

    def _get_run_config(self, enable_thinking):
        """Extract CoT-OFF/ON run configuration.

        Parameters
        ----------
        enable_thinking : bool
            Whether to get CoT-ON (True) or CoT-OFF (False) configuration

        Returns
        -------
        RunConfig
            The run configuration

        Raises
        ------
        ValueError
            If no run configuration is found for the specified mode
        """
        mode = "CoT-ON" if enable_thinking else "CoT-OFF"
        runs = [
            r for r in self.config.runs if r.enable_thinking == enable_thinking
        ]

        if not runs:
            raise ValueError(f"No {mode} run configuration found")

        if len(runs) > 1:
            logger.warning("Multiple %s runs found, using first", mode)

        return runs[0]

    def _load_state(self):
        """Load state from output directory to resume from last completed step.

        Raises
        ------
        ValueError
            If the output directory or its contents don't exist
        """
        if not self.output_dir.exists():
            raise ValueError(
                f"Output directory {self.output_dir} doesn't exist"
            )

        # Load the baseline
        baseline_path = self.output_dir / AblationRunResult._BASELINE_FILE
        if not baseline_path.exists():
            raise ValueError("No BaselineResult found in output directory")

        with baseline_path.open("r") as f:
            self.baseline = BaselineResult.from_dict(json.load(f))

        # Auto-detect the last completed step by finding all step files
        step_num = 0
        while True:
            step_file = (
                self.output_dir
                / AblationRunResult._STEP_FILE_STUB.format(step=step_num)
            )
            if not step_file.exists():
                break

            with step_file.open("r") as f:
                step_result = AblationStepResult.from_dict(json.load(f))
                self.ablation_steps.append(step_result)

            step_num += 1

        # Set current state based on last completed step
        if self.ablation_steps:
            last_step = self.ablation_steps[-1]
            self.ablated_ngrams = list(last_step.ablated_ngrams) + list(
                last_step.newly_found_ngrams
            )
            self.current_step = len(self.ablation_steps)

            logger.info(
                "Resuming from step %d with %d ablated n-grams "
                "(last completed step: %d)",
                self.current_step,
                len(self.ablated_ngrams),
                last_step.step,
            )
        else:
            self.current_step = 0
            logger.info("No completed steps found. Starting from step 0")

    def _save_baseline(self):
        """Save baseline result."""
        if not self.output_dir:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        baseline_file = self.output_dir / AblationRunResult._BASELINE_FILE
        with baseline_file.open("w") as f:
            json.dump(self.baseline.to_dict(), f, indent=2)

        logger.debug("Saved baseline to %s", baseline_file)

    def _save_step(self, step_result):
        """Save a single step result immediately after completion.

        Parameters
        ----------
        step_result : AblationStepResult
            The ablation step result
        """
        if not self.output_dir:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        step_file = self.output_dir / AblationRunResult._STEP_FILE_STUB.format(
            step=step_result.step
        )
        with step_file.open("w") as f:
            json.dump(step_result.to_dict(), f, indent=2)

        logger.debug("Saved step %d to %s", step_result.step, step_file)

    def run(self):
        """Run the ablations.

        Returns
        -------
        AblationRunResult
            Complete results from the ablation run
        """
        set_seed(self.config.seed)

        # Only run baseline if not resuming
        if self.baseline is None:
            self.baseline = self._run_baseline()
            self._save_baseline()

        termination_reason = self._run_ablation_loop()

        return self._build_result(termination_reason)

    def _run_baseline(self):
        """Run CoT-OFF baseline and judge responses.

        Returns
        -------
        BaselineResult
            Baseline results
        """
        logger.info("Generating CoT-OFF answers")

        responses = run_batches(
            self.examples,
            model=self.model,
            tokenizer=self.tokenizer,
            inference_config=self.config,
            batch_size=self.config.batch_size,
            enable_thinking=self.cot_off_run.enable_thinking,
            generation_kwargs=self.cot_off_run.generation_kwargs,
        )

        logger.info("Judging %d CoT-OFF responses", len(responses))
        verdicts = self.judge.judge_batch(responses)

        num_correct = sum(v.correct for v in verdicts)
        num_total = len(verdicts)
        accuracy = (num_correct / num_total) if num_total > 0 else 0.0

        logger.info(
            "Baseline (CoT-OFF): %d/%d correct (%.2f%%)",
            num_correct,
            num_total,
            accuracy * 100,
        )

        return BaselineResult(
            responses=tuple(responses),
            verdicts=tuple(verdicts),
            num_correct=num_correct,
            num_total=num_total,
            accuracy=accuracy,
        )

    def _run_ablation_loop(self):
        """Run iterative ablation steps until termination, using optional
        incremental saving.

        Returns
        -------
        str
            Termination reason
        """
        logger.info(
            "Beginning ablations. Using masking strategy '%s' (penalty: %.2f)",
            self.config.ablations.strategy,
            self.config.ablations.penalty_weight,
        )

        while self.current_step < self.config.ablations.max_steps:
            step_result, should_continue, reason = self._process_step()

            self._save_step(step_result)
            self.ablation_steps.append(step_result)

            if not should_continue:
                return reason

            # Update state for the next iteration
            self.ablated_ngrams.extend(step_result.newly_found_ngrams)
            self.current_step += 1

            logger.info(
                "Total n-grams for next step: %d", len(self.ablated_ngrams)
            )

        return "max_steps"

    def _process_step(self):
        """Process a single ablation step.

        Returns
        -------
        tuple[AblationStepResult, bool, str]
            Step result, whether to continue, and termination reason
        """
        logger.info(
            "CoT-ON ablation step: %d. Running %d examples with %d "
            "ablated n-grams",
            self.current_step,
            len(self.examples),
            len(self.ablated_ngrams),
        )

        responses, verdicts = self._generate_and_judge(self.examples)

        # Calculate metrics
        current_correct = sum(v.correct for v in verdicts)
        current_total = len(verdicts)
        current_accuracy = (
            (current_correct / current_total) if current_total > 0 else 0.0
        )
        self._log_accuracy_comparison(
            current_correct, current_total, current_accuracy
        )

        is_better = cot_is_better(self.baseline.verdicts, verdicts)
        logger.info("Performance: %s", "Better" if is_better else "Worse")

        # Distinctiveness: use all examples for cot_off/cot_on comparison
        distinct_results = run_distinctiveness(
            cot_off=self.baseline.responses,
            cot_on=responses,
            tokenizer=self.tokenizer,
            metric=self.metric,
            ngram_range=self.config.ablations.ngram_range,
        )

        if len(distinct_results) == 0:
            result = self._create_step_result(
                verdicts,
                is_better,
                current_correct,
                current_total,
                current_accuracy,
                [],
                [],
                0.0,
                {},
            )
            return result, False, "no_grams"

        distinctiveness_results, new_ngrams, cutoff, score_stats = (
            process_distinctiveness(
                distinct_results,
                cutoff_quantile=self.config.ablations.cutoff,
                top_k=self.config.ablations.top_k,
            )
        )

        if len(new_ngrams) == 0:
            result = self._create_step_result(
                verdicts,
                is_better,
                current_correct,
                current_total,
                current_accuracy,
                distinctiveness_results,
                [],
                cutoff,
                score_stats,
            )
            return result, False, "no_grams_above_cutoff"

        logger.info(
            "Found %d new n-grams above cutoff %.3f", len(new_ngrams), cutoff
        )

        result = self._create_step_result(
            verdicts,
            is_better,
            current_correct,
            current_total,
            current_accuracy,
            distinctiveness_results,
            new_ngrams,
            cutoff,
            score_stats,
        )

        # Always continue collecting distinctive tokens regardless of performance.
        # The ablation loop stops via max_steps or when no new n-grams are found,
        # not because CoT is currently worse than baseline.  (A degraded run is
        # still scientifically interesting — we want the tokens even at 0 % acc.)
        should_continue = True
        reason = "" if is_better else "degraded_but_continuing"

        return result, should_continue, reason

    def _generate_and_judge(self, examples):
        """Generate responses with ablation and judge them.

        Parameters
        ----------
        examples : list[MultipleChoiceQA]
            QA examples to process

        Returns
        -------
        tuple[list[QAWithResponse], list[LLMJudgeVerdict]]
            Responses from the LLM and judge verdicts
        """
        logits_processor = NGramMaskingLogitsProcessor(
            self.ablated_ngrams,
            special_tokens=self.special_tokens,
            strategy=self.config.ablations.strategy,
            penalty_weight=self.config.ablations.penalty_weight,
        )

        responses = run_batches(
            examples,
            model=self.model,
            tokenizer=self.tokenizer,
            inference_config=self.config,
            batch_size=self.config.batch_size,
            enable_thinking=self.cot_on_run.enable_thinking,
            generation_kwargs=self.cot_on_run.generation_kwargs,
            logits_processors=[logits_processor],
        )

        logger.info("Judging %d CoT-ON responses", len(responses))
        verdicts = self.judge.judge_batch(responses)

        return responses, verdicts

    def _log_accuracy_comparison(
        self, current_correct, current_total, current_accuracy
    ):
        """Log accuracy comparison between baseline and current step.

        Parameters
        ----------
        current_correct : int
            Number of correct responses
        current_total : int
            Total responses
        current_accuracy : float
            Accuracy of correct responses
        """
        logger.info(
            "Baseline (CoT-OFF): %d/%d correct (%.2f%%)",
            self.baseline.num_correct,
            self.baseline.num_total,
            self.baseline.accuracy * 100,
        )

        logger.info(
            "Current (CoT-ON): %d/%d correct (%.2f%%)",
            current_correct,
            current_total,
            current_accuracy * 100,
        )

        accuracy_delta = current_accuracy - self.baseline.accuracy
        correct_delta = current_correct - self.baseline.num_correct
        logger.info(
            "Delta: %+.2f%% (%+d questions)",
            accuracy_delta * 100,
            correct_delta,
        )

    def _create_step_result(
        self,
        verdicts,
        is_better,
        current_correct,
        current_total,
        current_accuracy,
        distinctiveness_results,
        new_ngrams,
        cutoff,
        score_stats,
    ):
        """Create an AblationStepResult.

        Parameters
        ----------
        verdicts : list[LLMJudgeVerdict]
            All verdicts for this step
        is_better : bool
            Whether CoT-ON performance is better than baseline
        current_correct : int
            Number of correct answers in this step
        current_total : int
            Total number of questions in this step
        current_accuracy : float
            Accuracy for this step
        distinctiveness_results : list
            Distinctiveness analysis results
        new_ngrams : list[tuple]
            Newly found n-grams above cutoff threshold
        cutoff : float
            Cutoff threshold used for n-gram selection
        score_stats : dict
            Dictionary containing score statistics (min, max, mean, median)

        Returns
        -------
        AblationStepResult
            Complete results for this ablation step
        """
        return AblationStepResult(
            step=self.current_step,
            verdicts=tuple(verdicts),
            ablated_ngrams=tuple(self.ablated_ngrams),
            newly_found_ngrams=tuple(new_ngrams),
            distinctiveness_results=tuple(distinctiveness_results),
            is_better=is_better,
            num_correct=current_correct,
            num_total=current_total,
            accuracy=current_accuracy,
            accuracy_delta=current_accuracy - self.baseline.accuracy,
            cutoff=cutoff,
            score_min=score_stats.get("min", 0.0),
            score_max=score_stats.get("max", 0.0),
            score_mean=score_stats.get("mean", 0.0),
            score_median=score_stats.get("median", 0.0),
        )

    def _build_result(self, termination_reason):
        """Build the final AblationRunResult.

        Parameters
        ----------
        termination_reason : str
            The termination reason

        Returns
        -------
        AblationRunResult
            Result from the ablation run
        """
        total_ngrams = (
            len(self.ablation_steps[-1].ablated_ngrams)
            + len(self.ablation_steps[-1].newly_found_ngrams)
            if self.ablation_steps
            else 0
        )
        final_accuracy = (
            self.ablation_steps[-1].accuracy
            if self.ablation_steps
            else self.baseline.accuracy
        )
        final_accuracy_delta = (
            self.ablation_steps[-1].accuracy_delta
            if self.ablation_steps
            else 0.0
        )

        logger.info(
            "Ablation complete after %d steps. Ablated %d n-grams",
            self.current_step,
            total_ngrams,
        )

        return AblationRunResult(
            model=self.config.model,
            judge=self.config.ablations.judge,
            num_examples=len(self.examples),
            seed=self.config.seed,
            cutoff=self.config.ablations.cutoff,
            max_steps=self.config.ablations.max_steps,
            ngram_range=self.config.ablations.ngram_range,
            metric=self.config.ablations.metric,
            strategy=self.config.ablations.strategy,
            penalty_weight=self.config.ablations.penalty_weight,
            top_k=self.config.ablations.top_k,
            baseline=self.baseline,
            ablation_steps=tuple(self.ablation_steps),
            total_steps_completed=len(self.ablation_steps),
            total_ngrams_ablated=total_ngrams,
            termination_reason=termination_reason,
            final_accuracy=final_accuracy,
            final_accuracy_delta=final_accuracy_delta,
        )
