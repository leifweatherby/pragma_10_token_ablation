"""CoT generation results dataclasses."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from .mcqa import LLMJudgeVerdict, QAWithResponse


@dataclass(frozen=True)
class DistinctivenessResult:
    """Results from distinctiveness analysis.

    Parameters
    ----------
    ngram : tuple[int, ...]
        Token IDs of the n-gram
    score : float
        Distinctiveness score (e.g. fighting words)
    rank : int
        N-gram rank for score vis-a-vis other distinctive n-grams
    above_cutoff : bool
        If true, n-gram will be ablated; see AblationRunResult
    """

    ngram: tuple[int, ...]
    score: float
    rank: int
    above_cutoff: bool

    @classmethod
    def from_dict(cls, data):
        """Reconstruct from dictionary.

        Parameters
        ----------
        data : dict
            Serialized DistinctivenessResult

        Returns
        -------
        DistinctivenessResult
            Reconstructed instance
        """
        return cls(
            ngram=tuple(data["ngram"]),
            score=data["score"],
            rank=data["rank"],
            above_cutoff=data["above_cutoff"],
        )

    def to_dict(self):
        """Convert to dictionary.

        Returns
        -------
        dict
            Serialized DistinctivenessResult
        """
        return {
            "ngram": self.ngram,
            "score": self.score,
            "rank": self.rank,
            "above_cutoff": self.above_cutoff,
        }


@dataclass(frozen=True)
class AblationStepResult:
    """Results from a single ablation step.

    Parameters
    ----------
    step : int
        Ablation step number
    verdicts : tuple[LLMJudgeVerdict, ...]
        Judged QA pairs; see LLMJudgeVerdict
    ablated_ngrams : tuple[tuple[int, ...]]
        Token IDs of all ablated n-grams
    newly_found_ngrams : tuple[tuple[int, ...]]
        Token IDs of each new n-gram that is above the metric cutoff for this step
    disinctveness_results : tuple[DistinctivenessResult, ...]
        Distinctiveness data for each n-gram
    is_better : bool
        If true, the net accuracy of all verdicts exceeds net CoT-OFF baseline
    num_correct : int
        Number of correct CoT-ON responses
    num_total : int
        Number of verdicts
    accuracy : float
        Accuracy of CoT-ON responses
    accuracy_delta : float
        Accuracy of CoT-ON responses - accuracy of CoT-OFF responses
    cutoff : float
        Cutoff value of the metric; n-grams with score >= cutoff will be
        ablated
    score_min : float
        Minimum metric score for all n-grams
    score_max : float
        Maximum metric score for all n-grams
    score_mean : float
        Mean metric score for all n-grams
    score_median : float
        Median metric score for all n-grams
    """

    step: int
    verdicts: tuple[LLMJudgeVerdict, ...]
    ablated_ngrams: tuple[tuple[int, ...]]
    newly_found_ngrams: tuple[tuple[int, ...]]
    distinctiveness_results: tuple[DistinctivenessResult, ...]

    # Performance metrics
    is_better: bool
    num_correct: int
    num_total: int
    accuracy: float
    accuracy_delta: float

    # Distinctiveness metrics
    cutoff: float
    score_min: float
    score_max: float
    score_mean: float
    score_median: float

    @classmethod
    def from_dict(cls, data):
        """Reconstruct from dictionary.

        Parameters
        ----------
        data : dict
            Serialized AblationStepResult

        Returns
        -------
        AblationStepResult
            Reconstructed instance
        """
        perf = data["performance"]
        metrics = data["distinctiveness_metrics"]

        return cls(
            step=data["step"],
            verdicts=tuple(
                LLMJudgeVerdict.from_dict(v) for v in data["verdicts"]
            ),
            ablated_ngrams=tuple(tuple(ng) for ng in data["ablated_ngrams"]),
            newly_found_ngrams=tuple(
                tuple(ng) for ng in data["newly_found_ngrams"]
            ),
            distinctiveness_results=tuple(
                DistinctivenessResult.from_dict(d)
                for d in data["distinctiveness_results"]
            ),
            is_better=perf["is_better"],
            num_correct=perf["num_correct"],
            num_total=perf["num_total"],
            accuracy=perf["accuracy"],
            accuracy_delta=perf["accuracy_delta"],
            cutoff=metrics["cutoff"],
            score_min=metrics["score_min"],
            score_max=metrics["score_max"],
            score_mean=metrics["score_mean"],
            score_median=metrics["score_median"],
        )

    def to_dict(self):
        """Convert to dictionary.

        Returns
        -------
        dict
            Serialized AblationStepResult
        """
        return {
            "step": self.step,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "ablated_ngrams": list(self.ablated_ngrams),
            "newly_found_ngrams": list(self.newly_found_ngrams),
            "distinctiveness_results": [
                d.to_dict() for d in self.distinctiveness_results
            ],
            "performance": {
                "is_better": self.is_better,
                "num_correct": self.num_correct,
                "num_total": self.num_total,
                "accuracy": self.accuracy,
                "accuracy_delta": self.accuracy_delta,
            },
            "distinctiveness_metrics": {
                "cutoff": self.cutoff,
                "score_min": self.score_min,
                "score_max": self.score_max,
                "score_mean": self.score_mean,
                "score_median": self.score_median,
            },
        }


@dataclass(frozen=True)
class BaselineResult:
    """Results from CoT-OFF baseline run.

    Parameters
    ----------
    responses : tuple[QAWithResponse, ...]
        QA pairs with model responses
    verdicts : tuple[LLMJudgeVerdict, ...]
        Judged responses
    num_correct : int
        Number of correct responses
    num_total : int
        Number of total responses
    accuracy : int
        Accuracy of all responses
    """

    responses: tuple[QAWithResponse, ...]
    verdicts: tuple[LLMJudgeVerdict, ...]
    num_correct: int
    num_total: int
    accuracy: float

    @classmethod
    def from_dict(cls, data):
        """Reconstruct from dictionary.

        Parameters
        ----------
        data : dict
            Serialized BaselineResult

        Returns
        -------
        BaselineResult
            Reconstructed instance
        """
        perf = data["performance"]

        return cls(
            responses=tuple(
                QAWithResponse.from_dict(r) for r in data["responses"]
            ),
            verdicts=tuple(
                LLMJudgeVerdict.from_dict(v) for v in data["verdicts"]
            ),
            num_correct=perf["num_correct"],
            num_total=perf["num_total"],
            accuracy=perf["accuracy"],
        )

    def to_dict(self):
        """Convert to dictionary.

        Returns
        -------
        dict
            Serialized BaselineResult
        """
        return {
            "responses": [r.to_dict() for r in self.responses],
            "verdicts": [v.to_dict() for v in self.verdicts],
            "performance": {
                "num_correct": self.num_correct,
                "num_total": self.num_total,
                "accuracy": self.accuracy,
            },
        }


@dataclass(frozen=True)
class AblationRunResult:
    """Complete results from an ablation run.

    Parameters
    ----------
    model : str
        LLM name
    judge : str
        Judge name
    num_examples : int
        Number of QA pairs
    seed : int
        Random seed
    cutoff: float
        Quantile cutoff value for the metric; n-grams with score >= cutoff for
        a step will be ablated
    max_steps : int
        Maximum number of steps the ablations process can take
    ngram_range : tuple[int, int]
        N-gram range that will be assigned scores by metric (e.g. (1, 2) ->
        unigrams and bigrams)
    metric : str
        Name of the metric (e.g. 'fighting_words'); see Metric
    strategy : str
        Masking strategy; must be "mask" or "penalize"
    penalty_weight : float
        Penalty weight to use when strategy is "penalize"; must be (0, 1]
    top_k : int or None
        Number of most distinctive values to ablate
    baseline : BaselineResult
        Baseline results from CoT-OFF
    ablation_steps : tuple[AblationStepResult, ...]
        Steps from CoT-ON
    total_steps_completed : int
        Total steps taken
    total_ngrams_ablated : int
        Total number of n-grams ablated
    termination_reason : str
        The reason ablations were stopped (e.g. 'degraded', 'max_steps')
    final_accuracy : float
        Accuracy of CoT-ON responses on the last step
    final_accuracy_delta : float
        Accuracy of CoT-ON responses on the last step - accuracy of CoT-OFF
        responses
    """

    # Metadata
    model: str
    judge: str
    num_examples: int
    seed: int

    # Configuration
    cutoff: float
    max_steps: int
    ngram_range: tuple[int, int]
    metric: str
    strategy: str
    penalty_weight: float
    top_k: int | None

    # Results
    baseline: BaselineResult
    ablation_steps: tuple[AblationStepResult, ...]

    # Summary
    total_steps_completed: int
    total_ngrams_ablated: int
    termination_reason: str
    final_accuracy: float
    final_accuracy_delta: float

    # Internal attributes for .save_run()
    _RUN_INFO_FILE = "run_info.json"
    _BASELINE_FILE = "baseline.json"
    _STEP_FILE_STUB = "step_{step:03d}.json"

    @classmethod
    def from_dict(cls, path):
        """Load from directory.

        Parameters
        ----------
        path : str or Path
            Ablation run result directory path

        Returns
        -------
        AblationRunResult
            Reconstructed instance
        """
        path = Path(path)

        # Load run info
        with (path / cls._RUN_INFO_FILE).open("r") as f:
            info = json.load(f)

        # Load baseline
        with (path / cls._BASELINE_FILE).open("r") as f:
            baseline = BaselineResult.from_dict(json.load(f))

        # Load all step files
        steps = []
        step_num = 0
        while True:
            step_file = path / cls._STEP_FILE_STUB.format(step=step_num)
            if not step_file.exists():
                break

            with step_file.open("r") as f:
                steps.append(AblationStepResult.from_dict(json.load(f)))

            step_num += 1

        return cls(
            model=info["model"],
            judge=info["judge"],
            num_examples=info["num_examples"],
            seed=info["seed"],
            cutoff=info["cutoff"],
            max_steps=info["max_steps"],
            ngram_range=tuple(info["ngram_range"]),
            metric=info["metric"],
            strategy=info["strategy"],
            penalty_weight=info["penalty_weight"],
            top_k=info["top_k"],
            baseline=baseline,
            ablation_steps=tuple(steps),
            total_steps_completed=info["total_steps_completed"],
            total_ngrams_ablated=info["total_ngrams_ablated"],
            termination_reason=info["termination_reason"],
            final_accuracy=info["final_accuracy"],
            final_accuracy_delta=info["final_accuracy_delta"],
        )

    def to_dict(self):
        """Convert to dictionary.

        Returns
        -------
        dict
            Serialized fields
        """
        return {
            "metadata": {
                "model": self.model,
                "judge": self.judge,
                "num_examples": self.num_examples,
                "seed": self.seed,
            },
            "configuration": {
                "cutoff": self.cutoff,
                "max_steps": self.max_steps,
                "ngram_range": self.ngram_range,
                "metric": self.metric,
                "strategy": self.strategy,
                "penalty_weight": self.penalty_weight,
                "top_k": self.top_k,
            },
            "baseline": self.baseline.to_dict(),
            "ablation_steps": [s.to_dict() for s in self.ablation_steps],
            "summary": {
                "total_steps_completed": self.total_steps_completed,
                "total_ngrams_ablated": self.total_ngrams_ablated,
                "termination_reason": self.termination_reason,
                "final_accuracy": self.final_accuracy,
                "final_accuracy_delta": self.final_accuracy_delta,
            },
        }

    def save_run(self, path):
        """Save to multiple JSON files in a directory.

        Parameters
        ----------
        path : str or Path
            Path to output directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save run info
        with (path / self._RUN_INFO_FILE).open("w") as f:
            json.dump(
                {
                    "model": self.model,
                    "judge": self.judge,
                    "num_examples": self.num_examples,
                    "seed": self.seed,
                    "cutoff": self.cutoff,
                    "max_steps": self.max_steps,
                    "ngram_range": self.ngram_range,
                    "metric": self.metric,
                    "strategy": self.strategy,
                    "penalty_weight": self.penalty_weight,
                    "top_k": self.top_k,
                    "total_steps_completed": self.total_steps_completed,
                    "total_ngrams_ablated": self.total_ngrams_ablated,
                    "termination_reason": self.termination_reason,
                    "final_accuracy": self.final_accuracy,
                    "final_accuracy_delta": self.final_accuracy_delta,
                },
                f,
                indent=2,
            )

        # Save baseline
        with (path / self._BASELINE_FILE).open("w") as f:
            json.dump(self.baseline.to_dict(), f, indent=2)

        # Save steps
        for step in self.ablation_steps:
            filename = self._STEP_FILE_STUB.format(step=step.step)
            with (path / filename).open("w") as f:
                json.dump(step.to_dict(), f, indent=2)


@dataclass(frozen=True)
class OnePassResult:
    """Complete results from a onepass run.

    Parameters
    ----------
    model : str
        LLM name
    seed : int
        Random seed
    cot_off : tuple[QAWithResponse, ...]
        LLM responses for CoT-OFF
    cot_on : tuple[QAWithResponse, ...]
        LLM responses for CoT-ON
    """

    # Metadata
    model: str
    seed: int

    # CoT-OFF/ON
    cot_off: tuple[QAWithResponse, ...] = field(default_factory=tuple)
    cot_on: tuple[QAWithResponse, ...] = field(default_factory=tuple)

    # Internal attributes for .save_run()
    _RUN_INFO_FILE = "run_info.json"
    _COT_OFF_FILE = "cot_off.json"
    _COT_ON_FILE = "cot_on.json"

    def to_dict(self):
        """Convert to dictionary.

        Returns
        -------
        dict
            Serialized OnePassResult
        """
        return {
            "metadata": {
                "model": self.model,
                "seed": self.seed,
            },
            "runs": {
                "cot_off": [r.to_dict() for r in self.cot_off],
                "cot_on": [r.to_dict() for r in self.cot_on],
            },
        }

    def save_run(self, path):
        """Save to multiple JSON files in a directory.

        Parameters
        ----------
        path : str or Path
            Path to output directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with (path / self._RUN_INFO_FILE).open("w") as f:
            json.dump({"model": self.model, "seed": self.seed}, f, indent=2)

        # Save CoT-OFF results
        with (path / self._COT_OFF_FILE).open("w") as f:
            json.dump([r.to_dict() for r in self.cot_off], f, indent=2)

        # Save CoT-ON results
        with (path / self._COT_ON_FILE).open("w") as f:
            json.dump([r.to_dict() for r in self.cot_on], f, indent=2)
