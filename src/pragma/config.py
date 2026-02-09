"""Inference configuration dataclasses."""

import tomllib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


class LoaderMixin(ABC):
    """A simple mix-in for populating configurations from TOML files."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data):
        """Build configuration from a dictionary."""
        raise NotImplementedError

    @classmethod
    def from_toml(cls, config_file):
        """Load configuration file.

        Parameters
        ----------
        config_file : Path or str
            Path to the TOML configuration file

        Returns
        -------
        LoaderMixin
            Loaded configuration
        """
        with Path(config_file).open("rb") as f:
            data = tomllib.load(f)

        return cls.from_dict(data)


@dataclass
class AblationConfig(LoaderMixin):
    """Configuration for CoT ablations."""

    judge: str = "claude-sonnet-4-5"
    base_url: str = "https://api.anthropic.com/v1/"
    api_key: str = "ANTHROPIC_API_KEY"
    metric: str = "fighting_words"
    ngram_range: tuple[int, int] = (1, 3)
    strategy: str = "mask"
    penalty_weight: float = 1.0
    cutoff: float = 0.95
    top_k: int or None = None
    max_steps: float = 100

    def __post_init__(self):
        """Ensure the requested initialization options are valid.

        Raises
        ------
        ValueError
            If the metric or masking strategy is invalid
        """
        if self.metric not in ("dunning", "fighting_words", "pmi"):
            raise ValueError(
                "Invalid metric. Select: 'dunning', 'fighting_words', 'pmi'"
            )

        if self.strategy not in ("mask", "penalize"):
            raise ValueError(
                "Invalid masking strategy. Select: 'mask', 'penalize'"
            )

    @classmethod
    def from_dict(cls, data):
        """Populate configuration from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary of ablation configuration parameters

        Returns
        -------
        AblationConfig
            Populated configuration
        """
        return cls(
            judge=data.get("judge", "claude-sonnet-4-5"),
            base_url=data.get("base_url", "https://api.anthropic.com/v1/"),
            api_key=data.get("api_key", "ANTHROPIC_API_KEY"),
            metric=data.get("metric", "fighting_words"),
            ngram_range=tuple(data.get("ngram_range", [1, 3])),
            strategy=data.get("strategy", "mask"),
            penalty_weight=data.get("penalty_weight", 1.0),
            cutoff=data.get("cutoff", 0.95),
            top_k=data.get("top_k", "") or None, # Override empty TOML string
            max_steps=int(data.get("max_steps", 100)),
        )


@dataclass
class RunConfig(LoaderMixin):
    """Configuration for an inference run."""

    name: str = "default"
    enable_thinking: bool = True
    generation_kwargs: dict = field(default_factory=dict)

    @property
    def generation(self):
        """Alias for generation kwargs."""
        return self.generation_kwargs

    @classmethod
    def from_dict(cls, data):
        """Populate configuration from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary of configuration parameters
        Returns
        -------
        RunConfig
            Populated configuration
        """
        # Accept common variants for generation kwargs
        gen = (
            data.get("generation_kwargs")
            or data.get("generation")
            or data.get("kwargs")
            or {}
        )
        # Ensure it's a plain dict copy
        generation_kwargs = dict(gen)

        return cls(
            name=data.get("name", "default"),
            enable_thinking=bool(data.get("enable_thinking", True)),
            generation_kwargs=generation_kwargs,
        )


@dataclass
class InferenceConfig(LoaderMixin):
    """Top-level configuration for inference runs."""

    # Model and data
    model: str = "Qwen/Qwen3-8B"
    dataset_name: str = "allenai/ai2_arc"
    dataset_config: str = "ARC-Challenge"
    split: str = "train"

    # IO
    output: Path | str = "data/qwen3-8B"

    # Runtime
    batch_size: int = 32
    seed: int = 357
    attn_implementation: str = "sdpa"

    # Think token configuration
    think_substring: str = "think"
    think_token_format: str = "<{tok}>"

    # Ablations
    ablations: AblationConfig | None = None

    # Runs
    runs: list[RunConfig] = field(default_factory=list)

    def __post_init__(self):
        """Ensure IO paths are Paths and validate output is a directory.

        Raises
        ------
        ValueError
            If the output isn't a directory
        """
        self.output = Path(self.output)

        # If output exists, ensure it's a directory
        if self.output.exists() and not self.output.is_dir():
            raise ValueError(
                f"Output path '{self.output}' exists but isn't a directory"
            )

    @classmethod
    def from_dict(cls, data):
        """Build configuration from a dictionary.
        Parameters
        ----------
        data : dict
            Dictionary with keys for 'model', 'dataset', 'io', 'runtime',
            'runs'

        Returns
        -------
        InferenceConfig
            Populated configuration

        Raises
        ------
        KeyError
            If the dictionary is missing required keys in nested sections
        """
        try:
            model = data.get("model")
            dataset = data.get("dataset", {})
            io = data.get("io", {})
            runtime = data.get("runtime", {})
            ablations_block = data.get("ablations", None)
            runs_block = data.get("runs", [])

            # Model can be a dict or a string
            if isinstance(model, dict):
                model_name = model["name"]
                think_substring = model.get("substring", "think")
                think_token_format = model.get("token_format", "<{tok}>")
            elif isinstance(model, str):
                model_name = model
                think_substring = "think"
                think_token_format = "<{tok}>"
            else:
                raise KeyError("model")

            # Parse ablations
            ablations = None
            if ablations_block:
                ablations = AblationConfig.from_dict(ablations_block)

            runs = []
            if isinstance(runs_block, list):
                for run in runs_block:
                    runs.append(RunConfig.from_dict(run or {}))
            elif isinstance(runs_block, dict):
                for name, run in runs_block.items():
                    run = dict(run or {})
                    run["name"] = run.get("name", name)
                    runs.append(RunConfig.from_dict(run))

            return cls(
                model=model_name,
                dataset_name=dataset["name"],
                dataset_config=dataset["config"],
                split=dataset["split"],
                output=io["output"],
                batch_size=int(runtime.get("batch_size", 32)),
                seed=int(runtime.get("seed", 357)),
                attn_implementation=runtime.get("attn_implementation", "sdpa"),
                think_substring=think_substring,
                think_token_format=think_token_format,
                ablations=ablations,
                runs=runs,
            )

        except KeyError as e:
            raise KeyError(f"Missing required config key: {e}") from e
