"""Tests for configuration loading."""

from pathlib import Path
import glob
import tempfile
import pytest

from pragma.config import InferenceConfig, RunConfig, AblationConfig


def get_config_files():
    """Discover all TOML config files in the config/ directory."""
    return glob.glob("config/*.toml")


@pytest.fixture(params=get_config_files(), ids=lambda p: Path(p).name)
def config_file(request):
    """Parametrize tests across all config files."""
    return request.param


def test_config_loads_successfully(config_file):
    """Test that all config files load without errors."""
    config = InferenceConfig.from_toml(config_file)
    assert config is not None
    assert isinstance(config, InferenceConfig)


def test_config_has_required_fields(config_file):
    """Test that loaded configs have all required fields."""
    config = InferenceConfig.from_toml(config_file)

    # Check required fields exist and are not None
    assert config.model is not None
    assert config.dataset_name is not None
    assert config.dataset_config is not None
    assert config.split is not None
    assert config.output is not None
    assert config.runs is not None

    # Check types
    assert isinstance(config.model, str)
    assert isinstance(config.dataset_name, str)
    assert isinstance(config.dataset_config, str)
    assert isinstance(config.split, str)
    assert isinstance(config.output, Path)
    assert isinstance(config.runs, list)

    # Optional ablations
    if config.ablations is not None:
        assert isinstance(config.ablations, AblationConfig)


def test_config_runs_are_run_configs(config_file):
    """Test that runs list contains RunConfig objects."""
    config = InferenceConfig.from_toml(config_file)

    # Check that all runs are RunConfig instances
    for run in config.runs:
        assert isinstance(run, RunConfig)
        assert hasattr(run, "name")
        assert hasattr(run, "enable_thinking")
        assert hasattr(run, "generation_kwargs")


def test_ablations_config_when_present():
    """Test that ablations config loads correctly when present."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False
    ) as f:
        f.write("""
[model]
name = "test/model"

[dataset]
name = "test/dataset"
config = "test-config"
split = "train"

[io]
output = "data/test-dir"

[ablations]
judge = "claude-sonnet-4-5"
base_url = "https://api.anthropic.com/v1/"
api_key = "ANTHROPIC_API_KEY"
metric = "fighting_words"
ngram_range = [1, 3]
strategy = "mask"
penalty_weight = 1.0
cutoff = 0.95
top_k = ""
max_steps = 100
""")
        temp_path = f.name

    try:
        config = InferenceConfig.from_toml(temp_path)
        assert config.ablations is not None
        assert isinstance(config.ablations, AblationConfig)
        assert config.ablations.judge == "claude-sonnet-4-5"
        assert config.ablations.base_url == "https://api.anthropic.com/v1/"
        assert config.ablations.api_key == "ANTHROPIC_API_KEY"
        assert config.ablations.metric == "fighting_words"
        assert config.ablations.strategy == "mask"
        assert config.ablations.penalty_weight == 1.0
        assert config.ablations.ngram_range == (1, 3)
        assert config.ablations.cutoff == 0.95
        assert config.ablations.top_k is None
        assert config.ablations.max_steps == 100
    finally:
        Path(temp_path).unlink()


def test_ablations_defaults_to_none():
    """Test that ablations defaults to None when not provided."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False
    ) as f:
        f.write("""
[model]
name = "test/model"

[dataset]
name = "test/dataset"
config = "test-config"
split = "train"

[io]
output = "data/test-dir"
""")
        temp_path = f.name

    try:
        config = InferenceConfig.from_toml(temp_path)
        assert config.ablations is None
    finally:
        Path(temp_path).unlink()


# Error handling tests


def test_missing_model_raises_error():
    """Test that missing model field raises KeyError."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False
    ) as f:
        f.write("""
[dataset]
name = "test/dataset"
config = "test-config"
split = "train"

[io]
output = "data/test-dir"
""")
        temp_path = f.name

    try:
        with pytest.raises(KeyError, match="model"):
            InferenceConfig.from_toml(temp_path)
    finally:
        Path(temp_path).unlink()


def test_missing_dataset_name_raises_error():
    """Test that missing dataset.name raises KeyError."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False
    ) as f:
        f.write("""
[model]
name = "test/model"

[dataset]
config = "test-config"
split = "train"

[io]
output = "data/test-dir"
""")
        temp_path = f.name

    try:
        with pytest.raises(KeyError, match="name"):
            InferenceConfig.from_toml(temp_path)
    finally:
        Path(temp_path).unlink()


def test_missing_io_output_raises_error():
    """Test that missing io.output raises KeyError."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False
    ) as f:
        f.write("""
[model]
name = "test/model"

[dataset]
name = "test/dataset"
config = "test-config"
split = "train"

[io]
""")
        temp_path = f.name

    try:
        with pytest.raises(KeyError, match="output"):
            InferenceConfig.from_toml(temp_path)
    finally:
        Path(temp_path).unlink()


def test_empty_runs_list_allowed():
    """Test that empty runs list is allowed."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False
    ) as f:
        f.write("""
[model]
name = "test/model"

[dataset]
name = "test/dataset"
config = "test-config"
split = "train"

[io]
output = "data/test-dir"

[runs]
""")
        temp_path = f.name

    try:
        config = InferenceConfig.from_toml(temp_path)
        assert config.runs == []
    finally:
        Path(temp_path).unlink()


def test_malformed_toml_raises_error():
    """Test that malformed TOML raises an error."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False
    ) as f:
        f.write("""
[model
name = "missing bracket"
""")
        temp_path = f.name

    try:
        with pytest.raises(Exception):  # tomllib.TOMLDecodeError
            InferenceConfig.from_toml(temp_path)
    finally:
        Path(temp_path).unlink()
