#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main entry point for the program."""

import argparse
import logging
import os
from pathlib import Path

from pragma import InferenceConfig, LLMJudge
from pragma.generate import (
    AblationRunner,
    load_benchmark,
    load_model,
    run_onepass,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


class EnvironmentVariableError(Exception):
    """Raised when a required environment variable isn't set."""

    pass


def setup_logging(verbose):
    """Configure logging based on verbosity level.

    Parameters
    -----------
    verbose : bool
        If True, set to logging.DEBUG
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv=None):
    """Run the script."""
    parser = argparse.ArgumentParser(
        prog="pragma",
        description="Perform ablation studies on chain-of-thought generation",
        epilog="Example: %(prog)s config.toml --runtype onepass",
    )
    parser.add_argument(
        "config",
        type=Path,
        metavar="CONFIG",
        help="TOML config",
    )
    parser.add_argument(
        "-r",
        "--runtype",
        choices=["onepass", "ablations"],
        default="ablations",
        metavar="RUNTYPE",
        help="Type of CoT to run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume abation run from last completed step",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG level logging",
    )

    # Parse arguments and set up logging
    args = parser.parse_args()
    setup_logging(args.verbose)

    # Validate resume flag. If all's okay, load the config
    if args.resume and args.runtype != "ablations":
        raise ValueError("--resume can only be used with --runtype ablations")

    cfg = InferenceConfig.from_toml(args.config)
    if not cfg.runs:
        raise ValueError("No runs defined in the config")

    # Check for an API key, which we need for ablations
    api_key = os.environ.get(cfg.ablations.api_key, None)
    if args.runtype == "ablations" and api_key is None:
        raise EnvironmentVariableError(
            f"No API key set at {cfg.ablations.api_key}"
        )

    # Load the model and dataset, transforming the latter into examples
    tokenizer, model = load_model(cfg)
    examples = load_benchmark(cfg)

    # Set up outputs
    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating CoT with %s", args.runtype)

    if args.runtype == "onepass":
        results = run_onepass(
            examples=examples,
            model=model,
            tokenizer=tokenizer,
            inference_config=cfg,
        )

    elif args.runtype == "ablations":
        judge = LLMJudge(
            cfg.ablations.judge,
            base_url=cfg.ablations.base_url,
            api_key=api_key,
        )
        runner = AblationRunner(
            examples,
            model,
            tokenizer,
            cfg,
            judge,
            output_dir=output_path,
            resume=args.resume,
        )
        results = runner.run()

    # If doing an ablation, the last save will update run_info.json with final
    # metadata
    results.save_run(output_path)


if __name__ == "__main__":
    main()
