#!/usr/bin/env python3
"""Extract vocabulary from HuggingFace tokenizer to JSON for frontend use."""

import json
import sys
from pathlib import Path
from transformers import AutoTokenizer


def extract_vocab(model_name: str, output_path: Path):
    """Extract vocabulary from a HuggingFace tokenizer.

    Parameters
    ----------
    model_name : str
        HuggingFace model name (e.g., "Qwen/Qwen3-0.6B")
    output_path : Path
        Path to save vocab.json
    """
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get the vocabulary mapping
    vocab = tokenizer.get_vocab()

    # Create reverse mapping: token_id -> token_string
    id_to_token = {id: token for token, id in vocab.items()}

    # Extract special tokens info
    special_tokens = {
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "unk_token": tokenizer.unk_token,
        "pad_token": tokenizer.pad_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    # Create output data
    output_data = {
        "model": model_name,
        "vocab_size": len(vocab),
        "special_tokens": special_tokens,
        "vocab": id_to_token,
    }

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✓ Extracted {len(vocab)} tokens to {output_path}")
    print(f"  Model: {model_name}")
    print(f"  Vocab size: {len(vocab)}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        # Try to auto-detect model from run_info.json
        data_dir = Path(__file__).parent.parent.parent / "data"
        run_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

        if not run_dirs:
            print("Error: No runs found in data/ directory")
            print("Usage: python extract_vocab.py [model_name]")
            print("Example: python extract_vocab.py Qwen/Qwen3-0.6B")
            sys.exit(1)

        # Use first run's model
        run_info_path = run_dirs[0] / "run_info.json"
        with open(run_info_path) as f:
            run_info = json.load(f)
        model_name = run_info["model"]
        print(f"Auto-detected model from {run_dirs[0].name}: {model_name}")
    else:
        model_name = sys.argv[1]

    # Output to frontend public directory
    output_path = Path(__file__).parent.parent / "frontend" / "public" / "vocab.json"

    extract_vocab(model_name, output_path)


if __name__ == "__main__":
    main()
