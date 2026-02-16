#!/usr/bin/env python3
"""Prepare data for static frontend viewer.

This script combines vocab extraction and data preparation:
1. Extracts vocabulary from HuggingFace tokenizer
2. Generates index.json with all runs metadata
3. Copies JSON files to frontend/public/data/ for static serving
4. Compresses large JSON files with gzip to reduce file sizes
"""

import gzip
import hashlib
import json
import re
import shutil
import sys
from collections import namedtuple
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


TokenizerSignature = namedtuple('TokenizerSignature', ['vocab_size', 'vocab_hash'])

def get_tokenizer_signature(model_name: str) -> TokenizerSignature:
    """Get unique signature to identify tokenizers for deduplication."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()
    vocab_str = json.dumps(vocab, sort_keys=True)
    vocab_hash = hashlib.sha256(vocab_str.encode()).hexdigest()[:16]
    return TokenizerSignature(vocab_size=len(vocab), vocab_hash=vocab_hash)

def model_name_to_vocab_key(model_name: str) -> str:
    """Convert model name to safe filename key.
    Examples: 'Qwen/Qwen3-0.6B' -> 'qwen-qwen3'
              'ibm-granite/granite-3.3-2b-instruct' -> 'ibmgranite-granite'
    """
    parts = model_name.split('/')
    if len(parts) == 2:
        org, model = parts
        model_base = re.split(r'[-_]\d', model)[0]
        org_clean = org.lower().replace('-', '').replace('_', '')
        model_clean = model_base.lower().replace('-', '').replace('_', '')
        return f"{org_clean}-{model_clean}"
    return model_name.lower().replace('/', '-').replace('_', '-')

def extract_all_vocabs(runs: list, output_dir: Path) -> dict:
    """Extract vocabularies for all unique tokenizers.
    Returns: dict mapping model_name -> vocab_filename
    """
    # Collect unique models
    models = {run['info'].get('model') for run in runs if run['info'].get('model')}
    print(f"\nFound {len(models)} unique models across {len(runs)} runs")

    # Group models by tokenizer signature
    tokenizer_groups = {}
    signatures_to_key = {}

    for model_name in sorted(models):
        print(f"  Analyzing tokenizer for {model_name}...")
        try:
            signature = get_tokenizer_signature(model_name)
            if signature not in tokenizer_groups:
                tokenizer_groups[signature] = []
                signatures_to_key[signature] = model_name_to_vocab_key(model_name)
            tokenizer_groups[signature].append(model_name)
        except Exception as e:
            print(f"    ⚠ Warning: Failed to load tokenizer for {model_name}: {e}")

    # Extract one vocab per unique tokenizer
    vocab_mapping = {}
    vocab_files_created = set()

    for signature, model_names in tokenizer_groups.items():
        vocab_key = signatures_to_key[signature]
        vocab_filename = f"vocab-{vocab_key}.json"
        representative_model = model_names[0]

        if vocab_filename not in vocab_files_created:
            vocab_path = output_dir / vocab_filename
            extract_vocab(representative_model, vocab_path)
            vocab_files_created.add(vocab_filename)
            print(f"  ✓ Created {vocab_filename} for {len(model_names)} model(s)")
            if len(model_names) > 1:
                print(f"    Shared by: {', '.join(model_names)}")

        for model_name in model_names:
            vocab_mapping[model_name] = vocab_filename

    # Write vocab-index.json
    vocab_index_path = output_dir / "vocab-index.json"
    with open(vocab_index_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_mapping, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Generated {len(vocab_files_created)} vocabulary file(s)")
    print(f"✓ Created vocab index: {vocab_index_path}")
    return vocab_mapping


def compress_json_file(input_path: Path, output_path: Path, compress_threshold_mb: float = 5.0) -> bool:
    """Compress a JSON file with gzip if it exceeds the size threshold.

    CloudFlare Pages has a 25 MiB per-file limit. Large JSON files (especially
    step files with many verdicts) can exceed this. We compress files larger than
    5 MB to ensure they stay well under the limit. Compression ratios are typically
    90%+ for these files (e.g., 17 MB → 1.2 MB).

    Parameters
    ----------
    input_path : Path
        Source JSON file path
    output_path : Path
        Destination path (will add .gz if compressed)
    compress_threshold_mb : float
        File size threshold in MB to trigger compression

    Returns
    -------
    bool
        True if file was compressed, False if copied uncompressed
    """
    file_size_mb = input_path.stat().st_size / (1024 * 1024)

    if file_size_mb > compress_threshold_mb:
        # Compress the file
        with open(input_path, 'rb') as f_in:
            with gzip.open(str(output_path) + '.gz', 'wb', compresslevel=9) as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    else:
        # Copy without compression
        shutil.copy2(input_path, output_path)
        return False


def prepare_data(data_dir: Path, output_dir: Path):
    """Prepare data for static frontend.

    Parameters
    ----------
    data_dir : Path
        Source data directory (e.g., pragma/data/)
    output_dir : Path
        Output directory (e.g., frontend/public/data/)
    """
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all runs
    runs = []
    run_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    if not run_dirs:
        print(f"Warning: No run directories found in {data_dir}")
        return None

    print(f"\nPreparing data from {len(run_dirs)} runs...")

    for run_dir in sorted(run_dirs):
        run_name = run_dir.name
        run_info_path = run_dir / "run_info.json"

        if not run_info_path.exists():
            print(f"  ⚠ Skipping {run_name} (no run_info.json)")
            continue

        # Load run info
        with open(run_info_path) as f:
            run_info = json.load(f)

        runs.append({"name": run_name, "info": run_info})

        # Create output directory for this run
        run_output_dir = output_dir / run_name
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Copy run_info.json (always uncompressed, it's small)
        shutil.copy2(run_info_path, run_output_dir / "info.json")

        # Copy baseline.json if exists (compress if large)
        baseline_path = run_dir / "baseline.json"
        if baseline_path.exists():
            compress_json_file(baseline_path, run_output_dir / "baseline.json")

        # Copy and compress step files
        step_files = sorted(run_dir.glob("step_*.json"))
        compressed_count = 0
        for step_file in step_files:
            was_compressed = compress_json_file(step_file, run_output_dir / step_file.name)
            if was_compressed:
                compressed_count += 1

        status = f"{len(step_files)} steps"
        if compressed_count > 0:
            status += f" ({compressed_count} compressed)"

        print(f"  ✓ {run_name}: copied {status} + baseline + info")

    # Generate index.json (always uncompressed)
    index_path = output_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Generated {index_path}")
    print(f"✓ Prepared {len(runs)} runs in {output_dir}")

    return runs


def main():
    """Main entry point."""
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data"
    frontend_dir = project_root / "app" / "frontend"
    output_dir = frontend_dir / "public" / "data"
    vocab_path = frontend_dir / "public" / "vocab.json"

    print("=" * 60)
    print("Pragma Data Preparation")
    print("=" * 60)

    # Prepare data files
    runs = prepare_data(data_dir, output_dir)

    if not runs:
        print("\nNo runs to process. Exiting.")
        sys.exit(1)

    # Extract vocabularies for all models
    print("\n" + "=" * 60)
    print("Extracting Vocabularies")
    print("=" * 60)

    vocab_mapping = extract_all_vocabs(runs, frontend_dir / "public")

    print("\n" + "=" * 60)
    print("✓ Data preparation complete!")
    print("=" * 60)
    print(f"\nData ready in: {output_dir}")
    print(f"Vocabularies: {len(set(vocab_mapping.values()))} unique tokenizer(s)")
    print(f"  - Models supported: {len(vocab_mapping)}")
    print("\nYou can now run: pixi r app")


if __name__ == "__main__":
    main()
