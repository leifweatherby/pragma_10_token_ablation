#!/usr/bin/env python3
"""Build a comprehensive metacognitive token set using treetok clustering.

Method:
1. Run Fighting Words on ARC-Easy CoT-ON vs CoT-OFF to get distinctiveness scores.
2. Take the top-1281 tokens (the dense metacognitive cluster above the score cliff).
3. Run treetok (Levenshtein distance=1) over the full Qwen3-0.6B vocabulary.
4. For each seed token in the top-1281, collect all cluster-mates (morphological
   variants: case, inflection, punctuation attachment) even if outside the top-1281.
5. Output the expanded set with full metadata for use as an ablation token list.
"""

import json
from pathlib import Path
from transformers import AutoTokenizer
from treetok import TokenClusterer

MODEL = "Qwen/Qwen3-0.6B"
STEP0_PATH = Path("data/qwen3-0.6B-arc-easy/step_000.json")
OUTPUT_PATH = Path("data/metacognitive_tokens.json")
CLIFF_RANK = 1281  # rank where score drops by 1.31 units


def main():
    print("Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    vocab_dict = tok.get_vocab()
    vocab_items = sorted(vocab_dict.items(), key=lambda kv: kv[1])
    vocab = [t for t, _ in vocab_items]
    token_ids = [i for _, i in vocab_items]
    print(f"Vocab size: {len(vocab)}", flush=True)

    print("Building treetok clusters over full vocab (distance=1)...", flush=True)
    clusterer = TokenClusterer(vocab, token_ids, max_distance=1)
    clusterer.build()
    print("BK-trees built, running union-find cluster()...", flush=True)
    clusterer.cluster()
    clusters = clusterer.get_cluster_info()
    print(f"Found {len(clusters)} clusters", flush=True)

    # Build lookup: token_id -> cluster
    id_to_cluster = {}
    for c in clusters:
        for tid in c["token_ids"]:
            id_to_cluster[tid] = c

    # Load step-0 distinctiveness results
    print("Loading distinctiveness scores...", flush=True)
    step0 = json.loads(STEP0_PATH.read_text())
    results = step0["distinctiveness_results"]
    score_cliff_before = results[CLIFF_RANK - 1]["score"]
    score_cliff_after  = results[CLIFF_RANK]["score"]

    # Seed set: top-1281 (dense metacognitive cluster above the cliff)
    seed_records = results[:CLIFF_RANK]
    seed_id_to_meta = {}
    for r in seed_records:
        for tid in r["ngram"]:
            seed_id_to_meta[tid] = {
                "rank": r["rank"],
                "score": r["score"],
                "above_cutoff": r.get("above_cutoff", False),
            }

    print(f"Seed set: {len(seed_id_to_meta)} token IDs from top-{CLIFF_RANK} ranks", flush=True)

    # Expand via treetok clusters
    expanded_ids = set(seed_id_to_meta.keys())
    cluster_details = {}

    for seed_id, meta in sorted(seed_id_to_meta.items(), key=lambda x: x[1]["rank"]):
        if seed_id not in id_to_cluster:
            continue
        c = id_to_cluster[seed_id]
        rep = c["representative"]

        if rep not in cluster_details:
            cluster_details[rep] = {
                "representative": rep,
                "representative_id": c["representative_id"],
                "seed_members": [],
                "expanded_members": [],
            }

        cluster_details[rep]["seed_members"].append({
            "token": tok.decode([seed_id]),
            "token_id": seed_id,
            "rank": meta["rank"],
            "score": round(meta["score"], 6),
            "above_cutoff": meta["above_cutoff"],
        })

        for xtid in c["token_ids"]:
            if xtid not in seed_id_to_meta:
                cluster_details[rep]["expanded_members"].append({
                    "token": tok.decode([xtid]),
                    "token_id": xtid,
                })
                expanded_ids.add(xtid)

    # Deduplicate expanded_members per cluster
    for cd in cluster_details.values():
        seen = set()
        deduped = []
        for m in cd["expanded_members"]:
            if m["token_id"] not in seen:
                seen.add(m["token_id"])
                deduped.append(m)
        cd["expanded_members"] = deduped

    n_expanded = len(expanded_ids) - len(seed_id_to_meta)
    print(f"Expanded set: {len(expanded_ids)} token IDs (+{n_expanded} via treetok)", flush=True)
    print(f"Clusters with expansions: {sum(1 for c in cluster_details.values() if c['expanded_members'])}", flush=True)

    # Build full token list: seeds first (sorted by rank), then treetok-only additions
    full_token_list = []
    for r in seed_records:
        for tid in r["ngram"]:
            full_token_list.append({
                "token": tok.decode([tid]),
                "token_id": tid,
                "rank": r["rank"],
                "score": round(r["score"], 6),
                "above_cutoff": r.get("above_cutoff", False),
                "source": "fighting_words",
            })
    for tid in sorted(expanded_ids - set(seed_id_to_meta.keys())):
        full_token_list.append({
            "token": tok.decode([tid]),
            "token_id": tid,
            "rank": None,
            "score": None,
            "above_cutoff": False,
            "source": "treetok_expansion",
        })

    output = {
        "method": {
            "step1_fighting_words": (
                "Run Fighting Words (log-odds ratio) on CoT-ON reasoning vs CoT-OFF "
                "responses for 100 ARC-Easy examples with Qwen3-0.6B."
            ),
            "step2_cliff_detection": (
                f"Identify the score cliff: top-{CLIFF_RANK} tokens score "
                f"{score_cliff_before:.4f}, rank-{CLIFF_RANK} drops to "
                f"{score_cliff_after:.4f} (gap={score_cliff_before - score_cliff_after:.4f}). "
                "All tokens above the cliff form the metacognitive cluster."
            ),
            "step3_treetok_expansion": (
                f"Run treetok TokenClusterer (Levenshtein distance=1) over the full "
                f"Qwen3-0.6B vocabulary ({len(vocab):,} tokens). For each seed token "
                "in the top-1281 cluster, collect all cluster-mates — capturing "
                "morphological variants (case, inflection, punctuation attachment) "
                "not present in the original Fighting Words cluster."
            ),
        },
        "summary": {
            "model": MODEL,
            "vocab_size": len(vocab),
            "seed_token_count": len(seed_id_to_meta),
            "expanded_token_count": len(expanded_ids),
            "treetok_additions": n_expanded,
            "seeds_with_expansions": sum(1 for c in cluster_details.values() if c["expanded_members"]),
            "score_at_cliff_before": score_cliff_before,
            "score_at_cliff_after": score_cliff_after,
            "score_cliff_gap": round(score_cliff_before - score_cliff_after, 6),
            "cliff_rank": CLIFF_RANK,
            "max_levenshtein_distance": 1,
        },
        "cluster_details": sorted(
            [cd for cd in cluster_details.values() if cd["expanded_members"]],
            key=lambda x: x["seed_members"][0]["rank"] if x["seed_members"] else 9999,
        ),
        "full_token_list": full_token_list,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {OUTPUT_PATH}", flush=True)

    # Print top expansions for review
    print("\nTop-30 seed expansions via treetok:")
    for cd in output["cluster_details"][:30]:
        seeds = ", ".join(repr(m["token"]) for m in cd["seed_members"][:2])
        exps  = ", ".join(repr(m["token"]) for m in cd["expanded_members"][:6])
        print(f"  [{seeds}] -> {exps}")


if __name__ == "__main__":
    main()
