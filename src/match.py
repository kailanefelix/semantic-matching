"""CLI entry-point: load CSVs, run semantic matching, save output."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from embedder import SkillEmbedder
from matcher import SkillMatcher


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_skills(path: str) -> pd.DataFrame:
    """Load new_skills.csv (well-formed, two columns: id, skill_raw)."""
    return pd.read_csv(path, encoding="utf-8")


def load_taxonomy(path: str) -> pd.DataFrame:
    """Load skill_taxonomy.csv.

    The description field contains unquoted commas, so the file is parsed
    by splitting each line on the *first two* commas only.
    """
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    rows: list[dict] = []
    for line in lines[1:]:  # skip header
        if not line.strip():
            continue
        parts = line.split(",", 2)
        rows.append(
            {
                "id": int(parts[0]),
                "skill_name": parts[1],
                "description": parts[2] if len(parts) > 2 else "",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semantic skill matching: map raw skills to a standardised taxonomy."
    )
    parser.add_argument("--skills", required=True, help="Path to new_skills.csv")
    parser.add_argument("--taxonomy", required=True, help="Path to skill_taxonomy.csv")
    parser.add_argument("--output", required=True, help="Path for the output CSV")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Cosine similarity threshold for accepting a match (default: 0.60)",
    )
    parser.add_argument(
        "--model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-transformers model name",
    )
    return parser


def print_summary(results: pd.DataFrame, threshold: float) -> None:
    total = len(results)
    matched = (results["match_status"] == "matched").sum()
    no_match = total - matched
    avg_score = results.loc[results["match_status"] == "matched", "score"].mean()

    print("\n" + "=" * 50)
    print("  Semantic Matching — Summary")
    print("=" * 50)
    print(f"  Total skills processed : {total}")
    print(f"  Matched (>= {threshold:.2f})       : {matched}  ({matched/total:.0%})")
    print(f"  No match               : {no_match}  ({no_match/total:.0%})")
    if matched > 0:
        print(f"  Avg score (matched)    : {avg_score:.4f}")
    print("=" * 50 + "\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print(f"Loading skills from   : {args.skills}")
    skills_df = load_skills(args.skills)

    print(f"Loading taxonomy from : {args.taxonomy}")
    taxonomy_df = load_taxonomy(args.taxonomy)

    print(f"Model                 : {args.model}")
    print(f"Threshold             : {args.threshold}")

    embedder = SkillEmbedder(model_name=args.model)
    matcher = SkillMatcher(embedder=embedder, threshold=args.threshold)

    print("\nEmbedding raw skills …")
    print("Embedding taxonomy …")
    results = matcher.match(skills_df, taxonomy_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Output saved to       : {args.output}")

    print_summary(results, args.threshold)


if __name__ == "__main__":
    main()
