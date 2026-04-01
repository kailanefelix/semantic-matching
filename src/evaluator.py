"""Evaluation module: synthetic ground truth, metrics, and diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from embedder import SkillEmbedder
from matcher import SkillMatcher

# ---------------------------------------------------------------------------
# Simple word-level Portuguese → English translation for paraphrase generation.
# Only covers the most common words found in HR / skills taxonomy contexts.
# ---------------------------------------------------------------------------
_PT_EN: dict[str, str] = {
    "gestão": "management",
    "de": "of",
    "pessoas": "people",
    "comunicação": "communication",
    "assertiva": "assertive",
    "resolução": "resolution",
    "conflitos": "conflicts",
    "inteligência": "intelligence",
    "emocional": "emotional",
    "negociação": "negotiation",
    "pensamento": "thinking",
    "crítico": "critical",
    "problemas": "problems",
    "tomada": "decision",
    "decisão": "making",
    "adaptabilidade": "adaptability",
    "tempo": "time",
    "liderança": "leadership",
    "feedback": "feedback",
    "planejamento": "planning",
    "estratégico": "strategic",
    "inovação": "innovation",
    "criatividade": "creativity",
    "trabalho": "work",
    "equipe": "team",
    "análise": "analysis",
    "dados": "data",
    "visualização": "visualization",
    "machine": "machine",
    "learning": "learning",
    "engenharia": "engineering",
    "produto": "product",
    "visão": "vision",
    "código": "code",
    "versionamento": "versioning",
    "computação": "computing",
    "em": "in",
    "nuvem": "cloud",
    "metodologias": "methodologies",
    "ágeis": "agile",
    "storytelling": "storytelling",
    "com": "with",
    "finanças": "finance",
    "corporativas": "corporate",
    "contabilidade": "accounting",
    "marketing": "marketing",
    "digital": "digital",
    "vendas": "sales",
    "atendimento": "customer",
    "cliente": "service",
    "gestao": "management",
}


def _translate_to_english(text: str) -> str:
    """Best-effort word-by-word Portuguese → English translation."""
    words = text.lower().split()
    translated = [_PT_EN.get(w, w) for w in words]
    return " ".join(translated)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_synthetic_ground_truth(taxonomy_df: pd.DataFrame) -> pd.DataFrame:
    """Generate paraphrase variations of each taxonomy skill for evaluation.

    For each taxonomy entry produces up to four variations:
    - Lowercase of the original name.
    - Single-character typo (second character removed).
    - Acronym / initials (multi-word skills).
    - English word-by-word translation of the name.

    Args:
        taxonomy_df: DataFrame with columns ``id`` and ``skill_name``.

    Returns:
        DataFrame with columns ``skill_raw`` and ``true_taxonomy_id``.
    """
    rows: list[dict] = []

    for _, skill in taxonomy_df.iterrows():
        name: str = str(skill["skill_name"]).strip()
        tid = skill["id"]

        # Variation 1: lowercase
        rows.append({"skill_raw": name.lower(), "true_taxonomy_id": tid})

        # Variation 2: typo — drop the second character
        if len(name) > 3:
            typo = name[0] + name[2:]
            rows.append({"skill_raw": typo, "true_taxonomy_id": tid})

        # Variation 3: acronym (first letter of each word that has > 2 chars)
        words = name.split()
        if len(words) > 1:
            abbr = "".join(w[0].upper() for w in words if len(w) > 2)
            if len(abbr) > 1:
                rows.append({"skill_raw": abbr, "true_taxonomy_id": tid})

        # Variation 4: English translation
        en_name = _translate_to_english(name)
        if en_name.lower() != name.lower():
            rows.append({"skill_raw": en_name, "true_taxonomy_id": tid})

    return pd.DataFrame(rows).drop_duplicates(subset=["skill_raw"])


def evaluate(
    matcher_results: pd.DataFrame,
    ground_truth: pd.DataFrame,
    threshold: float | None = None,
) -> dict:
    """Compute quality metrics by comparing matcher output with ground truth.

    Args:
        matcher_results: Output of ``SkillMatcher.match()``.
        ground_truth: DataFrame from ``generate_synthetic_ground_truth()``.
        threshold: Threshold value used (stored in the returned dict).

    Returns:
        Dictionary with keys: precision, recall, f1, coverage,
        avg_score_matched, avg_score_no_match, threshold_used.
    """
    merged = matcher_results.merge(ground_truth, on="skill_raw", how="inner")

    if merged.empty:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "coverage": 0.0,
            "avg_score_matched": 0.0,
            "avg_score_no_match": 0.0,
            "threshold_used": threshold,
        }

    matched_mask = merged["match_status"] == "matched"
    matched = merged[matched_mask]
    no_match = merged[~matched_mask]

    # Correct = matched AND the taxonomy_id is correct
    correct_mask = (
        matched["taxonomy_id"].astype(str) == matched["true_taxonomy_id"].astype(str)
    )
    n_correct = int(correct_mask.sum())

    precision = n_correct / len(matched) if len(matched) > 0 else 0.0
    recall = n_correct / len(merged)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Coverage over the *full* matcher_results (not just what matched ground truth)
    coverage = (
        (matcher_results["match_status"] == "matched").sum() / len(matcher_results)
        if len(matcher_results) > 0
        else 0.0
    )

    avg_score_matched = float(matched["score"].mean()) if len(matched) > 0 else 0.0
    avg_score_no_match = float(no_match["score"].mean()) if len(no_match) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "coverage": round(coverage, 4),
        "avg_score_matched": round(avg_score_matched, 4),
        "avg_score_no_match": round(avg_score_no_match, 4),
        "threshold_used": threshold,
    }


def plot_score_distribution(matcher_results: pd.DataFrame) -> None:
    """Plot a histogram of cosine similarity scores, split by match status.

    Saves the figure to ``data/output/score_distribution.png``.
    This plot is the primary health signal for the system in production.

    Args:
        matcher_results: Output of ``SkillMatcher.match()``.
    """
    import matplotlib.pyplot as plt

    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    matched = matcher_results[matcher_results["match_status"] == "matched"]["score"]
    no_match = matcher_results[matcher_results["match_status"] == "no_match"]["score"]

    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(0, 1, 26)
    ax.hist(matched, bins=bins, alpha=0.7, color="#2196F3", label=f"matched (n={len(matched)})")
    ax.hist(no_match, bins=bins, alpha=0.7, color="#F44336", label=f"no_match (n={len(no_match)})")

    ax.set_xlabel("Cosine similarity score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Score distribution: matched vs. no_match", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "score_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_path}")


def precision_recall_curve_by_threshold(
    new_skills_df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    embedder: SkillEmbedder,
    ground_truth: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate matcher quality across a range of thresholds.

    Embeddings are computed *once* and the threshold is varied over the
    pre-computed similarity matrix to keep runtime reasonable.

    Args:
        new_skills_df: DataFrame with columns ``id`` and ``skill_raw``.
        taxonomy_df: Taxonomy DataFrame.
        embedder: Fitted SkillEmbedder instance.
        ground_truth: Output of ``generate_synthetic_ground_truth()``.
        thresholds: List of threshold values to test.
                    Defaults to 0.30 → 0.90 in steps of 0.05.

    Returns:
        DataFrame with columns: threshold, precision, recall, f1, coverage.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.30, 0.91, 0.05)]

    # Compute embeddings once
    skill_embs = embedder.embed(new_skills_df["skill_raw"].tolist())
    taxonomy_embs = embedder.embed_taxonomy(taxonomy_df)
    sim_matrix = cosine_similarity(skill_embs, taxonomy_embs)

    taxonomy_ids = taxonomy_df["id"].tolist()
    taxonomy_names = taxonomy_df["skill_name"].tolist()

    rows: list[dict] = []
    for thresh in tqdm(thresholds, desc="Sweeping thresholds"):
        records: list[dict] = []
        for i, (_, skill_row) in enumerate(new_skills_df.iterrows()):
            best_idx = int(np.argmax(sim_matrix[i]))
            best_score = float(sim_matrix[i, best_idx])

            if best_score >= thresh:
                records.append(
                    {
                        "skill_raw": skill_row["skill_raw"],
                        "taxonomy_id": taxonomy_ids[best_idx],
                        "score": round(best_score, 4),
                        "match_status": "matched",
                    }
                )
            else:
                records.append(
                    {
                        "skill_raw": skill_row["skill_raw"],
                        "taxonomy_id": "",
                        "score": round(best_score, 4),
                        "match_status": "no_match",
                    }
                )

        results_df = pd.DataFrame(records)
        metrics = evaluate(results_df, ground_truth, threshold=thresh)
        rows.append(
            {
                "threshold": thresh,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "coverage": metrics["coverage"],
            }
        )

    return pd.DataFrame(rows)
