"""Matching module: cosine similarity + threshold decision."""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from embedder import SkillEmbedder


class SkillMatcher:
    """Matches raw skills to a taxonomy using cosine similarity on embeddings."""

    def __init__(self, embedder: SkillEmbedder, threshold: float = 0.60) -> None:
        """Initialise matcher.

        Args:
            embedder: A ready-to-use SkillEmbedder instance.
            threshold: Minimum cosine similarity to accept a match.
        """
        self.embedder = embedder
        self.threshold = threshold

    def match(
        self,
        new_skills_df: pd.DataFrame,
        taxonomy_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Match every raw skill to the closest taxonomy entry.

        Args:
            new_skills_df: DataFrame with columns ``id`` and ``skill_raw``.
            taxonomy_df: DataFrame with columns ``id``, ``skill_name``, ``description``.

        Returns:
            DataFrame with columns:
            ``id``, ``skill_raw``, ``taxonomy_id``, ``taxonomy_name``,
            ``score``, ``match_status`` (``"matched"`` or ``"no_match"``).
        """
        skill_texts = new_skills_df["skill_raw"].tolist()
        skill_embs = self.embedder.embed(skill_texts)
        taxonomy_embs = self.embedder.embed_taxonomy(taxonomy_df)

        # Shape: (n_skills, n_taxonomy)
        sim_matrix = cosine_similarity(skill_embs, taxonomy_embs)

        records: list[dict] = []
        taxonomy_ids = taxonomy_df["id"].tolist()
        taxonomy_names = taxonomy_df["skill_name"].tolist()

        for i, (_, row) in enumerate(new_skills_df.iterrows()):
            best_idx = int(np.argmax(sim_matrix[i]))
            best_score = float(sim_matrix[i, best_idx])

            if best_score >= self.threshold:
                records.append(
                    {
                        "id": row["id"],
                        "skill_raw": row["skill_raw"],
                        "taxonomy_id": taxonomy_ids[best_idx],
                        "taxonomy_name": taxonomy_names[best_idx],
                        "score": round(best_score, 4),
                        "match_status": "matched",
                    }
                )
            else:
                records.append(
                    {
                        "id": row["id"],
                        "skill_raw": row["skill_raw"],
                        "taxonomy_id": "",
                        "taxonomy_name": "",
                        "score": round(best_score, 4),
                        "match_status": "no_match",
                    }
                )

        return pd.DataFrame(records)
