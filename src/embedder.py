"""Embedding module: loads a sentence-transformers model and generates skill embeddings."""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class SkillEmbedder:
    """Wraps a sentence-transformers model for generating skill embeddings."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        """Load the sentence-transformers model."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into L2-normalised embeddings.

        Args:
            texts: Raw strings to embed.

        Returns:
            Float32 matrix of shape (len(texts), embedding_dim).
        """
        return self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_taxonomy(self, df: pd.DataFrame) -> np.ndarray:
        """Embed taxonomy rows by concatenating skill_name + ': ' + description.

        Expects *df* to have columns ``skill_name`` and ``description``.
        When description is missing or empty only ``skill_name`` is used.

        Returns:
            Float32 matrix of shape (len(df), embedding_dim).
        """
        texts: list[str] = []
        for _, row in df.iterrows():
            desc = str(row.get("description", "") or "").strip()
            name = str(row["skill_name"]).strip()
            text = f"{name}: {desc}" if desc else name
            texts.append(text)
        return self.embed(texts)
