import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class WordSimilarityService:
    elmo_model = None

    @classmethod
    def load_model(cls):
        """Loads ELMo from TF Hub into memory lazily on first use.
        We don't load it at startup because it is a large model (~350 MB)
        and is shared with the categorypredictor service."""
        # ELMo is intentionally loaded lazily (on first request) via _ensure_model()
        # to avoid slowing down the Flask startup sequence.
        print("Word Similarity Service registered (ELMo loads on first request).")

    @classmethod
    def _ensure_model(cls):
        """Lazily loads ELMo if not already loaded."""
        if cls.elmo_model is None:
            import tensorflow_hub as hub
            print("Loading ELMo for Word Similarity (one-time download)...")
            cls.elmo_model = hub.load("https://tfhub.dev/google/elmo/3").signatures["default"]
            print("Word Similarity ELMo model loaded.")

    @classmethod
    def _tokenize(cls, text: str) -> list:
        """Simple whitespace tokenizer that strips common punctuation."""
        for ch in ['.', ',', '?', '!', ';', ':']:
            text = text.replace(ch, '')
        return text.lower().split()

    @classmethod
    def calculate_similarity(cls, sentence1: str, sentence2: str, word: str) -> dict:
        """
        Computes the cosine similarity of a specific word's ELMo contextual
        embedding across two different sentences.

        Returns a dict with the similarity score and metadata.
        """
        import tensorflow as tf

        cls._ensure_model()

        word_target = word.lower().strip()
        tokens1 = cls._tokenize(sentence1)
        tokens2 = cls._tokenize(sentence2)

        if word_target not in tokens1:
            return {
                "error": f"The word '{word_target}' was not found in Sentence 1.",
                "similarity": None
            }
        if word_target not in tokens2:
            return {
                "error": f"The word '{word_target}' was not found in Sentence 2.",
                "similarity": None
            }

        idx1 = tokens1.index(word_target)
        idx2 = tokens2.index(word_target)

        input_tensor = tf.constant([sentence1, sentence2])
        result = cls.elmo_model(input_tensor)
        embeddings = result["elmo"]

        vec1 = embeddings[0, idx1, :].numpy().reshape(1, -1)
        vec2 = embeddings[1, idx2, :].numpy().reshape(1, -1)

        score = float(cosine_similarity(vec1, vec2)[0][0])

        return {
            "similarity": round(score, 4),
            "word": word_target,
            "sentence1": sentence1,
            "sentence2": sentence2,
            "error": None
        }
