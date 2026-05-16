import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class WordSimilarityModel:
    def __init__(self, model_url: str = "https://tfhub.dev/google/elmo/3"):
        logger.info("Loading ELMo model... This may take a moment.")
        # Load the model during class initialization
        self.model = hub.load(model_url).signatures["default"]
        logger.info("ELMo model loaded successfully.")

    def _simple_tokenize(self, text: str) -> list[str]:
        """
        Simple whitespace tokenization. 
        Note: For production, consider passing pre-tokenized arrays to ELMo's 
        'tokens' signature to guarantee perfect index alignment.
        """
        # Removing punctuation to avoid "bank." vs "bank" mismatches, 
        # while keeping the token count aligned with a basic split.
        clean_text = text.replace('.', '').replace(',', '').replace('?', '').replace('!', '')
        return clean_text.lower().split()

    def calculate_similarity(self, sentence1: str, sentence2: str, word: str) -> float:
        """Calculates cosine similarity of a specific word across two sentences."""
        word_target = word.lower().strip()
        
        tokens1 = self._simple_tokenize(sentence1)
        tokens2 = self._simple_tokenize(sentence2)

        if word_target not in tokens1 or word_target not in tokens2:
            raise ValueError(f"Target word '{word_target}' must be present in both sentences.")

        idx1 = tokens1.index(word_target)
        idx2 = tokens2.index(word_target)

        # Create tensor and run inference
        input_tensor = tf.constant([sentence1, sentence2]) 
        result = self.model(input_tensor)
        embeddings = result["elmo"] 

        # Extract vectors and reshape for sklearn
        vec1 = embeddings[0, idx1, :].numpy().reshape(1, -1)
        vec2 = embeddings[1, idx2, :].numpy().reshape(1, -1)

        return float(cosine_similarity(vec1, vec2)[0][0])