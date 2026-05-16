import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from categorypredictor.app.services import CategoryPredictorService

def test_empty_string():
    result = CategoryPredictorService.predict("", "word2vec")
    assert result["category"] == "Unknown"

def test_missing_model_word2vec():
    # Reset models to simulate missing state
    CategoryPredictorService.w2v_model = None
    CategoryPredictorService.w2v_ann = None
    result = CategoryPredictorService.predict("the dragon cast a spell", "word2vec")
    assert result["category"] == "Model Missing"

def test_preprocess():
    tokens = CategoryPredictorService._preprocess(
        "The quick brown fox jumps over the lazy dog"
    )
    # Stopwords like 'the', 'over' should be removed
    assert "the" not in tokens
    assert "fox" in tokens
