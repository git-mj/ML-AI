import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from wordsimilarity.app.services import WordSimilarityService


def test_tokenization_logic():
    text = "Hello, world!"
    tokens = WordSimilarityService._tokenize(text)
    assert tokens == ["hello", "world"]


def test_missing_word_returns_error(monkeypatch):
    monkeypatch.setattr(WordSimilarityService, "_ensure_model", classmethod(lambda cls: None))
    WordSimilarityService.elmo_model = lambda _: None

    result = WordSimilarityService.calculate_similarity("The cat sat", "The dog ran", "cat")

    assert result["similarity"] is None
    assert "Sentence 2" in result["error"]


def test_remote_downloads_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ML_AI_ALLOW_REMOTE_MODEL_DOWNLOADS", raising=False)
    monkeypatch.delenv("ELMO_MODEL_URL", raising=False)
    WordSimilarityService.elmo_model = None

    result = WordSimilarityService.calculate_similarity("I like apples", "Apples are red", "apples")

    assert result["similarity"] is None
    assert "Remote ELMo model downloads are disabled" in result["error"]
