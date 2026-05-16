import pytest
import numpy as np
from nlp_service import WordSimilarityModel

@pytest.fixture
def mock_model(mocker):
    """Fixture to create the service with a mocked ELMo model."""
    # Patch the hub.load call so we don't actually download the model
    mocker.patch("tensorflow_hub.load")
    service = WordSimilarityModel()
    
    # Mock the return signature of the model
    # ELMo returns a dict with an "elmo" key containing a 3D tensor
    # Shape: (batch_size, max_sequence_length, embedding_dim)
    mock_output = {
        "elmo": np.random.rand(2, 10, 1024).astype(np.float32)
    }
    service.model = mocker.Mock(return_value=mock_output)
    return service

def test_tokenization_logic(mock_model):
    """Test that the internal tokenizer handles basic punctuation."""
    text = "Hello, world!"
    tokens = mock_model._simple_tokenize(text)
    assert tokens == ["hello", "world"]

def test_calculate_similarity_missing_word(mock_model):
    """Ensure it raises ValueError if the word isn't in both sentences."""
    with pytest.raises(ValueError, match="must be present in both sentences"):
        mock_model.calculate_similarity("The cat sat", "The dog ran", "cat")

def test_calculate_similarity_success(mock_model):
    """Test the full flow with mocked embeddings."""
    score = mock_model.calculate_similarity(
        "I like apples", 
        "Apples are red", 
        "apples"
    )
    assert isinstance(score, float)
    # Since we used random numbers in the mock, we just check it returned a float