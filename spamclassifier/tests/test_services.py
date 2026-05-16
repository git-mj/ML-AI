import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from spamclassifier.app.services import SpamClassifierService

def test_missing_model_handling():
    # If the models aren't loaded, it should fail gracefully
    SpamClassifierService.lr_model = None
    SpamClassifierService.nb_model = None
    result = SpamClassifierService.classify("Test message", "lr")
    assert result["prediction"] == "Model Missing"

def test_empty_string():
    result = SpamClassifierService.classify("", "lr")
    assert result["prediction"] == "Unknown"
