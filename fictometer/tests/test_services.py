import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from fictometer.app.services import FictometerService

def test_missing_model_handling():
    # If the model isn't loaded (or doesn't exist), it should fail gracefully
    FictometerService.ml_pipeline = None
    result = FictometerService.analyze_text("Test string")
    assert result["prediction"] == "Model Missing"

# Additional tests would require the model to be trained first,
# so we avoid executing model inference in pure unit tests 
# to prevent breaking CI/CD if the model hasn't been generated yet.
