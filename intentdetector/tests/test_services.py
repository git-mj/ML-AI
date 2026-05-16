import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from intentdetector.app.services import IntentDetectorService

# Pre-load the mock dataset
IntentDetectorService.load_model()

def test_empty_string():
    result = IntentDetectorService.detect_intent("")
    assert result["intent"] == "Unknown"

def test_minor_misspelling_password():
    # Misspelled "password" as "pssword"
    result = IntentDetectorService.detect_intent("I forgot my pssword")
    assert result["domain"] == "Access"
    assert result["intent"] == "Password Reset"
    assert result["confidence"] > 60

def test_vpn_intent():
    # Misspelled "network" as "netwrk"
    result = IntentDetectorService.detect_intent("My netwrk is offline")
    assert result["domain"] == "Access"
    assert result["intent"] == "VPN"
    assert result["confidence"] > 60

def test_gibberish():
    # Text completely unrelated to any keywords
    result = IntentDetectorService.detect_intent("xzyquerty bla bla")
    assert result["intent"] == "Unknown"
