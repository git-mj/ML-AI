import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from admissionchatbot.app.services import AdmissionChatbotService

# Pre-load the dataset
AdmissionChatbotService.load_model()

def test_tuition_question():
    # Minor typos and extra words
    result = AdmissionChatbotService.get_answer("how much is tution cost plz?")
    assert "Tuition is approximately $15,000" in result["answer"]
    assert result["confidence"] > 60

def test_deadline_question():
    result = AdmissionChatbotService.get_answer("tell me the deadlin for applications")
    assert "The deadline for Fall admission is January 15th" in result["answer"]
    assert result["confidence"] > 60

def test_fallback_unrelated_question():
    result = AdmissionChatbotService.get_answer("what is the cafeteria food like?")
    assert "admissions-support@university.edu" in result["answer"]
    assert result["confidence"] < 60
