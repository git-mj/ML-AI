import pytest
from app.services import CountService, initialize_nltk

initialize_nltk()

def test_analyze_empty_string():
    results = CountService.count_text("")
    assert results['total_words'] == 0

def test_analyze_richness_calculation():
    text = "Apple apple Banana" # 2 unique (apple, banana), 3 total
    results = CountService.count_text(text)
    assert results['total_words'] == 3
    assert results['unique_words'] == 2
