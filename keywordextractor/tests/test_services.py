import pytest
import sys
import os

# Add root project dir to path so tests can run independently
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from keywordextractor.app.services import KeywordExtractorService, initialize_nltk

# Ensure NLTK resources are available for tests
initialize_nltk()
KeywordExtractorService.load_model()

def test_extract_empty_string():
    results = KeywordExtractorService.extract_keywords("")
    assert len(results) == 0

def test_extract_keywords():
    text = "The quick brown fox jumps over the lazy dog. The dog barks at the fox. Foxes are smart."
    results = KeywordExtractorService.extract_keywords(text, top_n=5)
    
    # Should identify fox and dog as primary nouns
    keywords = [r["keyword"] for r in results]
    assert 'fox' in keywords or 'foxes' in keywords
    assert 'dog' in keywords
    
def test_stopwords_removed():
    text = "The and is in on at to with"
    results = KeywordExtractorService.extract_keywords(text)
    # Stopwords should not be categorized as nouns, and if they are, they should be filtered
    assert len(results) == 0
