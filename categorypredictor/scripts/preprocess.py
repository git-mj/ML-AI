import string
import nltk

def ensure_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('brown', quiet=True)

def preprocess_tokens(tokens):
    """
    Cleans a list of word tokens:
    - Lowercases all words
    - Removes English stopwords
    - Removes punctuation and non-alphabetic tokens
    """
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    return [
        word.lower() for word in tokens
        if word.lower() not in stop_words
        and word not in string.punctuation
        and word.isalpha()
    ]

def tokens_to_string(tokens):
    """Cleans tokens and joins them into a single string (for ELMo input)."""
    return " ".join(preprocess_tokens(tokens))
