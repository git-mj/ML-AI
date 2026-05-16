import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag.perceptron import PerceptronTagger
from collections import Counter
import string

def initialize_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("stopwords", quiet=True)

class KeywordExtractorService:
    tagger = None
    stop_words = None

    @classmethod
    def load_model(cls):
        """Loads the pre-trained NLTK PerceptronTagger and stopwords into memory."""
        if cls.tagger is None:
            cls.tagger = PerceptronTagger()
            cls.stop_words = set(stopwords.words('english'))
            print("Keyword Extractor models loaded successfully.")

    @classmethod
    def extract_keywords(cls, text: str, top_n: int = 10) -> list:
        if not text or not text.strip():
            return []

        words = word_tokenize(text)
        
        if cls.tagger:
            tagged_words = cls.tagger.tag(words)
        else:
            tagged_words = nltk.pos_tag(words)
            
        if cls.stop_words is None:
            cls.stop_words = set(stopwords.words('english'))

        valid_nouns = []
        for word, tag in tagged_words:
            word_lower = word.lower()
            # Filter criteria:
            # 1. Is a noun (NN, NNS, NNP, NNPS)
            # 2. Not a stop word
            # 3. Consists of alphabetic characters
            if tag.startswith("NN") and word_lower not in cls.stop_words:
                if word.isalpha():
                    valid_nouns.append(word_lower)

        # Count frequencies
        word_counts = Counter(valid_nouns)
        
        # Format output as a list of dicts: [{"keyword": "apple", "count": 5}, ...]
        result = [{"keyword": word, "count": count} for word, count in word_counts.most_common(top_n)]
        
        return result
