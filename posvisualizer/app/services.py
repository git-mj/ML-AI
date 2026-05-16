import nltk
from nltk.tokenize import word_tokenize
from nltk.tag.perceptron import PerceptronTagger

def initialize_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

class POSService:
    # Class-level variable to hold the pre-trained model in memory
    tagger = None

    @classmethod
    def load_model(cls):
        """Loads the pre-trained NLTK PerceptronTagger into memory. Called once during app startup."""
        if cls.tagger is None:
            # Explicitly load the pre-trained tagger into memory
            cls.tagger = PerceptronTagger()
            print("POS Tagger model loaded successfully.")

    @classmethod
    def analyze_text(cls, text: str) -> list:
        """
        Tokenizes the text and performs POS tagging.
        Returns a list of tuples: [(word, tag, category), ...]
        """
        if not text or not text.strip():
            return []

        words = word_tokenize(text)
        
        # Use the pre-loaded tagger for better performance
        if cls.tagger:
            tagged_words = cls.tagger.tag(words)
        else:
            # Fallback just in case
            tagged_words = nltk.pos_tag(words)

        result = []
        for word, tag in tagged_words:
            category = "Other"
            if tag.startswith("NN"):
                category = "Noun"
            elif tag.startswith("VB"):
                category = "Verb"
            elif tag.startswith("JJ"):
                category = "Adjective"
            elif tag.startswith("RB"):
                category = "Adverb"
            elif tag.startswith("PRP"):
                category = "Pronoun"
            
            result.append((word, tag, category))
            
        return result
