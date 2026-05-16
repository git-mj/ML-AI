import os
import joblib
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

def initialize_nltk():
    nltk.download("punkt", quiet=True)

class CountService:
    # Class-level variable to hold the model in memory
    ml_pipeline = None

    @classmethod
    def load_model(cls):
        """Loads the ML model into memory. Called once during app startup."""
        if cls.ml_pipeline is None:
            # Construct absolute path to the model file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, 'models', 'sentiment_pipeline.joblib')
            
            # Load the model
            cls.ml_pipeline = joblib.load(model_path)
            print("ML Model loaded successfully.")

    @classmethod
    def count_text(cls, text: str) -> dict:
        if not text or not text.strip():
            return {"total_words": 0, "sentiment": "unknown"}

        # 1. Rule-based NLTK analysis (from your original code)
        sentences = sent_tokenize(text)
        words = [t.lower() for t in word_tokenize(text) if t.isalpha()]
        
        # 2. Machine Learning Inference
        # We pass the text inside a list because scikit-learn expects an array-like input
        #predicted_sentiment = cls.ml_pipeline.predict([text])[0]
        
        # Optional: Get confidence score (probabilities)
        #probabilities = cls.ml_pipeline.predict_proba([text])[0]
        #confidence = max(probabilities) * 100

        return {
            "total_words": len(words),
            "unique_words": len(set(words)),
            "sentence_count": len(sentences),
            "vocabulary_richness": round(len(set(words)) / len(words) * 100, 1) if words else 0
         #   "ml_analysis": {
          #      "sentiment": predicted_sentiment,
          #      "confidence_score": round(confidence, 2)
            }
