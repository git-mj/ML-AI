import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

def train_and_export():
    print("1. Loading training data...")
    # In reality, this would be thousands of rows loaded from a CSV or database
    X_train = [
        "Natural language processing is amazing and helpful.",
        "I absolutely love this new technology.",
        "This is terrible and completely useless.",
        "I hate when the system crashes.",
        "It is okay, nothing special but functional."
    ]
    y_train = ["positive", "positive", "negative", "negative", "neutral"]

    print("2. Building and training the model...")
    # A pipeline combines text vectorization (turning words to numbers) and the ML algorithm
    model_pipeline = make_pipeline(
        TfidfVectorizer(lowercase=True, stop_words='english'),
        MultinomialNB()
    )
    
    # Train the model
    model_pipeline.fit(X_train, y_train)

    print("3. Exporting the trained model...")
    # Create the models directory if it doesn't exist
    os.makedirs('../app/models', exist_ok=True)
    
    # Save the entire pipeline (vectorizer + trained model) to a file
    export_path = '../app/models/sentiment_pipeline.joblib'
    joblib.dump(model_pipeline, export_path)
    print(f"Model successfully saved to {export_path}")

if __name__ == "__main__":
    train_and_export()