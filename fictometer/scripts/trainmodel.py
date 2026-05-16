import os
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

print("Downloading NLTK Brown Corpus (if not already downloaded)...")
nltk.download('brown')
from nltk.corpus import brown

print("Extracting documents from the Brown Corpus...")

# We map genres to Fiction (1) and Non-Fiction (0)
fiction_categories = ['fiction', 'mystery', 'science_fiction', 'adventure', 'romance']
non_fiction_categories = ['news', 'editorial', 'reviews', 'religion', 'hobbies', 'lore', 'belles_lettres', 'government', 'learned']

documents = []
labels = []

# To keep the context meaningful, we will train on paragraphs rather than single sentences.
# NLTK's brown.paras() returns a list of paragraphs, where each paragraph is a list of sentences, where each sentence is a list of words.
for category in fiction_categories:
    for para in brown.paras(categories=category):
        # Flatten the paragraph into a single string
        words = [word for sentence in para for word in sentence]
        text = " ".join(words)
        if len(text) > 50: # Only use paragraphs with some substance
            documents.append(text)
            labels.append(1)

for category in non_fiction_categories:
    for para in brown.paras(categories=category):
        words = [word for sentence in para for word in sentence]
        text = " ".join(words)
        if len(text) > 50:
            documents.append(text)
            labels.append(0)

print(f"Extracted {len(documents)} paragraphs for training.")

print("Building the Scikit-Learn Pipeline...")
# Limit max_features to 10000 to prevent the exported .joblib file from becoming too massive
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
    ('classifier', LogisticRegression(max_iter=1000))
])

print("Training the Fictometer model... (this may take a minute)")
pipeline.fit(documents, labels)
print("Training complete.")

# Export the model
model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'fictometer_model.joblib')
joblib.dump(pipeline, model_path)

print(f"Model successfully saved to {model_path}")
