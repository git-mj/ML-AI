"""
Section A: Word2Vec + ANN
Trains a Word2Vec model on the Brown Corpus and uses averaged word vectors
as document embeddings to train a Keras ANN text classifier.
"""
import os
import sys
import numpy as np
import joblib
import nltk
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from categorypredictor.scripts.preprocess import ensure_nltk, preprocess_tokens

print("=== Section A: Word2Vec + ANN Training ===\n")

ensure_nltk()
from nltk.corpus import brown

# -----------------------------------------------------------------------
# Map the 15 Brown Corpus categories to 7 broad classes
# -----------------------------------------------------------------------
CATEGORY_MAP = {
    'news':             'news',
    'editorial':        'editorial',
    'fiction':          'fiction',
    'mystery':          'fiction',
    'adventure':        'fiction',
    'romance':          'fiction',
    'science_fiction':  'science_fiction',
    'government':       'government',
    'learned':          'academic',
    'lore':             'academic',
    'hobbies':          'lifestyle',
    'reviews':          'lifestyle',
    'religion':         'lifestyle',
    'belles_lettres':   'lifestyle',
}

# -----------------------------------------------------------------------
# 1. Load and preprocess documents (one document per file in the corpus)
# -----------------------------------------------------------------------
print("1. Loading and preprocessing Brown Corpus documents...")
documents = []
labels = []

for category in brown.categories():
    broad_label = CATEGORY_MAP.get(category)
    if broad_label is None:
        continue
    for fileid in brown.fileids(categories=category):
        tokens = brown.words(fileid)
        cleaned = preprocess_tokens(list(tokens))
        if len(cleaned) > 10:
            documents.append(cleaned)
            labels.append(broad_label)

print(f"   Loaded {len(documents)} documents across {len(set(labels))} classes: {set(labels)}")

# -----------------------------------------------------------------------
# 2. Train Word2Vec on corpus sentences
# -----------------------------------------------------------------------
print("\n2. Training Word2Vec model (vector_size=300, epochs=10)...")
all_sentences = [preprocess_tokens(list(brown.words(fid))) for fid in brown.fileids()]
w2v_model = Word2Vec(
    sentences=all_sentences,
    vector_size=300,
    window=5,
    min_count=2,
    workers=4,
    epochs=10
)

model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(model_dir, exist_ok=True)

w2v_path = os.path.join(model_dir, 'word2vec.model')
w2v_model.save(w2v_path)
print(f"   Word2Vec model saved to: {w2v_path}")

# -----------------------------------------------------------------------
# 3. Compute document embeddings (average of word vectors)
# -----------------------------------------------------------------------
print("\n3. Computing document embeddings (averaging word vectors)...")

def get_doc_embedding(tokens, model, vector_size=300):
    """Average the Word2Vec vectors for all known tokens in the document."""
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

X = np.array([get_doc_embedding(doc, w2v_model) for doc in documents])
print(f"   Embedding matrix shape: {X.shape}")

# -----------------------------------------------------------------------
# 4. Encode labels and save the encoder
# -----------------------------------------------------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
num_classes = len(encoder.classes_)

enc_path = os.path.join(model_dir, 'word2vec_label_encoder.joblib')
joblib.dump(encoder, enc_path)
print(f"   Label encoder saved. Classes: {list(encoder.classes_)}")

# -----------------------------------------------------------------------
# 5. Train / Test split
# -----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training samples: {len(X_train)} | Test samples: {len(X_test)}")

# -----------------------------------------------------------------------
# 6. Build and train the Keras ANN
# -----------------------------------------------------------------------
print("\n4. Building and training Keras ANN...")
model = keras.Sequential([
    keras.layers.Input(shape=(300,)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')
], name="word2vec_ann")

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# -----------------------------------------------------------------------
# 7. Evaluate and Save
# -----------------------------------------------------------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n[RESULT] Word2Vec ANN Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

ann_path = os.path.join(model_dir, 'word2vec_ann.keras')
model.save(ann_path)
print(f"Keras ANN model saved to: {ann_path}")
print("\n=== Section A Complete ===")
