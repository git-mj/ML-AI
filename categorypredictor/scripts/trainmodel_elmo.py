"""
Section B: ELMo + ANN
Uses a pre-trained ELMo model from TensorFlow Hub to produce contextual
word embeddings, averages them per document, then trains a Keras ANN.

ELMo embeddings are pre-computed once and cached to disk so subsequent
runs skip the expensive forward pass (may take several minutes on first run).
"""
import os
import sys
import numpy as np
import joblib
import nltk
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from categorypredictor.scripts.preprocess import ensure_nltk, tokens_to_string

print("=== Section B: ELMo + ANN Training ===\n")

ensure_nltk()
from nltk.corpus import brown

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

model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(model_dir, exist_ok=True)

cache_path = os.path.join(model_dir, 'elmo_embeddings_cache.npy')
labels_cache_path = os.path.join(model_dir, 'elmo_labels_cache.joblib')

# -----------------------------------------------------------------------
# 1. Load and preprocess documents
# -----------------------------------------------------------------------
print("1. Loading and preprocessing Brown Corpus documents...")
documents_str = []
labels = []

for category in brown.categories():
    broad_label = CATEGORY_MAP.get(category)
    if broad_label is None:
        continue
    for fileid in brown.fileids(categories=category):
        text = tokens_to_string(list(brown.words(fileid)))
        if len(text.split()) > 5:
            documents_str.append(text)
            labels.append(broad_label)

print(f"   Loaded {len(documents_str)} documents across {len(set(labels))} classes.")

# -----------------------------------------------------------------------
# 2. Load ELMo from TensorFlow Hub
# -----------------------------------------------------------------------
print("\n2. Loading ELMo from TensorFlow Hub...")
print("   (First run will download ~350 MB — subsequent runs use local cache)")
try:
    elmo = hub.load("https://tfhub.dev/google/elmo/3")
    print("   ELMo loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load ELMo from TensorFlow Hub: {e}")
    print("Please ensure you have an active internet connection and tensorflow-hub installed.")
    sys.exit(1)

# -----------------------------------------------------------------------
# 3. Pre-compute or load cached ELMo embeddings
# -----------------------------------------------------------------------
if os.path.exists(cache_path) and os.path.exists(labels_cache_path):
    print("\n3. Loading cached ELMo embeddings from disk (skipping recomputation)...")
    X = np.load(cache_path)
    labels = joblib.load(labels_cache_path)
    print(f"   Loaded cached embeddings shape: {X.shape}")
else:
    print("\n3. Pre-computing ELMo embeddings (this may take several minutes)...")
    print("   ELMo produces 1024-dim contextual embeddings averaged per document.\n")

    def get_elmo_embedding(text: str) -> np.ndarray:
        """Runs text through ELMo and averages the token embeddings."""
        outputs = elmo.signatures['default'](tf.constant([text]))
        # outputs['elmo'] shape: [1, seq_len, 1024]
        word_embeddings = outputs['elmo'].numpy()[0]  # [seq_len, 1024]
        return np.mean(word_embeddings, axis=0)       # [1024]

    X = np.array([
        get_elmo_embedding(text)
        for text in tqdm(documents_str, desc="ELMo Embeddings")
    ])

    np.save(cache_path, X)
    joblib.dump(labels, labels_cache_path)
    print(f"\n   ELMo embeddings cached to: {cache_path}")

print(f"   Embedding matrix shape: {X.shape}")

# -----------------------------------------------------------------------
# 4. Encode labels and save encoder
# -----------------------------------------------------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
num_classes = len(encoder.classes_)

enc_path = os.path.join(model_dir, 'elmo_label_encoder.joblib')
joblib.dump(encoder, enc_path)
print(f"\n   Label encoder saved. Classes: {list(encoder.classes_)}")

# -----------------------------------------------------------------------
# 5. Train / Test split
# -----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training samples: {len(X_train)} | Test samples: {len(X_test)}")

# -----------------------------------------------------------------------
# 6. Build and train the Keras ANN (1024-dim ELMo input)
# -----------------------------------------------------------------------
print("\n4. Building and training Keras ANN (1024-dim ELMo input)...")
model = keras.Sequential([
    keras.layers.Input(shape=(1024,)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')
], name="elmo_ann")

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
print(f"\n[RESULT] ELMo ANN Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

ann_path = os.path.join(model_dir, 'elmo_ann.keras')
model.save(ann_path)
print(f"Keras ANN model saved to: {ann_path}")
print("\n=== Section B Complete ===")
