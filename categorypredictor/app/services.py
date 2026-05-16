import os
import string
import numpy as np
import joblib

class CategoryPredictorService:
    # ---- Word2Vec models ----
    w2v_model = None       # gensim Word2Vec
    w2v_ann = None         # Keras ANN
    w2v_encoder = None     # sklearn LabelEncoder

    # ---- ELMo models (loaded lazily to avoid slow startup) ----
    elmo_hub_model = None  # TF Hub ELMo
    elmo_ann = None        # Keras ANN
    elmo_encoder = None    # sklearn LabelEncoder

    @classmethod
    def _model_dir(cls):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, 'models')

    @classmethod
    def load_model(cls):
        """
        Loads the Word2Vec pipeline at startup (fast).
        ELMo is loaded lazily on the first ELMo prediction request
        to avoid a slow server startup.
        """
        cls._load_word2vec()

    @classmethod
    def _load_word2vec(cls):
        """Load gensim Word2Vec + Keras ANN + LabelEncoder into class vars."""
        try:
            from gensim.models import Word2Vec
            from tensorflow import keras

            d = cls._model_dir()
            w2v_path = os.path.join(d, 'word2vec.model')
            ann_path = os.path.join(d, 'word2vec_ann.keras')
            enc_path = os.path.join(d, 'word2vec_label_encoder.joblib')

            if all(os.path.exists(p) for p in [w2v_path, ann_path, enc_path]):
                cls.w2v_model   = Word2Vec.load(w2v_path)
                cls.w2v_ann     = keras.models.load_model(ann_path)
                cls.w2v_encoder = joblib.load(enc_path)
                print("Category Predictor (Word2Vec) models loaded successfully.")
            else:
                print("WARNING: Category Predictor Word2Vec models not found. "
                      "Run categorypredictor/scripts/trainmodel_word2vec.py first.")
        except Exception as e:
            print(f"WARNING: Could not load Word2Vec models: {e}")

    @classmethod
    def _load_elmo(cls):
        """Lazily load ELMo from TF Hub on first use."""
        try:
            import tensorflow_hub as hub
            from tensorflow import keras

            d = cls._model_dir()
            ann_path = os.path.join(d, 'elmo_ann.keras')
            enc_path = os.path.join(d, 'elmo_label_encoder.joblib')

            if not all(os.path.exists(p) for p in [ann_path, enc_path]):
                return False, "ELMo model files not found. Run trainmodel_elmo.py first."

            if cls.elmo_hub_model is None:
                print("Loading ELMo from TensorFlow Hub (one-time)...")
                cls.elmo_hub_model = hub.load("https://tfhub.dev/google/elmo/3")

            cls.elmo_ann     = keras.models.load_model(ann_path)
            cls.elmo_encoder = joblib.load(enc_path)
            return True, "ok"
        except Exception as e:
            return False, str(e)

    @classmethod
    def _preprocess(cls, text: str):
        """Tokenize, lowercase, remove stopwords & punctuation."""
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        tokens = text.lower().split()
        return [
            t.strip(string.punctuation) for t in tokens
            if t.strip(string.punctuation)
            and t.strip(string.punctuation) not in stop_words
            and t.strip(string.punctuation).isalpha()
        ]

    @classmethod
    def predict(cls, text: str, embedding: str = 'word2vec') -> dict:
        if not text or not text.strip():
            return {"category": "Unknown", "confidence": 0, "embedding": embedding}

        tokens = cls._preprocess(text)

        if embedding == 'elmo':
            return cls._predict_elmo(tokens)
        return cls._predict_word2vec(tokens)

    @classmethod
    def _predict_word2vec(cls, tokens: list) -> dict:
        if cls.w2v_model is None or cls.w2v_ann is None:
            return {"category": "Model Missing", "confidence": 0,
                    "embedding": "Word2Vec",
                    "error": "Run trainmodel_word2vec.py to generate models."}

        vectors = [cls.w2v_model.wv[t] for t in tokens if t in cls.w2v_model.wv]
        if not vectors:
            return {"category": "Unknown", "confidence": 0,
                    "embedding": "Word2Vec",
                    "error": "No vocabulary match found in the Word2Vec model."}

        doc_vec = np.mean(vectors, axis=0).reshape(1, -1)
        probs = cls.w2v_ann.predict(doc_vec, verbose=0)[0]
        idx = int(np.argmax(probs))
        return {
            "category":  cls.w2v_encoder.classes_[idx],
            "confidence": round(float(probs[idx]) * 100, 2),
            "embedding": "Word2Vec"
        }

    @classmethod
    def _predict_elmo(cls, tokens: list) -> dict:
        import tensorflow as tf

        ok, msg = cls._load_elmo()
        if not ok:
            return {"category": "Model Missing", "confidence": 0,
                    "embedding": "ELMo", "error": msg}

        clean_text = " ".join(tokens)
        if not clean_text:
            return {"category": "Unknown", "confidence": 0, "embedding": "ELMo"}

        outputs = cls.elmo_hub_model.signatures['default'](tf.constant([clean_text]))
        word_embs = outputs['elmo'].numpy()[0]          # [seq_len, 1024]
        doc_vec   = np.mean(word_embs, axis=0).reshape(1, -1)  # [1, 1024]

        probs = cls.elmo_ann.predict(doc_vec, verbose=0)[0]
        idx = int(np.argmax(probs))
        return {
            "category":  cls.elmo_encoder.classes_[idx],
            "confidence": round(float(probs[idx]) * 100, 2),
            "embedding": "ELMo"
        }
