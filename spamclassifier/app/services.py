import os
import joblib

class SpamClassifierService:
    lr_model = None
    nb_model = None

    @classmethod
    def load_model(cls):
        """Loads both the Logistic Regression and Naive Bayes models into memory."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        lr_path = os.path.join(base_dir, 'models', 'spam_lr_model.joblib')
        nb_path = os.path.join(base_dir, 'models', 'spam_nb_model.joblib')

        if cls.lr_model is None and os.path.exists(lr_path):
            cls.lr_model = joblib.load(lr_path)
            print("Spam Classifier (Logistic Regression) model loaded successfully.")
        elif not os.path.exists(lr_path):
            print(f"WARNING: Spam LR model not found at {lr_path}. Please run trainmodel.py.")

        if cls.nb_model is None and os.path.exists(nb_path):
            cls.nb_model = joblib.load(nb_path)
            print("Spam Classifier (Naive Bayes) model loaded successfully.")
        elif not os.path.exists(nb_path):
            print(f"WARNING: Spam NB model not found at {nb_path}. Please run trainmodel.py.")

    @classmethod
    def classify(cls, text: str, algorithm: str = "lr") -> dict:
        """
        Classifies the text as Spam or Ham using the chosen algorithm.
        algorithm: 'lr' for Logistic Regression, 'nb' for Naive Bayes.
        """
        if not text or not text.strip():
            return {"prediction": "Unknown", "confidence": 0, "algorithm": algorithm}

        # Select the model based on the algorithm flag
        if algorithm == "nb":
            model = cls.nb_model
            algo_name = "Naive Bayes"
        else:
            model = cls.lr_model
            algo_name = "Logistic Regression"

        if model is None:
            return {"prediction": "Model Missing", "confidence": 0, "algorithm": algo_name}

        # Predict: 0 = Ham, 1 = Spam
        pred_label = model.predict([text])[0]
        prediction_text = "Spam" if pred_label == 1 else "Ham"

        # Get confidence from predict_proba
        probabilities = model.predict_proba([text])[0]
        confidence = max(probabilities) * 100

        return {
            "prediction": prediction_text,
            "confidence": round(confidence, 2),
            "algorithm": algo_name
        }
