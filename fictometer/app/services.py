import os
import joblib

class FictometerService:
    ml_pipeline = None

    @classmethod
    def load_model(cls):
        """Loads the pre-trained Scikit-Learn joblib model into memory."""
        if cls.ml_pipeline is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'models', 'fictometer_model.joblib')
            
            if os.path.exists(model_path):
                cls.ml_pipeline = joblib.load(model_path)
                print("Fictometer ML Model loaded successfully.")
            else:
                print(f"WARNING: Fictometer model not found at {model_path}. Please run trainmodel.py.")

    @classmethod
    def analyze_text(cls, text: str) -> dict:
        """
        Classifies the text as Fiction or Non-Fiction using the trained ML pipeline.
        Returns the prediction and the confidence probability.
        """
        if not text or not text.strip():
            return {"prediction": "Unknown", "confidence": 0}

        if cls.ml_pipeline is None:
            # Fallback if model wasn't loaded (e.g. training script not run yet)
            return {"prediction": "Model Missing", "confidence": 0}

        # Predict returns an array, we grab the first item. 1 = Fiction, 0 = Non-Fiction
        pred_label = cls.ml_pipeline.predict([text])[0]
        prediction_text = "Fiction" if pred_label == 1 else "Non-Fiction"

        # predict_proba returns array of probabilities [[prob_0, prob_1]]
        probabilities = cls.ml_pipeline.predict_proba([text])[0]
        confidence = max(probabilities) * 100

        return {
            "prediction": prediction_text,
            "confidence": round(confidence, 2)
        }
