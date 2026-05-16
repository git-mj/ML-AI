import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from nlp_service import WordSimilarityModel

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
# Secure secret key fetching (will crash if missing in production)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback_dev_key")

# Initialize the model once
nlp_model = WordSimilarityModel()

@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "sent1": "I went to the bank to deposit money.",
        "sent2": "I sat on the river bank.",
        "word": "bank",
        "result": None,
        "error": None
    }

    if request.method == "POST":
        context["sent1"] = request.form.get("sent1", "").strip()
        context["sent2"] = request.form.get("sent2", "").strip()
        context["word"] = request.form.get("word", "").strip()
        
        try:
            context["result"] = nlp_model.calculate_similarity(
                context["sent1"], 
                context["sent2"], 
                context["word"]
            )
        except ValueError as ve:
            # Handle user errors (e.g., word not found)
            context["error"] = str(ve)
        except Exception as e:
            # Log the real error to console, give user a safe generic message
            app.logger.error(f"Inference error: {e}")
            context["error"] = "An internal error occurred while processing the text."

    return render_template("index.html", **context)

if __name__ == "__main__":
    # Use environment variables to toggle debug mode securely
    debug_mode = os.environ.get("FLASK_DEBUG", "True").lower() == "true"
    app.run(host="127.0.0.1", port=5000, debug=debug_mode)