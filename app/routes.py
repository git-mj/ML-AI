from flask import Blueprint, request, jsonify, render_template
from .services import CountService
from .utils import sanitize_text, api_error_response, api_success_response
from posvisualizer.app.services import POSService
from keywordextractor.app.services import KeywordExtractorService
from intentdetector.app.services import IntentDetectorService
from admissionchatbot.app.services import AdmissionChatbotService
from fictometer.app.services import FictometerService
from spamclassifier.app.services import SpamClassifierService
from categorypredictor.app.services import CategoryPredictorService
from wordsimilarity.app.services import WordSimilarityService

main_bp = Blueprint('main', __name__)

# --- FRONTEND ROUTES ---
@main_bp.route('/')
def index():
    """Serves the main frontend HTML page."""
    return render_template('index.html')

@main_bp.route('/wordcounter')
def wordcounter():
    return render_template('wordcounter.html')

@main_bp.route('/admissionchatbot')
def admissionchatbot():
    return render_template('admissionchatbot.html')

@main_bp.route('/fictometer')
def fictometer():
    return render_template('fictometer.html')

@main_bp.route('/intentdetector')
def intentdetector():
    return render_template('intentdetector.html')

@main_bp.route('/keywordextractor')
def keywordextractor():
    return render_template('keywordextractor.html')

@main_bp.route('/posvisualizer')
def posvisualizer():
    return render_template('posvisualizer.html')

@main_bp.route('/spamclassifier')
def spamclassifier():
    return render_template('spamclassifier.html')

@main_bp.route('/wordsimilarity')
def wordsimilarity():
    return render_template('wordsimilarity.html')

@main_bp.route('/categorypredictor')
def categorypredictor():
    return render_template('categorypredictor.html')
# -----------------------

@main_bp.route('/api/v1/category_predict', methods=['POST'])
def category_predict():
    if not request.is_json:
        return api_error_response("Content-Type must be application/json", 415)

    data = request.get_json()
    raw_text  = data.get('text')
    embedding = data.get('embedding', 'word2vec')  # 'word2vec' or 'elmo'

    if not raw_text or not isinstance(raw_text, str):
        return api_error_response("A valid 'text' string is required.", 400)

    if embedding not in ('word2vec', 'elmo'):
        return api_error_response("embedding must be 'word2vec' or 'elmo'.", 400)

    clean_text = sanitize_text(raw_text)

    try:
        results = CategoryPredictorService.predict(clean_text, embedding)
        return api_success_response(results)
    except Exception as e:
        print(f"Internal error during Category Prediction: {e}")
        return api_error_response("An internal error occurred", 500)

@main_bp.route('/api/v1/word_similarity', methods=['POST'])
def word_similarity():
    if not request.is_json:
        return api_error_response("Content-Type must be application/json", 415)

    data = request.get_json()
    sentence1 = data.get('sentence1', '').strip()
    sentence2 = data.get('sentence2', '').strip()
    word      = data.get('word', '').strip()

    if not sentence1 or not sentence2 or not word:
        return api_error_response("'sentence1', 'sentence2', and 'word' are all required.", 400)

    try:
        result = WordSimilarityService.calculate_similarity(sentence1, sentence2, word)
        if result.get("error"):
            return api_error_response(result["error"], 422)
        return api_success_response(result)
    except Exception as e:
        print(f"Internal error during Word Similarity: {e}")
        return api_error_response("An internal error occurred", 500)

@main_bp.route('/api/v1/spamclassifier', methods=['POST'])
def classify_spam():
    if not request.is_json:
        return api_error_response("Content-Type must be application/json", 415)
    
    data = request.get_json()
    raw_text = data.get('text')
    algorithm = data.get('algorithm', 'lr')  # default to Logistic Regression

    if not raw_text or not isinstance(raw_text, str):
        return api_error_response("A valid 'text' string is required.", 400)

    if algorithm not in ('lr', 'nb'):
        return api_error_response("Algorithm must be 'lr' or 'nb'.", 400)

    clean_text = sanitize_text(raw_text)

    try:
        results = SpamClassifierService.classify(clean_text, algorithm)
        return api_success_response(results)
    except Exception as e:
        print(f"Internal error during Spam Classification: {e}")
        return api_error_response("An internal error occurred", 500)

@main_bp.route('/api/v1/extract_keywords', methods=['POST'])
def extract_keywords():
    if not request.is_json:
        return api_error_response("Content-Type must be application/json", 415)
    
    data = request.get_json()
    raw_text = data.get('text')

    if not raw_text or not isinstance(raw_text, str):
        return api_error_response("A valid 'text' string is required.", 400)

    clean_text = sanitize_text(raw_text)

    try:
        results = KeywordExtractorService.extract_keywords(clean_text)
        return api_success_response({"keywords": results})
    except Exception as e:
        print(f"Internal error during Keyword Extraction: {e}")
        return api_error_response("An internal error occurred during extraction", 500)

@main_bp.route('/api/v1/analyze_pos', methods=['POST'])
def analyze_pos():
    if not request.is_json:
        return api_error_response("Content-Type must be application/json", 415)
    
    data = request.get_json()
    raw_text = data.get('text')

    if not raw_text or not isinstance(raw_text, str):
        return api_error_response("A valid 'text' string is required.", 400)

    clean_text = sanitize_text(raw_text)

    try:
        results = POSService.analyze_text(clean_text)
        return api_success_response({"tagged_words": results})
    except Exception as e:
        print(f"Internal error during POS analysis: {e}")
        return api_error_response("An internal error occurred during analysis", 500)

@main_bp.route('/api/v1/detect_intent', methods=['POST'])
def detect_intent():
    if not request.is_json:
        return api_error_response("Content-Type must be application/json", 415)
    
    data = request.get_json()
    raw_text = data.get('text')

    if not raw_text or not isinstance(raw_text, str):
        return api_error_response("A valid 'text' string is required.", 400)

    clean_text = sanitize_text(raw_text)

    try:
        results = IntentDetectorService.detect_intent(clean_text)
        return api_success_response({"intent_data": results})
    except Exception as e:
        print(f"Internal error during Intent Detection: {e}")
        return api_error_response("An internal error occurred during detection", 500)

@main_bp.route('/api/v1/chat_admission', methods=['POST'])
def chat_admission():
    if not request.is_json:
        return api_error_response("Content-Type must be application/json", 415)
    
    data = request.get_json()
    raw_text = data.get('text')

    if not raw_text or not isinstance(raw_text, str):
        return api_error_response("A valid 'text' string is required.", 400)

    clean_text = sanitize_text(raw_text)

    try:
        results = AdmissionChatbotService.get_answer(clean_text)
        return api_success_response(results)
    except Exception as e:
        print(f"Internal error during Admission Chat: {e}")
        return api_error_response("An internal error occurred", 500)

@main_bp.route('/api/v1/fictometer', methods=['POST'])
def analyze_fictometer():
    if not request.is_json:
        return api_error_response("Content-Type must be application/json", 415)
    
    data = request.get_json()
    raw_text = data.get('text')

    if not raw_text or not isinstance(raw_text, str):
        return api_error_response("A valid 'text' string is required.", 400)

    clean_text = sanitize_text(raw_text)

    try:
        results = FictometerService.analyze_text(clean_text)
        return api_success_response(results)
    except Exception as e:
        print(f"Internal error during Fictometer analysis: {e}")
        return api_error_response("An internal error occurred", 500)

@main_bp.route('/api/v1/counts', methods=['POST'])
def counts():
    if not request.is_json:
        return api_error_response("Content-Type must be application/json", 415)
    
    data = request.get_json()
    raw_text = data.get('text')

    if not raw_text or not isinstance(raw_text, str):
        return api_error_response("A valid 'text' string is required.", 400)

    # 1. Clean the text using our utility function
    clean_text = sanitize_text(raw_text)

    if len(clean_text) > 50000:
        return api_error_response("Text payload exceeds the maximum allowed length.", 413)

    try:
        # 2. Process the text
        results = CountService.count_text(clean_text)
        
        # 3. Return the standardized success format
        return api_success_response(results)
        
    except Exception as e:
        # In a real app, use the 'logging' module here instead of print
        print(f"Internal error during analysis: {e}") 
        return api_error_response("An internal error occurred during analysis", 500)