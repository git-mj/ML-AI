from flask import Blueprint, request, render_template
from .services import CountService
from .utils import (
    MAX_SENTENCE_CHARS,
    MAX_TEXT_CHARS,
    api_error_response,
    api_success_response,
    get_json_body,
    require_text_field,
    sanitize_text,
)
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
    data, error = get_json_body(request)
    if error:
        return error

    raw_text, error = require_text_field(data)
    if error:
        return error

    embedding = data.get('embedding', 'word2vec')  # 'word2vec' or 'elmo'

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
    data, error = get_json_body(request)
    if error:
        return error

    sentence1, error = require_text_field(data, 'sentence1', MAX_SENTENCE_CHARS)
    if error:
        return error
    sentence2, error = require_text_field(data, 'sentence2', MAX_SENTENCE_CHARS)
    if error:
        return error
    word, error = require_text_field(data, 'word', 100)
    if error:
        return error

    if any(char.isspace() for char in word):
        return api_error_response("'word' must be a single token.", 400)

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
    data, error = get_json_body(request)
    if error:
        return error

    raw_text, error = require_text_field(data)
    if error:
        return error

    algorithm = data.get('algorithm', 'lr')  # default to Logistic Regression

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
    data, error = get_json_body(request)
    if error:
        return error

    raw_text, error = require_text_field(data)
    if error:
        return error

    clean_text = sanitize_text(raw_text)

    try:
        results = KeywordExtractorService.extract_keywords(clean_text)
        return api_success_response({"keywords": results})
    except Exception as e:
        print(f"Internal error during Keyword Extraction: {e}")
        return api_error_response("An internal error occurred during extraction", 500)

@main_bp.route('/api/v1/analyze_pos', methods=['POST'])
def analyze_pos():
    data, error = get_json_body(request)
    if error:
        return error

    raw_text, error = require_text_field(data)
    if error:
        return error

    clean_text = sanitize_text(raw_text)

    try:
        results = POSService.analyze_text(clean_text)
        return api_success_response({"tagged_words": results})
    except Exception as e:
        print(f"Internal error during POS analysis: {e}")
        return api_error_response("An internal error occurred during analysis", 500)

@main_bp.route('/api/v1/detect_intent', methods=['POST'])
def detect_intent():
    data, error = get_json_body(request)
    if error:
        return error

    raw_text, error = require_text_field(data)
    if error:
        return error

    clean_text = sanitize_text(raw_text)

    try:
        results = IntentDetectorService.detect_intent(clean_text)
        return api_success_response({"intent_data": results})
    except Exception as e:
        print(f"Internal error during Intent Detection: {e}")
        return api_error_response("An internal error occurred during detection", 500)

@main_bp.route('/api/v1/chat_admission', methods=['POST'])
def chat_admission():
    data, error = get_json_body(request)
    if error:
        return error

    raw_text, error = require_text_field(data)
    if error:
        return error

    clean_text = sanitize_text(raw_text)

    try:
        results = AdmissionChatbotService.get_answer(clean_text)
        return api_success_response(results)
    except Exception as e:
        print(f"Internal error during Admission Chat: {e}")
        return api_error_response("An internal error occurred", 500)

@main_bp.route('/api/v1/fictometer', methods=['POST'])
def analyze_fictometer():
    data, error = get_json_body(request)
    if error:
        return error

    raw_text, error = require_text_field(data)
    if error:
        return error

    clean_text = sanitize_text(raw_text)

    try:
        results = FictometerService.analyze_text(clean_text)
        return api_success_response(results)
    except Exception as e:
        print(f"Internal error during Fictometer analysis: {e}")
        return api_error_response("An internal error occurred", 500)

@main_bp.route('/api/v1/counts', methods=['POST'])
def counts():
    data, error = get_json_body(request)
    if error:
        return error

    raw_text, error = require_text_field(data, max_length=MAX_TEXT_CHARS)
    if error:
        return error

    # 1. Clean the text using our utility function
    clean_text = sanitize_text(raw_text)

    try:
        # 2. Process the text
        results = CountService.count_text(clean_text)
        
        # 3. Return the standardized success format
        return api_success_response(results)
        
    except Exception as e:
        # In a real app, use the 'logging' module here instead of print
        print(f"Internal error during analysis: {e}") 
        return api_error_response("An internal error occurred during analysis", 500)
