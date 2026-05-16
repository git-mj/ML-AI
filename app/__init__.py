from flask import Flask
from .services import initialize_nltk, CountService
from posvisualizer.app.services import initialize_nltk as init_pos_nltk, POSService
from keywordextractor.app.services import initialize_nltk as init_keyword_nltk, KeywordExtractorService
from intentdetector.app.services import IntentDetectorService
from admissionchatbot.app.services import AdmissionChatbotService
from fictometer.app.services import FictometerService
from spamclassifier.app.services import SpamClassifierService
from categorypredictor.app.services import CategoryPredictorService
from wordsimilarity.app.services import WordSimilarityService

def create_app():
    app = Flask(__name__)
    
    # Initialize NLP resources and Load ML Models into memory BEFORE accepting requests
    initialize_nltk()
    init_pos_nltk()
    init_keyword_nltk()
    CountService.load_model()
    POSService.load_model()
    KeywordExtractorService.load_model()
    IntentDetectorService.load_model()
    AdmissionChatbotService.load_model()
    FictometerService.load_model()
    SpamClassifierService.load_model()
    CategoryPredictorService.load_model()  # loads Word2Vec; ELMo loads lazily
    WordSimilarityService.load_model()     # ELMo loads lazily on first request

    from .routes import main_bp
    app.register_blueprint(main_bp)

    return app