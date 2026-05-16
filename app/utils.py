import re
from flask import jsonify
from typing import Tuple

def sanitize_text(text: str) -> str:
    """
    Cleans and prepares the input text before passing it to the ML model or NLTK.
    Helps prevent injection attacks and standardizes the input.
    """
    if not text:
        return ""
    
    # Security: Strip basic HTML tags to prevent XSS if you ever render this back to the user
    text_no_html = re.sub(r'<[^>]+>', '', text)
    
    # Normalization: Replace multiple spaces, tabs, or newlines with a single space
    clean_text = re.sub(r'\s+', ' ', text_no_html)
    
    # You can add more domain-specific cleaning here (e.g., removing URLs)
    
    return clean_text.strip()

def api_error_response(message: str, status_code: int = 400) -> Tuple[str, int]:
    """
    Ensures all error responses from the API share the exact same JSON structure.
    """
    response = jsonify({
        "status": "error",
        "error": message
    })
    return response, status_code

def api_success_response(data: dict, status_code: int = 200) -> Tuple[str, int]:
    """
    Ensures all successful responses share the exact same JSON structure.
    """
    response = jsonify({
        "status": "success",
        "data": data
    })
    return response, status_code