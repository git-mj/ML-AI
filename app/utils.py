from flask import jsonify
from typing import Any, Tuple

MAX_TEXT_CHARS = 50000
MAX_SENTENCE_CHARS = 5000

def sanitize_text(text: str) -> str:
    """
    Normalizes whitespace before passing text to NLP services.

    Security should be handled by validating input sizes and escaping output at
    render time. Stripping HTML here would alter legitimate text analysis input.
    """
    if not text:
        return ""

    return " ".join(text.split()).strip()

def get_json_body(request_obj) -> Tuple[dict[str, Any] | None, Tuple[Any, int] | None]:
    if not request_obj.is_json:
        return None, api_error_response("Content-Type must be application/json", 415)

    data = request_obj.get_json(silent=True)
    if not isinstance(data, dict):
        return None, api_error_response("A valid JSON object is required.", 400)

    return data, None

def require_text_field(data: dict[str, Any], field: str = "text", max_length: int = MAX_TEXT_CHARS) -> Tuple[str | None, Tuple[Any, int] | None]:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        return None, api_error_response(f"A valid '{field}' string is required.", 400)

    value = value.strip()
    if len(value) > max_length:
        return None, api_error_response(
            f"'{field}' exceeds the maximum allowed length of {max_length} characters.",
            413,
        )

    return value, None

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
