import pytest
from flask import Flask

from app.utils import MAX_TEXT_CHARS, require_text_field, sanitize_text


@pytest.fixture(autouse=True)
def app_context():
    app = Flask(__name__)
    with app.app_context():
        yield


def test_sanitize_text_preserves_markup_as_text():
    assert sanitize_text("<b>Hello</b>\nworld") == "<b>Hello</b> world"


def test_require_text_field_rejects_missing_text():
    value, error = require_text_field({})

    assert value is None
    assert error[1] == 400


def test_require_text_field_rejects_oversized_text():
    value, error = require_text_field({"text": "x" * (MAX_TEXT_CHARS + 1)})

    assert value is None
    assert error[1] == 413


def test_require_text_field_accepts_valid_text():
    value, error = require_text_field({"text": " hello "})

    assert value == "hello"
    assert error is None
