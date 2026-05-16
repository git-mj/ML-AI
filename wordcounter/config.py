import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default-hard-to-guess-string')
    # Security: Limit max content length to 1MB
    MAX_CONTENT_LENGTH = 1 * 1024 * 1024 
    JSON_SORT_KEYS = False