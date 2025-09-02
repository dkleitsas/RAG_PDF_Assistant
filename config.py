import os
from dotenv import load_dotenv

try:
    import streamlit as st
    _has_streamlit = True
except ImportError:
    _has_streamlit = False

load_dotenv()

def get_config_value(key, default=None):
    if _has_streamlit and hasattr(st, 'secrets') and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

GOOGLE_API_KEY = get_config_value("GOOGLE_API_KEY")
GEMINI_MODEL = get_config_value("GEMINI_MODEL", "gemini-2.0-flash-lite")
GEMINI_TEMPERATURE = float(get_config_value("GEMINI_TEMPERATURE", "0.1"))

VECTOR_STORE_DIR = get_config_value("VECTOR_STORE_DIR", "vector_store")
EMBEDDINGS_MODEL = get_config_value("EMBEDDINGS_MODEL", "models/text-embedding-004")

CHUNK_SIZE = int(get_config_value("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(get_config_value("CHUNK_OVERLAP", "200"))

MAX_FILE_SIZE = 50 * 1024 * 1024
ALLOWED_FILE_TYPES = ['.pdf']

LOG_LEVEL = get_config_value("LOG_LEVEL", "INFO")
LOG_FILE = get_config_value("LOG_FILE", "app.log")

def validate_config():
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is required. Please set it in your environment variables or .env file.")
    
    return True
