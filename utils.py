import os
import tempfile
from typing import List, Optional
import streamlit as st
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def validate_pdf_file(file) -> bool:
    if file is None:
        return False
    

    if not file.name.lower().endswith('.pdf'):
        return False
    

    if file.size > 50 * 1024 * 1024:
        return False
    
    return True

def save_uploaded_files(uploaded_files: List) -> List[str]:
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                saved_paths.append(tmp_file.name)
                logger.info(f"Saved uploaded file: {uploaded_file.name} -> {tmp_file.name}")
    
    return saved_paths

def cleanup_temp_files(file_paths: List[str]) -> None:
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")

def format_file_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def get_file_info(file_path: str) -> dict:
    stat = os.stat(file_path)
    return {
        "name": os.path.basename(file_path),
        "size": format_file_size(stat.st_size),
        "size_bytes": stat.st_size,
        "modified": stat.st_mtime
    }

def create_download_link(file_path: str, filename: str) -> str:
    with open(file_path, "rb") as f:
        data = f.read()
    
    import base64
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download {filename}</a>'

def validate_api_key() -> bool:
    api_key = os.getenv("GOOGLE_API_KEY")
    return api_key is not None and api_key.strip() != ""

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )

def create_directory_if_not_exists(directory: str) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)

def get_system_info() -> dict:
    import platform
    import sys
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine()
    }
