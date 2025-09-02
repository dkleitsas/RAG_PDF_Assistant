import streamlit as st
import os
import tempfile
from typing import List, Dict
import logging
from dotenv import load_dotenv

from pdf_processor import PDFProcessor
from qa_system import QASystem
from utils import (
    validate_pdf_file, save_uploaded_files, cleanup_temp_files,
    format_file_size, validate_api_key, setup_logging
)

load_dotenv()

setup_logging()
st.set_page_config(
    page_title="Multi-Document AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #e8eef5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        border: 1px solid #d1d9e0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .source-box {
        background-color: #e9ecef;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []

def initialize_qa_system():
    if st.session_state.qa_system is None:
        st.session_state.qa_system = QASystem()
        if st.session_state.qa_system.load_documents():
            st.session_state.documents_loaded = True
        else:
            st.session_state.documents_loaded = False
    return True

def process_uploaded_files(uploaded_files: List) -> bool:
    if not uploaded_files:
        return False
    
    valid_files = []
    for file in uploaded_files:
        if validate_pdf_file(file):
            valid_files.append(file)
        else:
            st.warning(f"Invalid file: {file.name}. Please upload PDF files only (max 50MB).")
    
    if not valid_files:
        return False
    
    with st.spinner("Processing PDF files..."):
        temp_file_paths = save_uploaded_files(valid_files)
        st.session_state.temp_files.extend(temp_file_paths)
        
        pdf_processor = PDFProcessor()
        all_chunks = pdf_processor.process_multiple_pdfs(temp_file_paths)
        
        if all_chunks:
            st.session_state.qa_system.add_documents(all_chunks)
            st.session_state.documents_loaded = True
            
            st.success(f"Successfully processed {len(valid_files)} PDF file(s)!")
            return True
        else:
            st.error("No text could be extracted from the uploaded PDFs.")
            return False

def display_source_citations(sources: List[Dict]):
    if not sources:
        return
    
    st.markdown("### 📚 Source Citations")
    st.markdown("*Citations are ranked by relevance to your question and the generated answer.*")
    
    for i, source in enumerate(sources, 1):
        filename = source.get("filename", "Unknown")
        page = source.get("page", "Unknown page")
        content_preview = source.get("content_preview", "")
        relevance_score = source.get("relevance_score", 0)
        
        relevance_percent = int(relevance_score * 100)
        relevance_emoji = "🟢" if relevance_percent >= 70 else "🟡" if relevance_percent >= 40 else "🔴"
        
        page_display = f"Page {page}" if page and page != "Unknown page" else "Page Unknown"
        
        with st.expander(f"{relevance_emoji} Source {i}: {filename} ({page_display}) - {relevance_percent}% relevant"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**File:** {filename}")
                st.markdown(f"**Page:** {page_display}")
            with col2:
                st.metric("Relevance", f"{relevance_percent}%")
            
            st.markdown("**Full Context:**")
            st.markdown(f"*{content_preview}*")

def main():
    
    st.markdown('<h1 class="main-header">📚 Multi-Document AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your PDF documents with AI-powered answers and source citations</p>', unsafe_allow_html=True)
    
    if not validate_api_key():
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ Google API Key Required</h3>
            <p>Please set your Google API key in the environment variables or create a <code>.env</code> file with:</p>
            <code>GOOGLE_API_KEY=your_api_key_here</code>
        </div>
        """, unsafe_allow_html=True)
        return
    
    if not initialize_qa_system():
        return
    
    with st.sidebar:
        st.header("📁 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files (max 50MB each)"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                if process_uploaded_files(uploaded_files):
                    st.rerun()
        
        if st.session_state.documents_loaded:
            st.markdown("---")
            st.subheader("📊 Document Statistics")
            
            stats = st.session_state.qa_system.get_vector_store_stats()
            st.metric("Total Documents", stats.get("total_documents", 0))
            st.metric("Index Size", stats.get("index_size", 0))
            
            if st.button("🗑️ Clear All Documents", type="secondary"):
                st.session_state.qa_system.clear_documents()
                st.session_state.documents_loaded = False
                cleanup_temp_files(st.session_state.temp_files)
                st.session_state.temp_files = []
                st.rerun()
    
    if not st.session_state.documents_loaded:
        st.markdown("""
        <div class="info-box">
            <h3> Getting Started</h3>
            <ol>
                <li>Upload your PDF documents using the sidebar</li>
                <li>Click "Process Documents" to extract and index the content</li>
                <li>Start asking questions about your documents!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Example Questions")
        st.markdown("""
        - "What are the main safety procedures mentioned in the manual?"
        - "What do I need to know about taking out a credit card with this bank?"
        - "What methodologies are used in this research paper?"
        - "Which of these candidates have relevant ML experience?"
        """)
        
    else:
        st.markdown("### Ask a Question")
        
        question = st.text_input(
            "Enter your question about the uploaded documents:",
            placeholder="e.g., What are the safety procedures mentioned in the manual?",
            key="question_input"
        )
        
        similarity_threshold = 0.6
        temperature = 0.1
        
        if st.button(" Ask Question", type="primary", disabled=not question.strip()):
            if question.strip():
                with st.spinner(" Generating answer..."):
                    result = st.session_state.qa_system.ask_question(question, similarity_threshold)
                    
                    if result.get("error"):
                        st.error(f"Error: {result['error']}")
                    else:
                        st.markdown("### Answer")
                        st.markdown(result["answer"])
                        
                        if result["sources"]:
                            display_source_citations(result["sources"])
                        else:
                            st.info("No highly relevant citations found for this answer (all sources below 30% relevance threshold).")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Built with LangChain, FAISS, and Google Gemini 2.0 Flash Lite</p>
        <p>Upload your PDFs and get AI-powered answers with source citations!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
