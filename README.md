# Multi-Document AI Assistant

A powerful AI assistant capable of answering questions from multiple PDF manuals with source citations using LangChain, FAISS, and Google Gemini 2.0 Flash Lite.

## Features

- **Multi-PDF Processing**: Upload and process multiple PDF documents
- **Intelligent Text Chunking**: Smart text segmentation for optimal context retrieval
- **FAISS Vector Store**: Fast and efficient similarity search
- **Source Citations**: Every answer includes references to source documents
- **Streamlit Web Interface**: User-friendly web application
- **Google Gemini 2.0 Flash Lite Integration**: Advanced language model for accurate responses

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Upload your PDF documents in the web interface

3. Ask questions about the uploaded documents

4. Get answers with source citations

## Project Structure

- `app.py`: Main Streamlit application
- `pdf_processor.py`: PDF processing and text extraction
- `vector_store.py`: FAISS vector store management
- `qa_system.py`: Question answering system with LangChain
- `utils.py`: Utility functions
- `requirements.txt`: Python dependencies

## How It Works

1. **Document Processing**: PDFs are parsed and text is extracted
2. **Text Chunking**: Text is split into manageable chunks with overlap
3. **Embedding Generation**: Chunks are converted to vector embeddings
4. **Vector Storage**: Embeddings are stored in FAISS for fast retrieval
5. **Question Answering**: User questions are processed through the retrieval-augmented generation pipeline
6. **Source Citation**: Relevant source documents are cited in responses

## Requirements

- Python 3.8+
- Google API key
- Internet connection for API calls
