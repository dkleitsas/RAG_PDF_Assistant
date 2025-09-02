# Multi-Document AI Assistant

A powerful AI assistant capable of answering questions from multiple PDF manuals with source citations using LangChain, FAISS, and Google Gemini 2.0 Flash Lite. Give it a try [here!](https://pdf-assistant-rag.streamlit.app/) (Might take a second to load.)

Note: Still a WIP, the retrieval component could frankly be a lot better. Also bear in mind a lower end LLM model is used due to cost constraints so please verify any answer you get from this.
 
## Features

- **Multi-PDF Processing**: Upload and process multiple PDF documents
- **Intelligent Text Chunking**: Smart text segmentation for optimal context retrieval
- **FAISS Vector Store**: Fast and efficient similarity search
- **Source Citations**: Every answer includes references to source documents
- **Streamlit Web Interface**: User-friendly web application
- **Google Gemini 2.0 Flash Lite Integration**: Advanced language model for accurate responses


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
