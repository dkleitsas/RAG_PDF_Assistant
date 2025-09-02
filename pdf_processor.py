import PyPDF2
import os
from typing import List, Dict, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
        )
    
    def extract_text_from_pdf_by_pages(self, pdf_path: str) -> List[Dict]:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            pages_data = []
            base_metadata = {
                'filename': os.path.basename(pdf_path),
                'num_pages': len(pdf_reader.pages),
                'file_path': pdf_path
            }
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    page_metadata = base_metadata.copy()
                    page_metadata['page'] = page_num + 1
                    
                    pages_data.append({
                        'text': page_text.strip(),
                        'metadata': page_metadata
                    })
            
            logger.info(f"Extracted {len(pages_data)} pages from {pdf_path}")
            return pages_data
    
    def chunk_page_text(self, page_text: str, page_metadata: Dict) -> List[Dict]:
        chunks = self.text_splitter.split_text(page_text)
        
        chunked_documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = page_metadata.copy()
            chunk_metadata['chunk_id'] = i
            chunk_metadata['chunk_size'] = len(chunk)
            
            chunked_documents.append({
                'text': chunk,
                'metadata': chunk_metadata
            })
        
        return chunked_documents
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        pages_data = self.extract_text_from_pdf_by_pages(pdf_path)
        
        all_chunks = []
        for page_data in pages_data:
            page_chunks = self.chunk_page_text(page_data['text'], page_data['metadata'])
            all_chunks.extend(page_chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(pages_data)} pages in {pdf_path}")
        return all_chunks
    
    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict]:
        all_chunks = []
        
        for pdf_path in pdf_paths:
            chunks = self.process_pdf(pdf_path)
            all_chunks.extend(chunks)
            logger.info(f"Processed {pdf_path}: {len(chunks)} chunks")
        
        logger.info(f"Total chunks processed: {len(all_chunks)}")
        return all_chunks
