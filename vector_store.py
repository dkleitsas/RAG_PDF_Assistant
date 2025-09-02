import os
import pickle
import numpy as np
from typing import List, Dict, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    
    def __init__(self, embeddings_model: str = "models/text-embedding-004", persist_directory: str = "vector_store"):
        self.embeddings_model = embeddings_model
        self.persist_directory = persist_directory
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)
        self.vector_store = None
        self.unique_files = set()
        
        os.makedirs(persist_directory, exist_ok=True)
    
    def create_documents_from_chunks(self, chunks: List[Dict]) -> List[Document]:
        documents = []
        
        for chunk in chunks:
            doc = Document(
                page_content=chunk['text'],
                metadata=chunk['metadata']
            )
            documents.append(doc)
        
        return documents
    
    def add_documents(self, chunks: List[Dict]) -> None:
        documents = self.create_documents_from_chunks(chunks)

        for chunk in chunks:
            filename = chunk['metadata'].get('filename')
            if filename:
                self.unique_files.add(filename)
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings,
            )
            logger.info(f"Created new vector store with {len(documents)} documents")
        else:
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to existing vector store")
        
        self.save()
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if self.vector_store is None:
            logger.warning("Vector store is empty. Load or add documents first.")
            return []
        
        results = self.vector_store.similarity_search(query, k=k)
        logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        if self.vector_store is None:
            logger.warning("Vector store is empty. Load or add documents first.")
            return []
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        logger.info(f"Found {len(results)} similar documents with scores for query: {query[:50]}...")
        return results
    
    def save(self) -> None:
        if self.vector_store is not None:
            self.vector_store.save_local(self.persist_directory)
            files_path = os.path.join(self.persist_directory, "unique_files.pkl")
            with open(files_path, 'wb') as f:
                pickle.dump(self.unique_files, f)
            logger.info(f"Vector store saved to {self.persist_directory}")
    
    def load(self) -> bool:
        try:
            if os.path.exists(self.persist_directory):
                self.vector_store = FAISS.load_local(
                    folder_path=self.persist_directory,
                    embeddings=self.embeddings
                )
                files_path = os.path.join(self.persist_directory, "unique_files.pkl")
                if os.path.exists(files_path):
                    with open(files_path, 'rb') as f:
                        self.unique_files = pickle.load(f)
                else:
                    self._rebuild_unique_files()
                logger.info(f"Vector store loaded from {self.persist_directory}")
                return True
            else:
                logger.info("No existing vector store found")
                return False
                
        except:
            return False
    
    def get_stats(self) -> Dict:
        if self.vector_store is None:
            return {"total_documents": 0, "index_size": 0}
        
        return {
            "total_documents": len(self.unique_files),
            "index_size": self.vector_store.index.ntotal if hasattr(self.vector_store.index, 'ntotal') else 0
        }
    
    def clear(self) -> None:
        self.vector_store = None
        self.unique_files = set()
        logger.info("Vector store cleared")
    
    def _rebuild_unique_files(self) -> None:
        if self.vector_store is not None:
            for doc_id, doc in self.vector_store.docstore._dict.items():
                filename = doc.metadata.get('filename')
                if filename:
                    self.unique_files.add(filename)
            logger.info(f"Rebuilt unique files set with {len(self.unique_files)} files")
    
    def delete_persisted_store(self) -> None:
        import shutil
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            logger.info(f"Deleted persisted vector store: {self.persist_directory}")
