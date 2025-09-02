import os
import re
from typing import List, Dict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from vector_store import VectorStore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QASystem:
    
    def __init__(self, model_name: str = "gemini-2.0-flash-lite", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.vector_store = VectorStore()
        self.qa_chain = None
    
    def preprocess_query(self, query: str) -> str:
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        original_query = query
        query = query.lower().strip()
        query = re.sub(r'[^\w\s]', ' ', query)
        query = ' '.join(query.split())
        
        words = query.split()
        filtered_words = [word for word in words if word not in stop_words or len(word) > 3]
        
        processed_query = ' '.join(filtered_words)
        
        if len(processed_query) < len(original_query) * 0.3:
            processed_query = original_query.lower().strip()
        
        logger.info(f"Query preprocessing: '{original_query}' -> '{processed_query}'")
        return processed_query
    
    def get_optimal_retrieval_count(self, query: str) -> int:
        word_count = len(query.split())
        
        if word_count <= 3:
            return 3
        elif word_count <= 8:
            return 5
        else:
            return 7
    
    def _setup_qa_chain(self, retrieval_count: int = 5):
        
        if self.vector_store.vector_store is None:
            logger.warning("Vector store is empty. Cannot setup QA chain.")
            return False
        
        prompt_template = """You are a helpful AI assistant that answers questions based on the provided context from PDF documents. 

Context information:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question based on the provided documents."
3. DO NOT include any citations, source references, document names, or page numbers in your answer
4. Provide a clean, direct answer without mentioning where the information came from
5. Be accurate and concise in your responses
6. If you're unsure about something, acknowledge the uncertainty
7. Focus on the most relevant information from the context
8. If multiple sources provide conflicting information, mention this but do not cite the specific sources

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.vector_store.as_retriever(search_kwargs={"k": retrieval_count}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        logger.info(f"QA chain setup successfully with retrieval count: {retrieval_count}")
        return True
    
    def load_documents(self) -> bool:
        if self.vector_store.load():
            return self._setup_qa_chain()
        return False
    
    def add_documents(self, chunks: List[Dict]) -> None:
        self.vector_store.add_documents(chunks)
        if not self._setup_qa_chain():
            logger.error("Failed to setup QA chain after adding documents")
    
    def filter_relevant_documents(self, documents: List[Document], query: str, similarity_threshold: float = 0.6) -> List[Document]:
        if not documents:
            return []
        
        scored_docs = self.vector_store.similarity_search_with_score(query, k=len(documents)*2)
        relevant_docs = []
        for doc, score in scored_docs:
            similarity = 1.0 / (1.0 + score) if score > 0 else 1.0
            
            if similarity >= similarity_threshold:
                relevant_docs.append(doc)
                logger.info(f"Document accepted: {similarity:.3f} - {doc.metadata.get('filename', 'Unknown')}")
            else:
                logger.info(f"Document filtered: {similarity:.3f} - {doc.metadata.get('filename', 'Unknown')}")
        
        if not relevant_docs and documents:
            logger.warning(f"No documents met similarity threshold {similarity_threshold}, using top 3")
            return documents[:3]
        
        return relevant_docs[:len(documents)]

    def ask_question(self, question: str, similarity_threshold: float = 0.6) -> Dict:
        if self.qa_chain is None:
            return {
                "answer": "No documents have been loaded. Please upload some PDF documents first.",
                "sources": [],
                "error": "No QA chain available"
            }
        
        processed_query = self.preprocess_query(question)
        
        retrieval_count = self.get_optimal_retrieval_count(question)
        
        self._setup_qa_chain(retrieval_count)
        
        result = self.qa_chain({"query": processed_query})
        
        answer = result.get("result", "No answer generated")
        source_documents = result.get("source_documents", [])
        
        relevant_documents = self.filter_relevant_documents(
            source_documents, 
            processed_query, 
            similarity_threshold
        )
        
        processed_sources = self._process_source_documents(relevant_documents, question, answer)
        
        logger.info(f"Query: '{question}' -> Retrieved: {len(source_documents)}, Relevant: {len(relevant_documents)}")
        
        return {
            "answer": answer,
            "sources": processed_sources,
            "error": None,
            "retrieval_count": retrieval_count,
            "relevant_count": len(relevant_documents),
            "processed_query": processed_query
        }
    
    def _extract_relevant_sentences(self, content: str, query: str, answer: str, max_sentences: int = 3) -> str:

        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        if not sentences:
            return content[:200] + "..." if len(content) > 200 else content
        

        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        answer_words = set(re.findall(r'\b\w{3,}\b', answer.lower()))
        important_words = query_words.union(answer_words)
        

        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(re.findall(r'\b\w{3,}\b', sentence.lower()))
            score = len(sentence_words.intersection(important_words))
            if score > 0:
                scored_sentences.append((sentence, score))
        

        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:max_sentences]]
        
        if not top_sentences:
            
            return '. '.join(sentences[:2]) + '.' if len(sentences) >= 2 else sentences[0] + '.'
        
        return '. '.join(top_sentences) + '.'
    
    def _calculate_citation_relevance(self, doc: Document, query: str, answer: str) -> float:
        content = doc.page_content.lower()
        query_lower = query.lower()
        answer_lower = answer.lower()
        

        query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
        answer_words = set(re.findall(r'\b\w{3,}\b', answer_lower))
        content_words = set(re.findall(r'\b\w{3,}\b', content))
        

        query_overlap = len(query_words.intersection(content_words)) / max(len(query_words), 1)
        answer_overlap = len(answer_words.intersection(content_words)) / max(len(answer_words), 1)
        

        relevance_score = (query_overlap * 0.3) + (answer_overlap * 0.7)
        

        query_phrases = [phrase.strip() for phrase in query_lower.split() if len(phrase.strip()) > 3]
        phrase_bonus = sum(1 for phrase in query_phrases if phrase in content) * 0.1
        
        return min(relevance_score + phrase_bonus, 1.0)
    
    def _process_source_documents(self, source_documents: List[Document], query: str = "", answer: str = "") -> List[Dict]:
        processed_sources = []
        
        for doc in source_documents:
            metadata = doc.metadata
            content = doc.page_content
            

            page_number = metadata.get('page', None)
            

            relevance_score = self._calculate_citation_relevance(doc, query, answer) if query and answer else 0.5
            

            content_preview = content.strip()
            
            source_info = {
                "filename": metadata.get("filename", "Unknown"),
                "page": page_number,
                "content_preview": content_preview,
                "chunk_id": metadata.get("chunk_id", "Unknown"),
                "relevance_score": relevance_score
            }
            
            processed_sources.append(source_info)
        

        processed_sources.sort(key=lambda x: x['relevance_score'], reverse=True)
        

        relevant_sources = [s for s in processed_sources if s['relevance_score'] > 0.3]
        

        return relevant_sources[:5]
    

    def get_vector_store_stats(self) -> Dict:
        return self.vector_store.get_stats()
    
    def clear_documents(self) -> None:
        self.vector_store.clear()
        self.qa_chain = None
        logger.info("All documents cleared from the system")
    
    def format_answer_with_sources(self, answer: str, sources: List[Dict]) -> str:
        if not sources:
            return answer
        

        formatted_answer = answer + "\n\n**Sources:**\n"
        
        for i, source in enumerate(sources, 1):
            filename = source.get("filename", "Unknown")
            page = source.get("page", "Unknown page")
            
            citation = f"{i}. {filename}"
            if page and page != "Unknown page":
                citation += f" (Page {page})"
            
            formatted_answer += citation + "\n"
        
        return formatted_answer
