from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional
import json
import logging
import time
from datetime import datetime
import os
import uuid
import traceback
import asyncio

# Document processing
import PyPDF2
import docx2txt
from email import message_from_string
import requests

# Vector embeddings and search
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# LLM integration
import openai
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    AUTH_TOKEN = os.getenv("AUTH_TOKEN", "a50fae96ae3b62a216ea8e170f437ac05867f1456dd8d81ca8bede22c1bcb161")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    VECTOR_DIM = 384
    MAX_FILE_SIZE = 10 * 1024 * 1024  # Reduced to 10MB for serverless
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.eml'}

config = Config()

# Pydantic models
class QueryRequest(BaseModel):
    documents: List[HttpUrl]
    questions: List[str]

class AnswerItem(BaseModel):
    question: str
    answer: str
    confidence: float
    source_clause: Optional[str] = None
    reasoning: str

class QueryResponse(BaseModel):
    answers: List[AnswerItem]
    processing_time: float
    total_documents_processed: int

class Document(BaseModel):
    id: str
    url: str
    content: str
    metadata: Dict[str, Any]

class Clause(BaseModel):
    id: str
    document_id: str
    text: str
    clause_type: str
    embeddings: List[float]
    metadata: Dict[str, Any]

# Global variables for caching (will persist during function warm-up)
embedding_model = None
anthropic_client = None

def get_embedding_model():
    """Get or initialize embedding model"""
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return embedding_model

def get_anthropic_client():
    """Get or initialize Anthropic client"""
    global anthropic_client
    if anthropic_client is None and config.ANTHROPIC_API_KEY:
        anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return anthropic_client

# Document processor
class DocumentProcessor:
    def __init__(self):
        pass
        
    async def download_document(self, url: str) -> bytes:
        """Download document from URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            if len(response.content) > config.MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="File too large")
            
            return response.content
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    
    def extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            import io
            pdf_file = io.BytesIO(content)
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""
    
    def extract_text_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX content"""
        try:
            import io
            docx_file = io.BytesIO(content)
            text = docx2txt.process(docx_file)
            return text
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return ""
    
    def extract_text_from_email(self, content: bytes) -> str:
        """Extract text from email content"""
        try:
            msg = message_from_string(content.decode('utf-8'))
            if msg.is_multipart():
                text = ""
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        text += part.get_payload(decode=True).decode('utf-8')
                return text
            else:
                return msg.get_payload(decode=True).decode('utf-8')
        except Exception as e:
            logger.error(f"Email extraction failed: {e}")
            return ""
    
    async def process_document(self, url: str) -> Document:
        """Process a single document"""
        try:
            content_bytes = await self.download_document(url)
            
            url_lower = url.lower()
            if url_lower.endswith('.pdf'):
                text = self.extract_text_from_pdf(content_bytes)
            elif url_lower.endswith('.docx'):
                text = self.extract_text_from_docx(content_bytes)
            elif url_lower.endswith('.eml'):
                text = self.extract_text_from_email(content_bytes)
            else:
                text = content_bytes.decode('utf-8', errors='ignore')
            
            if not text.strip():
                raise HTTPException(status_code=400, detail="No text extracted from document")
            
            doc_id = str(uuid.uuid4())
            
            metadata = {
                "url": url,
                "file_size": len(content_bytes),
                "extracted_length": len(text),
                "processed_at": datetime.utcnow().isoformat()
            }
            
            return Document(
                id=doc_id,
                url=url,
                content=text,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Document processing failed for {url}: {e}")
            raise HTTPException(status_code=500, detail=f"Document processing failed: {e}")
    
    def segment_into_clauses(self, document: Document) -> List[Clause]:
        """Segment document into semantic clauses"""
        clauses = []
        model = get_embedding_model()
        
        paragraphs = document.content.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) < 50:
                continue
                
            sentences = paragraph.split('. ')
            if len(sentences) > 3:
                for j in range(0, len(sentences), 3):
                    clause_text = '. '.join(sentences[j:j+3])
                    if clause_text.strip():
                        clause_id = str(uuid.uuid4())
                        clause_type = self._classify_clause(clause_text)
                        embeddings = model.encode([clause_text])[0].tolist()
                        
                        clauses.append(Clause(
                            id=clause_id,
                            document_id=document.id,
                            text=clause_text.strip(),
                            clause_type=clause_type,
                            embeddings=embeddings,
                            metadata={"paragraph_index": i, "sentence_group": j}
                        ))
            else:
                clause_id = str(uuid.uuid4())
                clause_type = self._classify_clause(paragraph)
                embeddings = model.encode([paragraph])[0].tolist()
                
                clauses.append(Clause(
                    id=clause_id,
                    document_id=document.id,
                    text=paragraph.strip(),
                    clause_type=clause_type,
                    embeddings=embeddings,
                    metadata={"paragraph_index": i}
                ))
        
        return clauses
    
    def _classify_clause(self, text: str) -> str:
        """Classify clause type based on content patterns"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['premium', 'deductible', 'coverage', 'claim', 'policy']):
            return 'insurance'
        if any(term in text_lower for term in ['shall', 'liable', 'agreement', 'contract', 'legal']):
            return 'legal'
        if any(term in text_lower for term in ['employee', 'employment', 'benefits', 'salary', 'vacation']):
            return 'hr'
        if any(term in text_lower for term in ['comply', 'regulation', 'requirement', 'audit', 'standard']):
            return 'compliance'
        if any(term in text_lower for term in ['medical', 'health', 'treatment', 'surgery', 'hospital']):
            return 'medical'
        
        return 'general'

# Vector search engine
class VectorSearch:
    def __init__(self):
        self.index = None
        self.clause_map = {}
        
    def build_index(self, clauses: List[Clause]):
        """Build FAISS index from clauses"""
        if not clauses:
            return
            
        embeddings = np.array([clause.embeddings for clause in clauses]).astype('float32')
        self.index = faiss.IndexFlatIP(config.VECTOR_DIM)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.clause_map = {i: clause for i, clause in enumerate(clauses)}
        
        logger.info(f"Built FAISS index with {len(clauses)} clauses")
    
    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        """Search for relevant clauses"""
        if self.index is None:
            return []
        
        model = get_embedding_model()
        query_embedding = model.encode([query])[0].astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in self.clause_map:
                results.append((self.clause_map[idx], float(score)))
        
        return results

# LLM integration
class LLMProcessor:
    def __init__(self):
        if config.OPENAI_API_KEY:
            openai.api_key = config.OPENAI_API_KEY
    
    async def generate_answer(self, question: str, relevant_clauses: List[tuple]) -> AnswerItem:
        """Generate answer using LLM"""
        try:
            context_parts = []
            
            for clause, similarity_score in relevant_clauses[:3]:
                context_parts.append(f"[Clause from {clause.clause_type}]: {clause.text}")
            
            context = "\n\n".join(context_parts)
            
            prompt = f"""Based on the following document clauses, please answer the question accurately and provide clear reasoning.

Context:
{context}

Question: {question}

Please provide:
1. A direct answer to the question
2. Your confidence level (0.0 to 1.0)
3. Clear reasoning for your answer
4. Reference to the specific clause if applicable

Format your response as JSON:
{{
    "answer": "your direct answer",
    "confidence": 0.0-1.0,
    "reasoning": "explanation of your reasoning",
    "source_clause": "most relevant clause text or null"
}}"""

            if config.OPENAI_API_KEY:
                response = await self._call_openai(prompt)
            elif config.ANTHROPIC_API_KEY:
                response = await self._call_anthropic(prompt)
            else:
                response = self._fallback_response(question, relevant_clauses)
            
            try:
                result = json.loads(response)
                return AnswerItem(
                    question=question,
                    answer=result.get("answer", "Unable to determine answer"),
                    confidence=float(result.get("confidence", 0.5)),
                    source_clause=result.get("source_clause"),
                    reasoning=result.get("reasoning", "Based on document analysis")
                )
            except json.JSONDecodeError:
                return AnswerItem(
                    question=question,
                    answer=response,
                    confidence=0.7,
                    source_clause=relevant_clauses[0][0].text if relevant_clauses else None,
                    reasoning="Direct LLM response"
                )
                
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            return AnswerItem(
                question=question,
                answer="Unable to process question due to system error",
                confidence=0.0,
                source_clause=None,
                reasoning=f"System error: {str(e)}"
            )
    
    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert document analyst specializing in insurance, legal, HR, and compliance documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API"""
        try:
            client = get_anthropic_client()
            if not client:
                raise Exception("Anthropic client not initialized")
                
            response = await client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise
    
    def _fallback_response(self, question: str, relevant_clauses: List[tuple]) -> str:
        """Fallback rule-based response when LLM is unavailable"""
        if not relevant_clauses:
            return json.dumps({
                "answer": "No relevant information found in the documents",
                "confidence": 0.1,
                "reasoning": "No matching clauses found",
                "source_clause": None
            })
        
        question_lower = question.lower()
        best_clause = relevant_clauses[0][0]
        
        if "cover" in question_lower and "surgery" in question_lower:
            if "surgery" in best_clause.text.lower():
                return json.dumps({
                    "answer": "Yes, surgery appears to be covered based on the policy terms",
                    "confidence": 0.8,
                    "reasoning": "Found surgery-related clause in policy",
                    "source_clause": best_clause.text
                })
        
        return json.dumps({
            "answer": f"Based on the document content: {best_clause.text[:200]}...",
            "confidence": 0.6,
            "reasoning": "Based on most relevant document clause",
            "source_clause": best_clause.text
        })

# Initialize components
doc_processor = DocumentProcessor()
vector_search = VectorSearch()
llm_processor = LLMProcessor()

# Create FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Main endpoint for processing queries"""
    start_time = time.time()
    
    try:
        # Verify auth token from headers
        # Note: In Vercel, you'll need to pass the token as a header
        
        logger.info(f"Processing query with {len(request.documents)} documents and {len(request.questions)} questions")
        
        documents = []
        all_clauses = []
        
        for doc_url in request.documents:
            try:
                doc = await doc_processor.process_document(str(doc_url))
                documents.append(doc)
                
                clauses = doc_processor.segment_into_clauses(doc)
                all_clauses.extend(clauses)
                
            except Exception as e:
                logger.error(f"Failed to process document {doc_url}: {e}")
                continue
        
        if not all_clauses:
            raise HTTPException(status_code=400, detail="No documents could be processed successfully")
        
        vector_search.build_index(all_clauses)
        
        answers = []
        for question in request.questions:
            try:
                relevant_clauses = vector_search.search(question, top_k=5)
                answer = await llm_processor.generate_answer(question, relevant_clauses)
                answers.append(answer)
                
            except Exception as e:
                logger.error(f"Failed to process question '{question}': {e}")
                answers.append(AnswerItem(
                    question=question,
                    answer="Failed to process question",
                    confidence=0.0,
                    reasoning=f"Processing error: {str(e)}"
                ))
        
        processing_time = time.time() - start_time
        
        response = QueryResponse(
            answers=answers,
            processing_time=round(processing_time, 2),
            total_documents_processed=len(documents)
        )
        
        logger.info(f"Query processed successfully in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# For Vercel serverless functions
handler = app
