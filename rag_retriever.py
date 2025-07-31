import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MedicalRAGRetriever:
    def __init__(self, 
                 index_path: str = "medical_guidelines_faiss_index.faiss",
                 metadata_path: str = "medical_guidelines_metadata.json",
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG retriever system
        """
        self.model_name = model_name
        self.index = None
        self.sections = []
        
        # Load index and metadata first
        self.load_index_and_metadata(index_path, metadata_path)
        
        # Initialize SentenceTransformer with enhanced error handling
        try:
            print("🔄 Loading SentenceTransformer model...")
            
            # Clear any existing meta tensors and cache
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force CPU usage and disable meta device
            import os
            os.environ['TORCH_USE_CUDA_DSA'] = '0'
            
            # Try loading with explicit settings to avoid meta tensor issues
            self.model = SentenceTransformer(
                model_name, 
                device='cpu',
                cache_folder='./.sentence_transformers_cache'
            )
            print("✅ SentenceTransformer model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading SentenceTransformer: {e}")
            
            # Try alternative initialization methods
            try:
                print("🔄 Trying alternative loading method...")
                
                # Clear the cache directory
                import shutil
                cache_dir = './.sentence_transformers_cache'
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                
                # Force download fresh model
                self.model = SentenceTransformer(
                    model_name,
                    device='cpu',
                    trust_remote_code=False,
                    use_auth_token=False
                )
                print("✅ SentenceTransformer model loaded on retry")
                
            except Exception as e2:
                print(f"❌ Failed to load model on retry: {e2}")
                
                # Final fallback - try a different model
                try:
                    print("🔄 Trying fallback model...")
                    self.model = SentenceTransformer(
                        'paraphrase-MiniLM-L6-v2',
                        device='cpu'
                    )
                    print("✅ Fallback model loaded successfully")
                    
                except Exception as e3:
                    print(f"❌ All model loading attempts failed: {e3}")
                    raise RuntimeError(f"Cannot initialize any SentenceTransformer model. Try running: pip install --upgrade sentence-transformers torch")
        # Initialize Gemini client (optional, can use local models)
        self.gemini_client = None
        if os.getenv("GEMINI_API_KEY"):
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.gemini_client = genai.GenerativeModel('gemini-1.5-flash')

    def load_index_and_metadata(self, index_path: str, metadata_path: str):
        """Load FAISS index and metadata"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            print(f"✅ Loaded FAISS index from {index_path}")
            
            # Load metadata
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                self.sections = metadata["sections"]
            print(f"✅ Loaded {len(self.sections)} sections from metadata")
            
        except Exception as e:
            print(f"❌ Error loading index/metadata: {e}")
            print("Please run embeddings_generator.py first!")
            raise

    def retrieve_relevant_sections(self, 
                                 query: str, 
                                 top_k: int = 3,
                                 similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Retrieve most relevant sections for a query
        """
        if self.index is None:
            raise ValueError("Index not loaded")
        
        # Encode query
        query_embedding = self.model.encode([query]).astype("float32")
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.sections):
                section = self.sections[idx].copy()
                section["similarity_score"] = float(1 / (1 + distance))  # Convert distance to similarity
                section["rank"] = i + 1
                
                # Filter by similarity threshold
                if section["similarity_score"] >= similarity_threshold:
                    results.append(section)
        
        return results

    def generate_answer_gemini(self, query: str, retrieved_sections: List[Dict]) -> str:
        """Generate answer using Google Gemini"""
        if not self.gemini_client:
            return "Gemini API key not configured. Please set GEMINI_API_KEY environment variable."
        
        # Prepare context from retrieved sections
        context = "\n\n".join([
            f"Section: {section['title']}\nContent: {section['content'][:1000]}..."
            for section in retrieved_sections
        ])
        
        # Create prompt
        prompt = f"""You are a medical assistant helping healthcare professionals with clinical guidelines.

Based on the retrieved medical guideline sections below, provide a precise and evidence-based answer to the query.

Query: {query}

Retrieved Clinical Guidelines:
{context}

Instructions:
1. Provide a clear, concise answer based only on the retrieved content
2. Include specific recommendations when available
3. Mention the source section(s) in your response
4. If the information is not sufficient, clearly state the limitations
5. Always emphasize consulting with healthcare professionals for patient care

Answer:"""

        try:
            response = self.gemini_client.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {e}"

    def generate_answer_local(self, query: str, retrieved_sections: List[Dict]) -> str:
        """Generate answer using rule-based approach (fallback)"""
        if not retrieved_sections:
            return "No relevant sections found for your query. Please try rephrasing your question."
        
        # Simple concatenation with source attribution
        answer_parts = []
        answer_parts.append(f"Based on the medical guidelines, here's what I found:\n")
        
        for section in retrieved_sections:
            # Extract key information
            content_preview = section['content'][:300] + "..." if len(section['content']) > 300 else section['content']
            answer_parts.append(f"\n**From section '{section['title']}':**")
            answer_parts.append(content_preview)
            answer_parts.append(f"(Relevance score: {section['similarity_score']:.2f})")
        
        answer_parts.append("\n\n**Important:** Always consult with healthcare professionals for patient care decisions.")
        
        return "\n".join(answer_parts)

    def query(self, 
              question: str, 
              top_k: int = 3, 
              use_gemini: bool = True,
              similarity_threshold: float = 0.5) -> Dict:
        """
        Main query function - retrieve and generate answer
        """
        print(f"🔍 Processing query: {question}")
        
        # Retrieve relevant sections
        retrieved_sections = self.retrieve_relevant_sections(
            question, top_k=top_k, similarity_threshold=similarity_threshold
        )
        
        if not retrieved_sections:
            return {
                "query": question,
                "answer": "No relevant information found in the medical guidelines.",
                "sources": [],
                "retrieved_sections": 0
            }
        
        # Generate answer
        if use_gemini and self.gemini_client:
            answer = self.generate_answer_gemini(question, retrieved_sections)
        else:
            answer = self.generate_answer_local(question, retrieved_sections)
        
        # Prepare sources
        sources = [
            {
                "title": section["title"],
                "similarity_score": section["similarity_score"],
                "content_preview": section["content"][:200] + "..."
            }
            for section in retrieved_sections
        ]
        
        return {
            "query": question,
            "answer": answer,
            "sources": sources,
            "retrieved_sections": len(retrieved_sections)
        }

def demo_queries():
    """Demo function with sample medical queries"""
    # Initialize retriever
    retriever = MedicalRAGRetriever()
    
    # Sample medical queries
    queries = [
        "What is the recommended treatment for type 2 diabetes?",
        "How should blood glucose be monitored in diabetic patients?",
        "What are the dietary recommendations for diabetes management?",
        "What complications can arise from diabetes?",
        "How should insulin therapy be managed?",
        "What are the diagnostic criteria for diabetes mellitus?"
    ]
    
    print("🏥 Medical Guidelines RAG - Demo Queries")
    print("=" * 60)
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 40)
        
        result = retriever.query(query, top_k=2, use_gemini=False)
        
        print(f"Answer: {result['answer'][:300]}...")
        print(f"Sources found: {result['retrieved_sections']}")
        
        if result['sources']:
            print("Top source:", result['sources'][0]['title'])

if __name__ == "__main__":
    demo_queries()
