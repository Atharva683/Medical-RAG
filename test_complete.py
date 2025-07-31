#!/usr/bin/env python3
"""
Complete test of the Medical RAG system with Gemini integration
"""

from rag_retriever import MedicalRAGRetriever

def test_complete_system():
    """Test the complete RAG system with Gemini"""
    print("🧪 Testing Complete Medical RAG System with Gemini")
    print("=" * 60)
    
    # Initialize retriever
    print("🔄 Initializing RAG system...")
    retriever = MedicalRAGRetriever()
    
    # Test query
    test_query = "What are the main treatment options for diabetes patients?"
    
    print(f"\n🔍 Test Query: {test_query}")
    print("-" * 40)
    
    # Test with Gemini
    if retriever.gemini_client:
        print("✅ Testing with Gemini AI...")
        result = retriever.query(test_query, top_k=3, use_gemini=True)
        
        print(f"\n🤖 **Gemini AI Response:**")
        print(result['answer'])
        print(f"\n📊 **Sources:** {result['retrieved_sections']} sections retrieved")
        
        for i, source in enumerate(result['sources'], 1):
            print(f"📚 Source {i}: {source['title']} (Score: {source['similarity_score']:.3f})")
    else:
        print("⚠️ Gemini API not configured, testing local fallback...")
        result = retriever.query(test_query, top_k=3, use_gemini=False)
        print(result['answer'])

if __name__ == "__main__":
    test_complete_system()
