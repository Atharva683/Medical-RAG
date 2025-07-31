#!/usr/bin/env python3
"""
Test script for the Medical RAG system with Gemini API
"""

from rag_retriever import MedicalRAGRetriever

def test_gemini_integration():
    """Test the RAG system with Gemini API"""
    print("🧪 Testing Medical RAG System with Gemini")
    print("=" * 50)
    
    # Initialize retriever
    retriever = MedicalRAGRetriever()
    
    # Test query
    test_query = "What are the main symptoms of diabetes?"
    
    print(f"🔍 Test Query: {test_query}")
    print("-" * 30)
    
    # Test with Gemini (if available)
    if retriever.gemini_client:
        print("✅ Gemini API configured, testing with AI generation...")
        result = retriever.query(test_query, top_k=2, use_gemini=True)
        print(f"\n🤖 Gemini Response:")
        print(result['answer'])
    else:
        print("ℹ️  Gemini API not configured, using local fallback...")
        result = retriever.query(test_query, top_k=2, use_gemini=False)
        print(f"\n🔧 Local Response:")
        print(result['answer'])
    
    print(f"\n📊 Retrieved {result['retrieved_sections']} relevant sections")
    
    if result['sources']:
        print(f"📚 Top source: {result['sources'][0]['title']}")
        print(f"🎯 Similarity score: {result['sources'][0]['similarity_score']:.3f}")

if __name__ == "__main__":
    test_gemini_integration()
