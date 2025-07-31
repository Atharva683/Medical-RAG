#!/usr/bin/env python3
"""
Simple test to verify Gemini API configuration
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

def test_gemini_only():
    """Test just the Gemini API configuration"""
    print("🧪 Testing Gemini API Configuration")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        return False
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Test query
        response = model.generate_content("What is diabetes? Give a brief medical definition.")
        
        print("✅ Gemini API is working!")
        print("🤖 Test response:")
        print(response.text[:200] + "..." if len(response.text) > 200 else response.text)
        
        return True
        
    except Exception as e:
        print(f"❌ Error with Gemini API: {e}")
        return False

if __name__ == "__main__":
    test_gemini_only()
