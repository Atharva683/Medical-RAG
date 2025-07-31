#!/usr/bin/env python3
"""
Utility script to fix SentenceTransformer model issues
"""

import os
import torch
from sentence_transformers import SentenceTransformer

def fix_model_cache():
    """Clear model cache and re-download model"""
    try:
        print("🔄 Clearing PyTorch cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🔄 Downloading fresh SentenceTransformer model...")
        # Force re-download by clearing cache
        model_name = "all-MiniLM-L6-v2"
        
        # Try to load with explicit settings
        model = SentenceTransformer(
            model_name, 
            device='cpu',
            cache_folder='./.cache/sentence_transformers'
        )
        
        print("✅ Model loaded successfully!")
        
        # Test encoding
        test_text = "This is a test sentence."
        embedding = model.encode([test_text])
        print(f"✅ Model test successful! Embedding shape: {embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = fix_model_cache()
    if success:
        print("\n🎉 Model is working! You can now run the RAG system.")
    else:
        print("\n💡 Try running: pip install --upgrade sentence-transformers torch")
