"""
Script to fix SentenceTransformer model loading issues
"""
import os
import shutil
import torch
from sentence_transformers import SentenceTransformer

def fix_model_loading():
    """Fix SentenceTransformer model loading issues"""
    
    print("🔧 Fixing SentenceTransformer model loading...")
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Remove existing cache
    cache_dirs = [
        './.sentence_transformers_cache',
        os.path.expanduser('~/.cache/torch/sentence_transformers'),
        os.path.expanduser('~/.cache/huggingface/transformers')
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"✅ Cleared cache: {cache_dir}")
            except Exception as e:
                print(f"⚠️ Could not clear {cache_dir}: {e}")
    
    # Force environment settings
    os.environ['TORCH_USE_CUDA_DSA'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Try to load model fresh
    try:
        print("🔄 Loading fresh SentenceTransformer model...")
        model = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device='cpu',
            cache_folder='./.sentence_transformers_cache'
        )
        
        # Test encoding
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        print(f"✅ Model loaded successfully! Embedding shape: {embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = fix_model_loading()
    if success:
        print("🎉 Model loading fixed! You can now run the RAG system.")
    else:
        print("❌ Could not fix model loading. Try reinstalling: pip install --upgrade sentence-transformers torch")