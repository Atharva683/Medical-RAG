"""
Installation script for Medical Guidelines RAG System
Run this script to install all dependencies and set up the system
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command with error handling"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main installation process"""
    print("🏥 Medical Guidelines RAG System - Installation")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    
    print(f"✅ Python version: {sys.version}")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("⚠️ Some dependencies might not be installed. Continuing anyway...")
    
    # Check if section_chunks.txt exists
    if not os.path.exists("section_chunks.txt"):
        print("❌ section_chunks.txt not found!")
        print("Please ensure you have:")
        print("1. dsa509.pdf in the current directory")
        print("2. Run chunk_1.py to generate section_chunks.txt")
        return False
    
    print("✅ section_chunks.txt found")
    
    # Generate embeddings and index
    if not run_command("python embeddings_generator.py", "Generating embeddings and FAISS index"):
        print("❌ Failed to generate embeddings. Please check the error above.")
        return False
    
    # Test the retriever
    print("🧪 Testing the RAG system...")
    try:
        from rag_retriever import MedicalRAGRetriever
        retriever = MedicalRAGRetriever()
        
        # Test query
        result = retriever.query("What is diabetes?", use_openai=False)
        if result and result.get("answer"):
            print("✅ RAG system test successful")
        else:
            print("⚠️ RAG system test returned empty result")
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Installation completed successfully!")
    print("\nNext steps:")
    print("1. Run the Streamlit app: streamlit run streamlit_app.py")
    print("2. Or test with evaluation: python evaluation.py")
    print("3. Or use directly: python rag_retriever.py")
    print("\n📚 Check README.md for detailed usage instructions")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Installation failed. Please check the errors above.")
        sys.exit(1)
