import re
import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import os

class MedicalGuidelinesEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a sentence transformer model.
        For medical domain, we could use:
        - 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli' (medical domain)
        - 'all-MiniLM-L6-v2' (general, faster)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.sections = []
        self.embeddings = None
        self.index = None
        
    def load_and_parse_chunks(self, chunks_file: str) -> List[Dict]:
        """Parse the section_chunks.txt file into structured data"""
        print(f"Loading chunks from: {chunks_file}")
        
        with open(chunks_file, "r", encoding="utf-8") as f:
            raw_text = f.read()
        
        # Parse sections using regex
        pattern = r"### (.*?)\n(.*?)\n={80}"
        matches = re.findall(pattern, raw_text, re.DOTALL)
        
        sections = []
        for i, (title, content) in enumerate(matches):
            sections.append({
                "id": i,
                "title": title.strip(),
                "content": content.strip(),
                "word_count": len(content.split()),
                "source": "WHO Diabetes Guidelines"
            })
        
        self.sections = sections
        print(f"✅ Parsed {len(sections)} sections")
        return sections
    
    def generate_embeddings(self) -> np.ndarray:
        """Generate embeddings for all sections"""
        if not self.sections:
            raise ValueError("No sections loaded. Call load_and_parse_chunks first.")
        
        print("🧠 Generating embeddings...")
        texts_to_embed = [section["content"] for section in self.sections]
        
        # Generate embeddings with progress bar
        embeddings = self.model.encode(
            texts_to_embed, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        self.embeddings = embeddings.astype("float32")
        print(f"✅ Generated embeddings shape: {self.embeddings.shape}")
        return self.embeddings
    
    def create_faiss_index(self) -> faiss.Index:
        """Create and populate FAISS index"""
        if self.embeddings is None:
            raise ValueError("No embeddings generated. Call generate_embeddings first.")
        
        print("📊 Creating FAISS index...")
        
        # Create index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.index.add(self.embeddings)
        
        print(f"✅ FAISS index created with {self.index.ntotal} vectors")
        return self.index
    
    def save_index_and_metadata(self, base_path: str = "medical_guidelines"):
        """Save FAISS index and metadata"""
        if self.index is None:
            raise ValueError("No index created. Call create_faiss_index first.")
        
        # Save FAISS index
        index_path = f"{base_path}_faiss_index.faiss"
        faiss.write_index(self.index, index_path)
        print(f"💾 FAISS index saved to: {index_path}")
        
        # Save metadata
        metadata_path = f"{base_path}_metadata.json"
        metadata = {
            "sections": self.sections,
            "model_name": self.model.get_sentence_embedding_dimension(),
            "total_sections": len(self.sections),
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else None
        }
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"💾 Metadata saved to: {metadata_path}")
        
        return index_path, metadata_path
    
    def create_sections_dataframe(self) -> pd.DataFrame:
        """Create a pandas DataFrame for analysis"""
        if not self.sections:
            raise ValueError("No sections loaded.")
        
        df = pd.DataFrame(self.sections)
        return df

def main():
    """Main execution function"""
    print("🏥 Medical Guidelines RAG - Embedding Pipeline")
    print("=" * 50)
    
    # Initialize embedder
    embedder = MedicalGuidelinesEmbedder()
    
    # Load and parse chunks
    sections = embedder.load_and_parse_chunks("section_chunks.txt")
    
    # Generate embeddings
    embeddings = embedder.generate_embeddings()
    
    # Create FAISS index
    index = embedder.create_faiss_index()
    
    # Save everything
    index_path, metadata_path = embedder.save_index_and_metadata()
    
    # Create analysis dataframe
    df = embedder.create_sections_dataframe()
    
    # Print summary statistics
    print("\n📊 Section Analysis:")
    print(f"Total sections: {len(sections)}")
    print(f"Average word count: {df['word_count'].mean():.1f}")
    print(f"Longest section: {df['word_count'].max()} words")
    print(f"Shortest section: {df['word_count'].min()} words")
    
    print("\n🔝 Top 5 Longest Sections:")
    top_sections = df.nlargest(5, 'word_count')[['title', 'word_count']]
    for _, row in top_sections.iterrows():
        print(f"  • {row['title']}: {row['word_count']} words")
    
    print(f"\n✅ Medical Guidelines RAG embeddings ready!")
    print(f"Index file: {index_path}")
    print(f"Metadata file: {metadata_path}")

if __name__ == "__main__":
    main()
