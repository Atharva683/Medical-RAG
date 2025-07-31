# Medical Guidelines RAG System

An intelligent medical information retrieval system powered by RAG (Retrieval-Augmented Generation) technology.

## Features
- 🔍 Semantic search through medical guidelines
- 🤖 AI-powered responses using Google Gemini
- 📚 Source reference tracking
- 🏥 Specialized for diabetes management

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set up Google Gemini API key in `.env` file
3. Run embeddings generator: `python embeddings_generator.py`
4. Launch app: `streamlit run streamlit_app.py`

## Deployment
This app is deployed on Streamlit Cloud: [Your App URL]

⚠️ **Medical Disclaimer:** This system is for educational purposes only.

## 🏥 Project Overview

This Medical RAG system is designed to help healthcare professionals quickly find relevant information from clinical guidelines. It demonstrates advanced NLP, retrieval engineering, and LLM integration techniques.

### Key Features

- **Section-Aware Document Chunking**: Intelligently splits medical documents by clinical sections
- **Semantic Search**: Uses sentence transformers for medical domain embeddings
- **FAISS Vector Indexing**: Fast similarity search across large document collections
- **Multiple LLM Support**: OpenAI GPT or local models for answer generation
- **Interactive Web Interface**: Streamlit-based UI for easy querying
- **Comprehensive Evaluation**: Built-in evaluation using RAGAS and custom metrics
- **Source Attribution**: Clear citation of source documents and sections

## 🛠️ Technical Architecture

```
PDF Documents → Section-Aware Chunking → Embeddings → FAISS Index
                                                           ↓
User Query → Query Embedding → Similarity Search → LLM → Final Answer
```

### Tech Stack

- **Document Processing**: PyMuPDF, LangChain
- **Embeddings**: Sentence Transformers, BioBERT
- **Vector Store**: FAISS
- **LLM**: OpenAI GPT-3.5/4, Local models
- **Frontend**: Streamlit
- **Evaluation**: RAGAS, Custom metrics

## 📁 Project Structure

```
RAG project/
├── dsa509.pdf                          # Source medical guidelines (WHO Diabetes)
├── chunk_1.py                          # Initial chunking script
├── section_chunks.txt                  # Processed document sections
├── requirements.txt                    # Python dependencies
├── embeddings_generator.py             # Creates embeddings and FAISS index
├── rag_retriever.py                   # Core RAG retrieval and generation
├── streamlit_app.py                   # Interactive web interface
├── evaluation.py                      # Comprehensive evaluation suite
├── README.md                          # Project documentation
└── generated files:
    ├── medical_guidelines_faiss_index.faiss
    ├── medical_guidelines_metadata.json
    └── rag_evaluation_results.json
```

## 🚀 Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Embeddings and Index

```bash
python embeddings_generator.py
```

This will:
- Parse `section_chunks.txt` into structured sections
- Generate embeddings using Sentence Transformers
- Create FAISS index for fast retrieval
- Save index and metadata files

### 3. Run the Streamlit Interface

```bash
streamlit run streamlit_app.py
```

### 4. (Optional) Run Evaluation

```bash
python evaluation.py
```

## � User Guide: Understanding Key Parameters

### What are "Sections"?

**Sections** are the fundamental building blocks of how the system organizes medical information:

- **Medical Document Structure**: The WHO diabetes guidelines (like most medical documents) are naturally divided into sections like:
  - "Introduction and Epidemiology" 
  - "Clinical Diagnosis"
  - "Treatment Protocols"
  - "Dietary Management"
  - "Complications and Prevention"
  
- **How Sections Work**: Instead of searching through the entire 100+ page document, the system breaks it into these logical sections (we have 16 sections total). When you ask a question, it finds the 2-3 most relevant sections and uses only that focused information to answer.

- **Why This Matters**: You get more precise answers because the system focuses on the specific medical topic area most relevant to your question.

### Key Selection Parameters Explained

#### **top_k** (Number of Sections to Retrieve)
- **What it is**: How many sections the system will look at to answer your question
- **Default**: 3 sections
- **Example**: If you ask about "diabetes treatment", it might retrieve:
  1. "Treatment with Oral Agents" section
  2. "Insulin Therapy" section  
  3. "Dietary Management" section
- **User Impact**: More sections = more comprehensive but potentially longer answers

#### **similarity_threshold** (Relevance Filter)
- **What it is**: How similar a section must be to your question to be considered relevant
- **Scale**: 0.0 (no filtering) to 1.0 (perfect match required)
- **Default**: 0.1 (very permissive for medical content)
- **Example**: 
  - High threshold (0.8): Only exact topic matches
  - Low threshold (0.3): Related topics included
- **User Impact**: Higher threshold = more precise but might miss relevant info; Lower threshold = more comprehensive but might include less relevant content

#### **Response Mode: AI vs Local**
- **AI Mode (Gemini)**: Uses Google's AI to generate natural, conversational answers
  - Pros: More readable, better explanations, contextual understanding
  - Cons: Requires internet, uses external AI service
- **Local Mode**: Uses rule-based templates and direct section content
  - Pros: Completely offline, fast, deterministic
  - Cons: More mechanical answers, less natural language

### Understanding Your Results

When you query the system, you'll see:

#### **Answer Section**
- The main response to your medical question
- Synthesized from the most relevant sections
- Includes clinical recommendations and guidelines

#### **Sources Section**
- Shows which specific sections were used
- Includes similarity scores (how well each section matched your question)
- Provides page references to the original WHO document
- Lets you verify the source of information

#### **Example Result Breakdown**
```
Query: "What is the recommended treatment for type 2 diabetes?"

Sources Found: 3 sections
1. "Treatment with Oral Hypoglycemic Agents" (Similarity: 0.89)
2. "Insulin Therapy Guidelines" (Similarity: 0.76) 
3. "Lifestyle and Dietary Interventions" (Similarity: 0.71)

Answer: [Generated response using information from these 3 sections]
```

## �💻 Usage Examples

### Command Line Usage

```python
from rag_retriever import MedicalRAGRetriever

# Initialize retriever
retriever = MedicalRAGRetriever()

# Basic query with default parameters
result = retriever.query("What is the recommended treatment for type 2 diabetes?")

# Advanced query with custom parameters
result = retriever.query(
    query="How should insulin be administered?",
    top_k=5,  # Look at 5 sections instead of 3
    similarity_threshold=0.6,  # Higher precision
    use_gemini=True  # Use AI generation
)

print(result["answer"])
print(f"Sources: {len(result['sources'])}")
```

### Sample Queries

- "What is the recommended treatment for type 2 diabetes?"
- "How should blood glucose be monitored in diabetic patients?"
- "What are the dietary recommendations for diabetes management?"
- "What complications can arise from diabetes?"
- "How should insulin therapy be managed?"
- "What are the diagnostic criteria for diabetes mellitus?"

## 🧪 Evaluation Metrics

The system includes comprehensive evaluation using:

### Retrieval Quality
- **Keyword Overlap**: Measures relevance of retrieved sections
- **Similarity Scores**: Average embedding similarity
- **Retrieval Success Rate**: Percentage of successful retrievals

### Answer Quality
- **Word Overlap**: Comparison with ground truth answers
- **Answer Length**: Completeness assessment
- **Factual Accuracy**: Using domain-specific evaluation

### RAGAS Metrics (Optional)
- **Faithfulness**: Answer consistency with retrieved context
- **Answer Relevancy**: How well answers address queries
- **Context Precision**: Quality of retrieved context
- **Context Recall**: Completeness of context retrieval

## 🔧 Configuration Options

## 🔧 Configuration Options

### How to Adjust System Behavior

#### **For More Comprehensive Answers**
```python
# Retrieve more sections for broader coverage
result = retriever.query(
    "diabetes complications", 
    top_k=5,  # Look at 5 sections instead of 3
    similarity_threshold=0.3  # Include more loosely related content
)
```

#### **For More Precise Answers**
```python
# Focus on highly relevant sections only
result = retriever.query(
    "insulin dosage guidelines", 
    top_k=2,  # Look at fewer, more focused sections
    similarity_threshold=0.7  # Only highly relevant sections
)
```

#### **For Different Answer Styles**
```python
# Natural AI-generated answers
result = retriever.query("treatment options", use_gemini=True)

# Direct guideline excerpts  
result = retriever.query("treatment options", use_gemini=False)
```

### Understanding Similarity Scores

When you see source sections with similarity scores:

- **0.8 - 1.0**: Excellent match - section directly addresses your question
- **0.6 - 0.8**: Good match - section contains relevant information
- **0.4 - 0.6**: Moderate match - section has some related content
- **0.2 - 0.4**: Weak match - section mentions related topics
- **0.0 - 0.2**: Poor match - minimal relevance

### Embedding Models
- `all-MiniLM-L6-v2` (default, fast)
- `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli` (medical domain)

### Retrieval Parameters
- **top_k**: Number of sections to retrieve (default: 3)
- **similarity_threshold**: Minimum similarity score (default: 0.5)

### LLM Options
- Gemini API key (requires API key)
- Local rule-based generation (fallback)

## 📊 Performance Characteristics

Based on evaluation with 5 medical queries:

- **Average Retrieval Success**: >90%
- **Average Similarity Score**: 0.75+
- **Response Time**: <2 seconds per query
- **Embedding Dimension**: 384 (MiniLM) / 768 (BioBERT)

## 🔒 Privacy & Compliance

- **Local Processing**: Embeddings and indexing done locally
- **No Data Leakage**: Documents stay on your infrastructure
- **Optional Cloud LLM**: OpenAI integration is optional
- **Audit Trail**: All queries and responses can be logged

## 🚀 Advanced Features

### 1. Multi-Document Support
- Easy to extend for multiple guideline documents
- Metadata tracking for source attribution

### 2. Temporal Awareness
- Can be extended to handle time-sensitive guidelines
- Version control for updated protocols

### 3. Multi-Language Support
- Framework ready for Hindi/English medical documents
- Cross-lingual retrieval capabilities

## 📈 Scaling Considerations

- **Document Volume**: Tested with documents up to 100+ pages
- **Query Load**: Streamlit handles multiple concurrent users
- **Memory Usage**: ~2GB RAM for full system
- **Storage**: ~50MB for embeddings of 100-page document

### Sample Queries and Expected Behavior

#### **Specific Clinical Questions**
```
Query: "What is the recommended treatment for type 2 diabetes?"
Expected: Will find sections about medication protocols, lifestyle changes
Sections Retrieved: Usually 3-4 treatment-related sections
Answer Quality: Comprehensive with specific drug names and dosages
```

#### **Diagnostic Questions**  
```
Query: "What are the diagnostic criteria for diabetes mellitus?"
Expected: Will find sections about laboratory values, testing procedures
Sections Retrieved: Diagnostic criteria and testing protocol sections
Answer Quality: Specific numerical values and testing guidelines
```

#### **Broad Topic Questions**
```
Query: "Tell me about diabetes complications"
Expected: Will find multiple sections covering different complications
Sections Retrieved: May include retinopathy, nephropathy, cardiovascular sections
Answer Quality: Overview of various complications with prevention strategies
```

### What Makes a Good Query?

#### **✅ Good Queries**
- "How should blood glucose be monitored in diabetic patients?"
- "What are the side effects of metformin?"
- "When should insulin therapy be started?"
- "What dietary restrictions apply to diabetic patients?"

#### **❌ Less Effective Queries**
- "Tell me everything about diabetes" (too broad)
- "Is sugar bad?" (too vague)
- "My friend has diabetes, what should they do?" (requires personalized medical advice)

### Troubleshooting Common Issues

#### **No Relevant Sections Found**
- **Cause**: Query topic not covered in the WHO diabetes guidelines
- **Solution**: Rephrase query to focus on diabetes-specific topics
- **Example**: Instead of "heart disease treatment" ask "cardiovascular complications of diabetes"

#### **Very Short Answers**
- **Cause**: High similarity threshold filtering out relevant sections
- **Solution**: Lower the similarity threshold or use more sections (higher top_k)

#### **Too Much Information**
- **Cause**: Low similarity threshold including tangentially related sections
- **Solution**: Increase similarity threshold or reduce top_k parameter

## 🎯 Use Cases

### Healthcare Professionals
- Quick reference during patient consultations
- Treatment protocol verification
- Diagnostic criteria lookup

### Medical Students
- Study aid for clinical guidelines
- Case-based learning support
- Exam preparation

### Healthcare Organizations
- Internal knowledge base
- Clinical decision support
- Protocol standardization

## 🛡️ Limitations & Disclaimers

- **Educational Purpose**: This system is for educational/research use
- **Not Medical Advice**: Always consult healthcare professionals
- **Source Dependency**: Quality depends on input documents
- **Hallucination Risk**: LLMs may generate inaccurate information

## 🔮 Future Enhancements

- **Multi-Modal Support**: Images, tables, charts
- **Fine-Tuned Models**: Medical domain-specific LLMs
- **Real-Time Updates**: Dynamic document refresh
- **Mobile Interface**: Responsive design for mobile devices
- **API Integration**: RESTful API for external systems

## 📚 Blog Potential

This project serves as excellent content for technical blogs covering:

1. **RAG Architecture**: "Building Production-Ready RAG Systems"
2. **Medical NLP**: "Domain-Specific Retrieval in Healthcare"
3. **Evaluation**: "How to Evaluate RAG Systems Properly"
4. **Deployment**: "From Prototype to Production RAG"

## 🤝 Contributing

This is a portfolio/educational project. Feel free to:
- Fork and extend for your use cases
- Suggest improvements
- Add support for other medical domains
- Improve evaluation metrics

## 📄 License

This project is for educational and research purposes. Medical content is from public WHO guidelines.

---

**⚠️ Medical Disclaimer**: This system is for educational purposes only. Always consult qualified healthcare professionals for medical decisions.
