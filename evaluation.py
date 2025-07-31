from datasets import Dataset
import json
import numpy as np
from typing import List, Dict
import pandas as pd

# Try importing RAGAS components
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    RAGAS_AVAILABLE = True
except ImportError:
    print("⚠️ RAGAS not available. Install with: pip install ragas")
    RAGAS_AVAILABLE = False

class MedicalRAGEvaluator:
    """
    Comprehensive evaluation system for Medical RAG pipeline
    """
    
    def __init__(self, retriever=None):
        self.retriever = retriever
        self.evaluation_results = {}
        
    def create_evaluation_dataset(self) -> List[Dict]:
        """
        Create evaluation dataset with medical queries and expected answers
        """
        evaluation_data = [
            {
                "question": "What is the recommended treatment for type 2 diabetes?",
                "ground_truth": "Treatment includes dietary therapy, exercise, oral hypoglycemic drugs, and insulin when needed. Diet should provide 25-35% calories from fat, 10-15% from protein, and 50-60% from carbohydrates.",
                "context_keywords": ["dietary therapy", "exercise", "oral hypoglycemic", "insulin"]
            },
            {
                "question": "How should blood glucose be monitored in diabetic patients?",
                "ground_truth": "Blood glucose testing is more informative than urine testing. Self-monitoring should be encouraged with frequency determined by treatment type and therapy targets.",
                "context_keywords": ["blood glucose testing", "self-monitoring", "urine testing"]
            },
            {
                "question": "What are the diagnostic criteria for diabetes mellitus?",
                "ground_truth": "Fasting blood glucose ≥7.8 mmol/L (140 mg/dl) or 2-hour post-glucose load ≥11.1 mmol/L (200 mg/dl) in plasma. Random glucose can also be used if exceeding diagnostic values.",
                "context_keywords": ["fasting", "glucose tolerance test", "diagnostic values"]
            },
            {
                "question": "What complications can arise from diabetes?",
                "ground_truth": "Long-term complications include cardiovascular diseases, retinopathy, nephropathy, neuropathy, and foot complications. Good glycemic control can reduce these risks.",
                "context_keywords": ["retinopathy", "nephropathy", "neuropathy", "cardiovascular"]
            },
            {
                "question": "What are the dietary recommendations for diabetes management?",
                "ground_truth": "Fat should provide 25-35% of calories (saturated fat <10%), protein 10-15%, carbohydrates 50-60%. Limit cholesterol to <300mg daily, restrict salt, moderate artificial sweeteners.",
                "context_keywords": ["fat intake", "protein", "carbohydrates", "cholesterol", "salt"]
            }
        ]
        
        return evaluation_data
    
    def evaluate_retrieval_quality(self, evaluation_data: List[Dict]) -> Dict:
        """
        Evaluate retrieval quality using keyword overlap and relevance
        """
        if not self.retriever:
            return {"error": "No retriever provided"}
        
        retrieval_scores = []
        
        for item in evaluation_data:
            question = item["question"]
            expected_keywords = item["context_keywords"]
            
            # Retrieve contexts with lower threshold for testing
            retrieved_sections = self.retriever.retrieve_relevant_sections(question, top_k=3, similarity_threshold=0.1)
            
            if retrieved_sections:
                # Calculate keyword overlap
                retrieved_text = " ".join([section["content"].lower() for section in retrieved_sections])
                keyword_matches = sum(1 for keyword in expected_keywords if keyword.lower() in retrieved_text)
                keyword_score = keyword_matches / len(expected_keywords)
                
                # Average similarity score
                avg_similarity = np.mean([section["similarity_score"] for section in retrieved_sections])
                
                retrieval_scores.append({
                    "question": question,
                    "keyword_overlap": keyword_score,
                    "avg_similarity": avg_similarity,
                    "num_retrieved": len(retrieved_sections)
                })
            else:
                retrieval_scores.append({
                    "question": question,
                    "keyword_overlap": 0.0,
                    "avg_similarity": 0.0,
                    "num_retrieved": 0
                })
        
        # Calculate overall metrics
        overall_metrics = {
            "avg_keyword_overlap": np.mean([score["keyword_overlap"] for score in retrieval_scores]),
            "avg_similarity": np.mean([score["avg_similarity"] for score in retrieval_scores]),
            "avg_retrieved_sections": np.mean([score["num_retrieved"] for score in retrieval_scores]),
            "retrieval_success_rate": sum(1 for score in retrieval_scores if score["num_retrieved"] > 0) / len(retrieval_scores)
        }
        
        return {
            "individual_scores": retrieval_scores,
            "overall_metrics": overall_metrics
        }
    
    def evaluate_answer_quality(self, evaluation_data: List[Dict]) -> Dict:
        """
        Evaluate answer quality using simple heuristics
        """
        if not self.retriever:
            return {"error": "No retriever provided"}
        
        answer_scores = []
        
        for item in evaluation_data:
            question = item["question"]
            ground_truth = item["ground_truth"]
            
            # Generate answer
            result = self.retriever.query(question, use_gemini=False)
            generated_answer = result["answer"]
            
            # Simple evaluation metrics
            answer_length = len(generated_answer.split())
            
            # Check if answer contains key terms from ground truth
            ground_truth_words = set(ground_truth.lower().split())
            answer_words = set(generated_answer.lower().split())
            word_overlap = len(ground_truth_words.intersection(answer_words)) / len(ground_truth_words)
            
            # Check if answer indicates "no information found"
            no_info_indicators = ["no relevant", "not found", "no information", "cannot find"]
            contains_no_info = any(indicator in generated_answer.lower() for indicator in no_info_indicators)
            
            answer_scores.append({
                "question": question,
                "answer_length": answer_length,
                "word_overlap": word_overlap,
                "contains_no_info": contains_no_info,
                "answer_preview": generated_answer[:200] + "..."
            })
        
        # Calculate overall metrics
        overall_metrics = {
            "avg_answer_length": np.mean([score["answer_length"] for score in answer_scores]),
            "avg_word_overlap": np.mean([score["word_overlap"] for score in answer_scores]),
            "no_info_rate": sum(1 for score in answer_scores if score["contains_no_info"]) / len(answer_scores),
            "successful_answers": sum(1 for score in answer_scores if not score["contains_no_info"]) / len(answer_scores)
        }
        
        return {
            "individual_scores": answer_scores,
            "overall_metrics": overall_metrics
        }
    
    def evaluate_with_ragas(self, evaluation_data: List[Dict]) -> Dict:
        """
        Evaluate using RAGAS framework (if available)
        """
        if not RAGAS_AVAILABLE:
            return {"error": "RAGAS not available. Install with: pip install ragas"}
        
        if not self.retriever:
            return {"error": "No retriever provided"}
        
        # Prepare data for RAGAS
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        for item in evaluation_data:
            question = item["question"]
            ground_truth = item["ground_truth"]
            
            # Generate answer and retrieve contexts
            result = self.retriever.query(question, use_gemini=False)
            
            questions.append(question)
            answers.append(result["answer"])
            ground_truths.append(ground_truth)
            
            # Extract contexts
            context_list = [source["content_preview"] for source in result["sources"]]
            contexts.append(context_list)
        
        # Create RAGAS dataset
        try:
            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truths": ground_truths
            })
            
            # Evaluate with RAGAS metrics
            result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
            )
            
            return {"ragas_results": result.to_pandas().to_dict()}
            
        except Exception as e:
            return {"error": f"RAGAS evaluation failed: {str(e)}"}
    
    def run_comprehensive_evaluation(self) -> Dict:
        """
        Run all evaluation methods and compile results
        """
        print("🧪 Starting comprehensive RAG evaluation...")
        
        # Create evaluation dataset
        eval_data = self.create_evaluation_dataset()
        
        # Run evaluations
        results = {
            "evaluation_dataset_size": len(eval_data),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Retrieval evaluation
        print("📊 Evaluating retrieval quality...")
        retrieval_results = self.evaluate_retrieval_quality(eval_data)
        results["retrieval_evaluation"] = retrieval_results
        
        # Answer quality evaluation
        print("💬 Evaluating answer quality...")
        answer_results = self.evaluate_answer_quality(eval_data)
        results["answer_evaluation"] = answer_results
        
        # RAGAS evaluation (if available)
        if RAGAS_AVAILABLE:
            print("🔬 Running RAGAS evaluation...")
            ragas_results = self.evaluate_with_ragas(eval_data)
            results["ragas_evaluation"] = ragas_results
        else:
            results["ragas_evaluation"] = {"error": "RAGAS not available"}
        
        self.evaluation_results = results
        return results
    
    def save_evaluation_results(self, filename: str = "rag_evaluation_results.json"):
        """Save evaluation results to file"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
        print(f"💾 Evaluation results saved to {filename}")
    
    def print_evaluation_summary(self):
        """Print a summary of evaluation results"""
        if not self.evaluation_results:
            print("❌ No evaluation results available. Run run_comprehensive_evaluation() first.")
            return
        
        print("\n" + "="*60)
        print("🏥 Medical RAG Evaluation Summary")
        print("="*60)
        
        # Retrieval metrics
        if "retrieval_evaluation" in self.evaluation_results:
            retrieval = self.evaluation_results["retrieval_evaluation"]["overall_metrics"]
            print(f"\n📊 Retrieval Performance:")
            print(f"  • Average Keyword Overlap: {retrieval['avg_keyword_overlap']:.2f}")
            print(f"  • Average Similarity Score: {retrieval['avg_similarity']:.2f}")
            print(f"  • Retrieval Success Rate: {retrieval['retrieval_success_rate']:.2f}")
            print(f"  • Avg Sections Retrieved: {retrieval['avg_retrieved_sections']:.1f}")
        
        # Answer metrics
        if "answer_evaluation" in self.evaluation_results:
            answer = self.evaluation_results["answer_evaluation"]["overall_metrics"]
            print(f"\n💬 Answer Quality:")
            print(f"  • Average Answer Length: {answer['avg_answer_length']:.1f} words")
            print(f"  • Word Overlap with Ground Truth: {answer['avg_word_overlap']:.2f}")
            print(f"  • Successful Answer Rate: {answer['successful_answers']:.2f}")
            print(f"  • No Information Rate: {answer['no_info_rate']:.2f}")
        
        # RAGAS metrics (if available)
        if "ragas_evaluation" in self.evaluation_results and "ragas_results" in self.evaluation_results["ragas_evaluation"]:
            print(f"\n🔬 RAGAS Metrics:")
            ragas = self.evaluation_results["ragas_evaluation"]["ragas_results"]
            for metric, value in ragas.items():
                if isinstance(value, (int, float)):
                    print(f"  • {metric}: {value:.3f}")

def main():
    """Main evaluation function"""
    # Import retriever
    try:
        from rag_retriever import MedicalRAGRetriever
        retriever = MedicalRAGRetriever()
        
        # Initialize evaluator
        evaluator = MedicalRAGEvaluator(retriever)
        
        # Run evaluation
        results = evaluator.run_comprehensive_evaluation()
        
        # Print summary
        evaluator.print_evaluation_summary()
        
        # Save results
        evaluator.save_evaluation_results()
        
        print(f"\n✅ Evaluation complete! Results saved to rag_evaluation_results.json")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        print("Make sure to run embeddings_generator.py first!")

if __name__ == "__main__":
    main()
