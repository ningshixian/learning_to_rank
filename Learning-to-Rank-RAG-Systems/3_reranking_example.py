"""
Re-ranking Example: Using Cross-Encoders for Document Re-ranking
This demonstrates how to use pre-trained models to re-rank search results.
"""

from sentence_transformers import CrossEncoder
from typing import List, Tuple
import numpy as np


class DocumentReranker:
    """A simple re-ranker using cross-encoder models."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the re-ranker.
        
        Args:
            model_name: HuggingFace model name for the cross-encoder
        """
        print(f"Loading re-ranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        print("Model loaded successfully!")
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = None
    ) -> List[Tuple[str, float]]:
        """
        Re-rank documents based on their relevance to the query.
        
        Args:
            query: Search query
            documents: List of document texts to re-rank
            top_k: Number of top documents to return (None = all)
        
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        # Create query-document pairs for the cross-encoder
        pairs = [[query, doc] for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Sort documents by score (higher is better)
        ranked_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top_k if specified
        if top_k:
            return ranked_docs[:top_k]
        
        return ranked_docs


def example_reranking():
    """Example of re-ranking search results."""
    
    # Sample query
    query = "How to optimize machine learning models for production?"
    
    # Sample documents (initially retrieved, possibly in wrong order)
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "To optimize ML models for production, consider model quantization, batch processing, and caching strategies.",
        "The history of machine learning dates back to the 1950s with the development of neural networks.",
        "Production ML optimization requires monitoring, A/B testing, and gradual rollout strategies.",
        "Deep learning models can be deployed using containers and orchestration tools.",
        "Key optimization techniques include pruning, distillation, and hardware acceleration for inference speed.",
    ]
    
    print(f"Query: {query}\n")
    print("Original document order:")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc[:80]}...")
    
    # Initialize re-ranker
    reranker = DocumentReranker()
    
    # Re-rank documents
    print("\n" + "="*80)
    print("Re-ranking documents...")
    ranked_results = reranker.rerank(query, documents, top_k=5)
    
    print("\nRe-ranked results:")
    print("="*80)
    for i, (doc, score) in enumerate(ranked_results, 1):
        print(f"{i}. [Score: {score:.4f}] {doc[:80]}...")
    
    return ranked_results


if __name__ == "__main__":
    print("="*80)
    print("Document Re-ranking Example")
    print("="*80 + "\n")
    
    results = example_reranking()
    print("\n✓ Re-ranking completed successfully!")

