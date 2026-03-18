"""
Complete RAG Pipeline with Re-ranking
Demonstrates a two-stage retrieval system for RAG applications.
"""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity


class RAGPipelineWithReranking:
    """
    A complete RAG pipeline with two-stage retrieval:
    1. Fast initial retrieval (bi-encoder)
    2. Accurate re-ranking (cross-encoder)
    """
    
    def __init__(
        self, 
        bi_encoder_name: str = "all-MiniLM-L6-v2",
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """Initialize the RAG pipeline with encoder models."""
        print("Initializing RAG pipeline...")
        print(f"Loading bi-encoder: {bi_encoder_name}")
        self.bi_encoder = SentenceTransformer(bi_encoder_name)
        
        print(f"Loading cross-encoder: {cross_encoder_name}")
        self.cross_encoder = CrossEncoder(cross_encoder_name)
        
        self.documents = []
        self.document_embeddings = None
        print("Pipeline initialized!\n")
    
    def add_documents(self, documents: List[str]):
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of document texts
        """
        print(f"Adding {len(documents)} documents to knowledge base...")
        self.documents = documents
        
        # Pre-compute document embeddings (bi-encoder)
        print("Computing document embeddings...")
        self.document_embeddings = self.bi_encoder.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print("Documents indexed!\n")
    
    def retrieve(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Stage 1: Fast retrieval using bi-encoder.
        
        Args:
            query: Search query
            top_k: Number of candidates to retrieve
        
        Returns:
            List of candidate documents with metadata
        """
        # Encode query
        query_embedding = self.bi_encoder.encode([query], convert_to_numpy=True)
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        
        # Get top-k candidates
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        candidates = []
        for idx in top_indices:
            candidates.append({
                'document': self.documents[idx],
                'index': int(idx),
                'bi_encoder_score': float(similarities[idx])
            })
        
        return candidates
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Stage 2: Re-rank candidates using cross-encoder.
        
        Args:
            query: Search query
            candidates: List of candidate documents from stage 1
            top_k: Number of final documents to return
        
        Returns:
            Re-ranked list of documents
        """
        # Prepare query-document pairs
        pairs = [[query, cand['document']] for cand in candidates]
        
        # Get relevance scores from cross-encoder
        scores = self.cross_encoder.predict(pairs)
        
        # Update candidates with cross-encoder scores
        for cand, score in zip(candidates, scores):
            cand['cross_encoder_score'] = float(score)
        
        # Sort by cross-encoder score
        ranked = sorted(candidates, key=lambda x: x['cross_encoder_score'], reverse=True)
        
        return ranked[:top_k]
    
    def query(self, query: str, initial_k: int = 20, final_k: int = 5) -> List[Dict]:
        """
        Complete RAG query pipeline: retrieve + re-rank.
        
        Args:
            query: Search query
            initial_k: Number of candidates to retrieve in stage 1
            final_k: Number of final documents to return after re-ranking
        
        Returns:
            Final ranked list of documents
        """
        print(f"Query: {query}\n")
        
        # Stage 1: Fast retrieval
        print(f"Stage 1: Retrieving top-{initial_k} candidates...")
        candidates = self.retrieve(query, top_k=initial_k)
        print(f"Retrieved {len(candidates)} candidates\n")
        
        # Stage 2: Re-ranking
        print(f"Stage 2: Re-ranking to top-{final_k}...")
        final_results = self.rerank(query, candidates, top_k=final_k)
        print(f"Re-ranking completed!\n")
        
        return final_results


def example_rag_pipeline():
    """Example usage of the RAG pipeline."""
    
    # Sample knowledge base
    documents = [
        "Machine learning models can be optimized using quantization techniques.",
        "Deep learning requires large amounts of training data and computational resources.",
        "Natural language processing enables machines to understand human language.",
        "Re-ranking improves retrieval quality by using cross-encoder models.",
        "Production ML systems need monitoring and A/B testing frameworks.",
        "Vector databases enable efficient similarity search for embeddings.",
        "BERT revolutionized NLP with bidirectional attention mechanisms.",
        "Two-stage retrieval combines fast bi-encoders with accurate cross-encoders.",
        "Neural ranking models learn relevance from query-document interactions.",
        "LambdaMART uses gradient boosting for learning-to-rank tasks.",
        "NDCG is the gold standard metric for ranking evaluation.",
        "Cross-attention allows models to capture fine-grained query-document relationships.",
    ]
    
    # Initialize pipeline
    pipeline = RAGPipelineWithReranking()
    pipeline.add_documents(documents)
    
    # Example queries
    queries = [
        "How to optimize machine learning models?",
        "What is re-ranking in information retrieval?",
        "Explain NDCG metric for ranking evaluation.",
    ]
    
    for query in queries:
        print("="*80)
        results = pipeline.query(query, initial_k=10, final_k=3)
        
        print("Final Results:")
        print("-"*80)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [Cross-encoder score: {result['cross_encoder_score']:.4f}]")
            print(f"   {result['document']}")
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("="*80)
    print("RAG Pipeline with Two-Stage Retrieval and Re-ranking")
    print("="*80 + "\n")
    
    example_rag_pipeline()
    print("✓ Pipeline execution completed!")

