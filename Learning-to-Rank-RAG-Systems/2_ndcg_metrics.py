"""
Ranking Evaluation Metrics
Implementation of NDCG, MRR, and MAP for evaluating ranking systems.
"""

import numpy as np
from typing import List, Union

def dcg_at_k(relevances: List[float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at position k.
    
    Args:
        relevances: List of relevance scores (higher is better)
        k: Cutoff position
    
    Returns:
        DCG@k score
    """
    relevances = np.asfarray(relevances)[:k]
    if relevances.size:
        # DCG formula: sum of (2^rel_i - 1) / log2(i + 1)
        return np.sum((2 ** relevances - 1) / np.log2(np.arange(2, relevances.size + 2)))
    return 0.0


def ndcg_at_k(relevances: List[float], k: int = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at position k.
    
    Args:
        relevances: List of relevance scores for the ranked list
        k: Cutoff position (if None, uses all positions)
    
    Returns:
        NDCG@k score (0 to 1, where 1 is perfect)
    """
    if k is None:
        k = len(relevances)
    
    # Calculate DCG for current ranking
    dcg = dcg_at_k(relevances, k)
    
    # Calculate IDCG (Ideal DCG) from perfect ranking
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    
    if not idcg:
        return 0.0
    
    return dcg / idcg


def mean_reciprocal_rank(relevance_lists: List[List[int]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    MRR = (1/|Q|) * sum(1/rank_i)
    where rank_i is the position of the first relevant document for query i.
    
    Args:
        relevance_lists: List of binary relevance lists (1=relevant, 0=irrelevant)
    
    Returns:
        MRR score
    """
    reciprocal_ranks = []
    
    for relevances in relevance_lists:
        # Find the first relevant document (relevance = 1)
        for idx, rel in enumerate(relevances):
            if rel >= 1:  # At least somewhat relevant
                reciprocal_ranks.append(1.0 / (idx + 1))
                break
        else:
            # No relevant document found
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def average_precision_at_k(relevances: List[int], k: int) -> float:
    """
    Calculate Average Precision at k.
    
    Args:
        relevances: Binary relevance list (1=relevant, 0=irrelevant)
        k: Cutoff position
    
    Returns:
        Average Precision@k
    """
    relevances = relevances[:k]
    if not relevances or sum(relevances) == 0:
        return 0.0
    
    precisions = []
    relevant_count = 0
    
    for i, rel in enumerate(relevances):
        if rel >= 1:
            relevant_count += 1
            precision = relevant_count / (i + 1)
            precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0.0


def mean_average_precision(relevance_lists: List[List[int]], k: int = None) -> float:
    """
    Calculate Mean Average Precision (MAP).
    
    Args:
        relevance_lists: List of binary relevance lists
        k: Cutoff position (if None, uses all positions)
    
    Returns:
        MAP score
    """
    average_precisions = []
    
    for relevances in relevance_lists:
        if k is None:
            ap = average_precision_at_k(relevances, len(relevances))
        else:
            ap = average_precision_at_k(relevances, k)
        average_precisions.append(ap)
    
    return np.mean(average_precisions) if average_precisions else 0.0


# Example usage
if __name__ == "__main__":
    print("=== Ranking Metrics Examples ===\n")
    
    # Example 1: NDCG with graded relevance
    print("1. NDCG@5 Example:")
    relevances = [3, 2, 3, 0, 1, 2, 1]  # Graded relevance scores
    ndcg_score = ndcg_at_k(relevances, k=5)
    print(f"   Relevance scores: {relevances[:5]}")
    print(f"   NDCG@5: {ndcg_score:.4f}\n")
    
    # Example 2: MRR
    print("2. MRR Example:")
    query_results = [
        [0, 0, 1, 0, 1],  # First relevant at position 3 (MRR = 1/3)
        [1, 0, 0, 0],      # First relevant at position 1 (MRR = 1/1)
        [0, 0, 0, 1],      # First relevant at position 4 (MRR = 1/4)
    ]
    mrr_score = mean_reciprocal_rank(query_results)
    print(f"   Query results: {query_results}")
    print(f"   MRR: {mrr_score:.4f}\n")
    
    # Example 3: MAP
    print("3. MAP Example:")
    relevance_lists = [
        [1, 0, 1, 0, 1],  # AP = (1/1 + 2/3 + 3/5) / 3
        [0, 1, 1, 0],      # AP = (1/2 + 2/3) / 2
        [1, 1, 0, 1],      # AP = (1/1 + 2/2 + 3/4) / 3
    ]
    map_score = mean_average_precision(relevance_lists)
    print(f"   Relevance lists: {relevance_lists}")
    print(f"   MAP: {map_score:.4f}\n")
    
    print("✓ All metrics calculated successfully!")

