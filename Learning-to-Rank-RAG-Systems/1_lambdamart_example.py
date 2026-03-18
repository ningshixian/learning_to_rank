"""
LambdaMART Example: Learning to Rank with LightGBM
This demonstrates how to train a ranking model using LambdaMART objective.
"""

import numpy as np
from lightgbm import LGBMRanker
import pandas as pd

def generate_sample_data():
    """Generate sample query-document pairs for training."""
    np.random.seed(42)
    
    # Simulate 3 queries with different numbers of documents
    queries = []
    documents = []
    relevance_scores = []
    query_ids = []
    
    # Query 1: 10 documents
    for doc_idx in range(10):
        query_ids.append(0)
        queries.append("machine learning algorithms")
        documents.append(f"Document about ML algorithm {doc_idx}")
        # Higher relevance for first few docs
        relevance_scores.append(3 if doc_idx < 3 else 1 if doc_idx < 6 else 0)
    
    # Query 2: 15 documents
    for doc_idx in range(15):
        query_ids.append(1)
        queries.append("deep learning neural networks")
        documents.append(f"Document about deep learning {doc_idx}")
        relevance_scores.append(3 if doc_idx < 5 else 2 if doc_idx < 10 else 0)
    
    # Query 3: 8 documents
    for doc_idx in range(8):
        query_ids.append(2)
        queries.append("natural language processing")
        documents.append(f"Document about NLP {doc_idx}")
        relevance_scores.append(3 if doc_idx < 2 else 1 if doc_idx < 5 else 0)
    
    return pd.DataFrame({
        'query_id': query_ids,
        'query': queries,
        'document': documents,
        'relevance': relevance_scores
    })

def extract_features(df):
    """Extract simple features from query-document pairs."""
    features = []
    for _, row in df.iterrows():
        query = row['query'].lower()
        doc = row['document'].lower()
        
        # Simple feature extraction
        feature_vector = [
            len(query.split()),  # Query length
            len(doc.split()),    # Document length
            len(set(query.split()) & set(doc.split())),  # Common words
            query.count('learning'),  # Keyword presence
            doc.count('learning'),
        ]
        features.append(feature_vector)
    
    return np.array(features)

def train_lambdamart_ranker():
    """Train a LambdaMART ranker model."""
    
    # Generate sample data
    train_df = generate_sample_data()
    
    # Extract features
    X_train = extract_features(train_df)
    y_train = train_df['relevance'].values
    
    # Group parameter: number of documents per query
    query_doc_counts = train_df.groupby('query_id').size().values
    
    # Initialize LambdaMART ranker
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100,
        verbose=-1
    )
    
    # Train the model
    print("Training LambdaMART ranker...")
    ranker.fit(
        X_train, 
        y_train,
        group=query_doc_counts,  # Crucial: tells model which docs belong to which query
    )
    
    print("Training completed!")
    
    # Make predictions
    predictions = ranker.predict(X_train)
    train_df['predicted_score'] = predictions
    
    # Show results grouped by query
    for query_id in train_df['query_id'].unique():
        print(f"\nQuery {query_id}: {train_df[train_df['query_id'] == query_id]['query'].iloc[0]}")
        query_results = train_df[train_df['query_id'] == query_id].sort_values(
            'predicted_score', ascending=False
        )
        print(query_results[['document', 'relevance', 'predicted_score']].head())
    
    return ranker, train_df

if __name__ == "__main__":
    ranker, results = train_lambdamart_ranker()
    print("\n✓ LambdaMART ranker trained successfully!")

