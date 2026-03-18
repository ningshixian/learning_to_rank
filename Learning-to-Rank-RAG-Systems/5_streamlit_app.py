"""
Streamlit Application: Interactive Learning to Rank Demo
A user-friendly interface to experiment with ranking and re-ranking.
"""

import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go


# Page configuration
st.set_page_config(
    page_title="Learning to Rank - RAG Systems Demo",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'bi_encoder' not in st.session_state:
    st.session_state.bi_encoder = None


@st.cache_resource
def load_models():
    """Load embedding models (cached for performance)."""
    with st.spinner("Loading models (this may take a moment on first run)..."):
        bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return bi_encoder, cross_encoder


def compute_ndcg(relevances, k=None):
    """Calculate NDCG@k."""
    if k is None:
        k = len(relevances)
    relevances = relevances[:k]
    
    def dcg(rels):
        if not rels:
            return 0.0
        return sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels))
    
    dcg_score = dcg(relevances)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg_score = dcg(ideal_relevances)
    
    return dcg_score / idcg_score if idcg_score > 0 else 0.0


def main():
    """Main application interface."""
    
    st.title("📊 Learning to Rank: Interactive RAG System Demo")
    st.markdown("""
    **Explore ranking algorithms and their impact on RAG (Retrieval-Augmented Generation) systems.**
    
    This demo showcases:
    - Bi-encoder retrieval (fast, scalable)
    - Cross-encoder re-ranking (accurate, detailed)
    - Two-stage retrieval pipeline
    - Ranking evaluation metrics (NDCG)
    """)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Load models
    bi_encoder, cross_encoder = load_models()
    st.session_state.bi_encoder = bi_encoder
    
    # Document management
    st.sidebar.subheader("📄 Document Management")
    
    if st.sidebar.button("Load Sample Documents"):
        sample_docs = [
            "Machine learning models can be optimized using quantization and pruning techniques for production deployment.",
            "Deep learning requires large datasets and powerful GPUs for training neural networks effectively.",
            "Re-ranking improves search quality by using cross-encoder models that capture query-document interactions.",
            "NDCG (Normalized Discounted Cumulative Gain) is the gold standard metric for evaluating ranking systems.",
            "Two-stage retrieval combines fast bi-encoder search with accurate cross-encoder re-ranking.",
            "BERT revolutionized NLP by introducing bidirectional attention mechanisms in transformer architectures.",
            "Vector databases enable efficient similarity search for embeddings using approximate nearest neighbor algorithms.",
            "LambdaMART uses gradient boosting trees with lambda gradients to optimize ranking objectives.",
        ]
        st.session_state.documents = sample_docs
        st.session_state.embeddings = bi_encoder.encode(sample_docs, show_progress_bar=False)
        st.sidebar.success(f"Loaded {len(sample_docs)} sample documents!")
    
    # Manual document input
    new_doc = st.sidebar.text_area("Add Document:", height=100)
    if st.sidebar.button("Add Document") and new_doc:
        st.session_state.documents.append(new_doc)
        if st.session_state.embeddings is not None:
            new_embedding = bi_encoder.encode([new_doc], show_progress_bar=False)
            st.session_state.embeddings = np.vstack([st.session_state.embeddings, new_embedding])
        else:
            st.session_state.embeddings = bi_encoder.encode([new_doc], show_progress_bar=False)
        st.sidebar.success("Document added!")
    
    if st.sidebar.button("Clear All Documents"):
        st.session_state.documents = []
        st.session_state.embeddings = None
        st.sidebar.info("Documents cleared!")
    
    # Display current documents
    if st.session_state.documents:
        st.sidebar.markdown(f"**Current Documents:** {len(st.session_state.documents)}")
    
    # Main content area
    if not st.session_state.documents:
        st.info("👈 Please load sample documents or add your own in the sidebar to begin.")
        return
    
    # Query input
    st.subheader("🔍 Query Interface")
    query = st.text_input("Enter your search query:", placeholder="e.g., How to optimize machine learning models?")
    
    if not query:
        st.info("Enter a query above to see ranking results.")
        return
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        initial_k = st.slider("Initial Retrieval (Stage 1)", min_value=3, max_value=min(20, len(st.session_state.documents)), value=10)
    with col2:
        final_k = st.slider("Final Results (After Re-ranking)", min_value=1, max_value=initial_k, value=5)
    
    # Stage 1: Bi-encoder retrieval
    st.subheader("📈 Stage 1: Bi-Encoder Retrieval")
    
    with st.spinner("Computing query embedding and similarities..."):
        query_embedding = bi_encoder.encode([query], show_progress_bar=False)
        similarities = cosine_similarity(query_embedding, st.session_state.embeddings)[0]
        
        # Get top-k candidates
        top_indices = np.argsort(similarities)[::-1][:initial_k]
        
        stage1_results = []
        for idx in top_indices:
            stage1_results.append({
                'rank': len(stage1_results) + 1,
                'document': st.session_state.documents[idx],
                'score': float(similarities[idx]),
                'index': int(idx)
            })
    
    # Display stage 1 results
    df_stage1 = pd.DataFrame(stage1_results)
    st.dataframe(df_stage1[['rank', 'score', 'document']], use_container_width=True, hide_index=True)
    
    # Stage 2: Cross-encoder re-ranking
    st.subheader("🎯 Stage 2: Cross-Encoder Re-Ranking")
    
    with st.spinner("Re-ranking with cross-encoder..."):
        # Prepare query-document pairs
        candidate_docs = [result['document'] for result in stage1_results]
        pairs = [[query, doc] for doc in candidate_docs]
        
        # Get cross-encoder scores
        cross_scores = cross_encoder.predict(pairs)
        
        # Update results
        for result, score in zip(stage1_results, cross_scores):
            result['cross_encoder_score'] = float(score)
        
        # Re-sort by cross-encoder score
        stage2_results = sorted(stage1_results, key=lambda x: x['cross_encoder_score'], reverse=True)[:final_k]
        
        # Update ranks
        for i, result in enumerate(stage2_results, 1):
            result['final_rank'] = i
    
    # Display stage 2 results
    df_stage2 = pd.DataFrame(stage2_results)
    if not df_stage2.empty:
        st.dataframe(df_stage2[['final_rank', 'cross_encoder_score', 'document']], use_container_width=True, hide_index=True)
        
        # Visualization
        st.subheader("📊 Ranking Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Stage 1 scores
            fig1 = px.bar(
                df_stage2.head(10),
                x='score',
                y='document',
                orientation='h',
                title='Stage 1: Bi-Encoder Scores',
                labels={'score': 'Similarity Score', 'document': 'Document'}
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Stage 2 scores
            fig2 = px.bar(
                df_stage2.head(10),
                x='cross_encoder_score',
                y='document',
                orientation='h',
                title='Stage 2: Cross-Encoder Scores',
                labels={'cross_encoder_score': 'Relevance Score', 'document': 'Document'}
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Ranking changes visualization
        if len(stage2_results) > 1:
            st.subheader("🔄 Ranking Position Changes")
            
            comparison_df = pd.DataFrame([
                {
                    'Document': doc[:50] + '...' if len(doc) > 50 else doc,
                    'Stage 1 Rank': result['rank'],
                    'Stage 2 Rank': result['final_rank'],
                    'Change': result['rank'] - result['final_rank']
                }
                for result, doc in zip(stage2_results, candidate_docs)
            ])
            
            fig3 = go.Figure()
            
            for _, row in comparison_df.iterrows():
                fig3.add_trace(go.Scatter(
                    x=['Stage 1', 'Stage 2'],
                    y=[row['Stage 1 Rank'], row['Stage 2 Rank']],
                    mode='lines+markers',
                    name=row['Document'],
                    line=dict(width=2)
                ))
            
            fig3.update_layout(
                title='Ranking Position Changes After Re-ranking',
                xaxis_title='Stage',
                yaxis_title='Rank (lower is better)',
                yaxis=dict(autorange='reversed'),
                height=500
            )
            
            st.plotly_chart(fig3, use_container_width=True)
    
    # Metrics
    st.subheader("📏 Evaluation Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Calculate NDCG (assuming relevance based on cross-encoder scores)
        relevances = [r['cross_encoder_score'] for r in stage2_results]
        ndcg = compute_ndcg(relevances, k=final_k)
        st.metric("NDCG@{}".format(final_k), f"{ndcg:.4f}")
    
    with col2:
        # Precision (assuming top results are relevant if score > threshold)
        threshold = np.mean(relevances)
        precision = sum(1 for r in relevances if r > threshold) / len(relevances) if relevances else 0
        st.metric("Precision@{}".format(final_k), f"{precision:.2%}")
    
    with col3:
        st.metric("Documents Retrieved", len(stage2_results))
    
    # Information section
    with st.expander("ℹ️ Learn More About This Demo"):
        st.markdown("""
        **Two-Stage Retrieval Explained:**
        
        1. **Stage 1 (Bi-Encoder):** Fast retrieval using pre-computed document embeddings.
           - Documents are encoded once and stored
           - Query is encoded and compared via cosine similarity
           - Scales to millions of documents in milliseconds
        
        2. **Stage 2 (Cross-Encoder):** Accurate re-ranking of top candidates.
           - Processes query-document pairs together
           - Captures fine-grained interactions
           - More accurate but slower (used only on top-k candidates)
        
        **Why This Matters:**
        - Bi-encoders are fast but may miss nuanced relevance
        - Cross-encoders are accurate but too slow for large-scale retrieval
        - Combining both gives you the best of both worlds!
        
        **NDCG (Normalized Discounted Cumulative Gain):**
        - Measures ranking quality considering position importance
        - Values closer to 1.0 indicate better rankings
        - Accounts for graded relevance (not just binary)
        """)


if __name__ == "__main__":
    main()

