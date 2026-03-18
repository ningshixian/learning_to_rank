# Learning to Rank for RAG Systems

This repository contains comprehensive code examples and implementations for Learning to Rank (LTR) techniques, specifically focused on enhancing Retrieval-Augmented Generation (RAG) systems.

## 📁 Repository Structure

```
Learning-to-Rank-RAG-Systems/
├── 1_lambdamart_example.py      # LambdaMART implementation with LightGBM
├── 2_ndcg_metrics.py            # Ranking evaluation metrics (NDCG, MRR, MAP)
├── 3_reranking_example.py       # Cross-encoder re-ranking example
├── 4_rag_pipeline.py            # Complete two-stage RAG pipeline
├── 5_streamlit_app.py           # Interactive Streamlit demo application
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation

1. Clone this repository:
```bash
git clone https://github.com/MahendraMedapati27/Learning-to-Rank-RAG-Systems.git
cd Learning-to-Rank-RAG-Systems
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Examples

#### 1. LambdaMART Example
```bash
python 1_lambdamart_example.py
```
Demonstrates training a ranking model using LambdaMART objective.

#### 2. Evaluation Metrics
```bash
python 2_ndcg_metrics.py
```
Shows how to calculate NDCG, MRR, and MAP metrics.

#### 3. Re-ranking Example
```bash
python 3_reranking_example.py
```
Simple example of re-ranking documents using cross-encoders.

#### 4. Complete RAG Pipeline
```bash
python 4_rag_pipeline.py
```
Demonstrates a full two-stage retrieval system.

#### 5. Interactive Streamlit App
```bash
streamlit run 5_streamlit_app.py
```
Launch an interactive web application to experiment with ranking.

## 📚 Key Concepts

### Two-Stage Retrieval

1. **Stage 1: Fast Retrieval (Bi-Encoder)**
   - Pre-compute document embeddings
   - Fast similarity search
   - Retrieves top-k candidates

2. **Stage 2: Accurate Re-ranking (Cross-Encoder)**
   - Process query-document pairs
   - Fine-grained relevance scoring
   - Re-rank to final top-n results

### Ranking Metrics

- **NDCG**: Normalized Discounted Cumulative Gain - considers position and graded relevance
- **MRR**: Mean Reciprocal Rank - focuses on first relevant result
- **MAP**: Mean Average Precision - averages precision across recall levels

## 🔧 Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## 📖 Usage Examples

See individual Python files for detailed code examples and usage instructions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 🔗 Let's Connect & Collaborate!

I'm passionate about sharing knowledge and building amazing AI solutions. Let's connect:

### 📱 Social Media & Professional Links

- 🐙 **GitHub**: [@MahendraMedapati27](https://github.com/MahendraMedapati27) - Check out my latest projects and code repositories
- 💼 **LinkedIn**: [Mahendra Medapati](https://www.linkedin.com/in/mahendra-medapati-429239289/) - Connect for professional discussions and industry insights
- 🐦 **X (Twitter)**: [@MahendraM27](https://x.com/MahendraM27) - Follow for updates, thoughts, and discussions on AI
- 📧 **Email**: [mahendramedapati.r469@gmail.com](mailto:mahendramedapati.r469@gmail.com) - Reach out directly for inquiries or collaboration

### ☕ Support This Work

If this guide helped you, consider supporting my work:

☕ **[Buy me a coffee](https://buymeacoffee.com/mahendramedapati)** — Your support helps me create more comprehensive guides like this!

---

## 📄 License

This project is open source and available under the MIT License.

