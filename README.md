# Bible RAG: Retrieval-Augmented Generation QA System

This project implements a Retrieval-Augmented Generation (RAG) system on the Bible corpus (KJV version) to enable open-domain question answering. It was developed as part of a final course project in Text Analytics.

## 🧠 Overview

The system uses various retrieval methods (BM25, MMR, Semantic Search, and Hybrid RRF) along with multiple chunking strategies and embedding models to extract the most relevant biblical passages. These are then passed to lightweight open-access LLMs (Qwen2.5 1.5B and 3B) to generate final answers.

## 🔧 Technologies & Tools
- Python 3
- Google Colab / Kaggle (for GPU)
- Sentence-Transformers & Hugging Face Transformers
- LangChain
- Scikit-learn, Pandas, NumPy

## 📚 Corpus
- King James Bible (`.txt`) from OpenBible.com

## ⚙️ Retrieval Techniques
- BM25
- MMR (Maximal Marginal Relevance)
- Semantic Vector Search (using BAAI and MiniLM embeddings)
- Hybrid RRF (Reciprocal Rank Fusion)

## 🧪 Evaluation
Responses were evaluated on:
- **Faithfulness** (similarity to retrieved context)
- **Relevance** (similarity to the question)
- **Similarity** (to ground truth answers)

Cosine similarity metrics were computed using sentence embeddings. Top-performing combination:  
**Qwen2.5 3B + MMR Retriever + BAAI/bge-small-en embeddings + Chunk size: 256**

## 🚀 How to Run
1. Open the notebook in Google Colab or Kaggle
2. Install required packages (`transformers`, `sentence-transformers`, etc.)
3. Run each section to load corpus, build vector DB, retrieve contexts, and generate responses

## 📈 Results
Detailed metrics and output samples are included in the final report and CSV logs in this repo.

## 🧑‍💻 Contributors
- Sahil Kumar  
- Zuhair Farhan
