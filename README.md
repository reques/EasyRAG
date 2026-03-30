# 🚀 EasyRAG

> A lightweight, high-performance Retrieval-Augmented Generation (RAG) system with optimized retrieval strategies and extensible Agent capabilities.

---

## 📖 Overview

EasyRAG is a practical, production-oriented RAG framework designed for real-world applications.  
It focuses on **high-precision retrieval**, **scalable Agent integration**, and **engineering usability**, providing a complete pipeline from data ingestion to answer generation.

---

## 🧱 Architecture
Data Processing → Vector Indexing → Retrieval Enhancement → LLM Generation

- End-to-end RAG pipeline
- Modular design for easy extension and customization

---

## 🔧 Features

- 🔍 **High-Precision Retrieval**
  - Milvus vector database + BGE embedding models  
  - Semantic chunking + retrieval optimization strategies  

- 🧠 **Agent-Ready Design**
  - Supports Function Call for tool invocation  
  - Easily extensible to multi-step Agent workflows  

- 🧩 **Structured Retrieval Output**
  - Returns structured results (chunk / doc / metadata)  
  - Improves interpretability and debugging  

- ⚙️ **Multi-Strategy Retrieval**
  - Top-K retrieval  
  - Filtering & conditional search  
  - Enhanced robustness for complex queries  

- 📡 **Production-Ready API**
  - FastAPI-based service  
  - Streaming response support  

- 📄 **Multi-Source Data Support**
  - OCR integration  
  - PDF / Markdown / multi-modal document parsing  

---

## ⚙️ Core Capabilities

- **Retrieval Optimization**: Semantic embedding + filtering strategies  
- **Agent Integration**: Function Call-based tool usage  
- **Flexible Data Ingestion**: Multi-format knowledge sources  
- **Engineering Deployment**: API service + Docker support  
- **Debugging & Observability**: Intermediate result tracking  

---

## 📈 Performance

- Improved **Precision@K** through chunking and retrieval strategy optimization  
- Enhanced answer relevance and stability in complex queries  
- Supports continuous optimization (retrieval / prompt / tool usage)

---

## 🛠 Tech Stack

- **LLM**: OpenAI / DeepSeek / compatible APIs  
- **Embedding**: BGE (BAAI)  
- **Vector DB**: Milvus  
- **Backend**: FastAPI  
- **Frameworks**: LangChain / LlamaIndex (optional)  
- **Deployment**: Docker  

---

## 📦 Project Structure
EasyRAG/
├── app/ # Core logic
├── models/ # (Not included, external download)
├── api/ # FastAPI service(coming soon)
├── scripts/ # Utility scripts(coming soon)
├── README.md

---

## 📥 Model & Data

> ⚠️ Model files are not included in this repository.

Download pretrained models from:

- HuggingFace / external storage (to be added)
- Place them under:
/models/

---

## 🚀 Quick Start

```bash
# 1. Clone repo
git clone https://github.com/reques/EasyRAG.git

# 2. Install dependencies
pip install -r requirements.txt

#install dicker environment
docker compose up -d

# 3. Run service
python run.py

🔮 Future Work
Multi-Agent workflow integration (LangGraph)
Advanced retrieval strategies (hybrid / rerank)
Knowledge graph augmentation
Evaluation pipeline

⭐ If you find this project useful

Give it a star ⭐ and feel free to contribute!
