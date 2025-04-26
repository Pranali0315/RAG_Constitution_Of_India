# 📜 ConstitutionRAG

**ConstitutionRAG** is an intelligent AI-powered system designed to provide context-aware answers to questions about the **Indian Constitution**. Using **Retrieval-Augmented Generation (RAG)**, it fetches relevant constitutional articles, parts, and schedules, and generates accurate responses using powerful language models.

## 🚀 Project Overview

This project combines vector search and generative AI to build a conversational assistant over the Indian Constitution. It helps users — students, legal researchers, and enthusiasts — understand constitutional references easily.

### Key Features

- **Ask Anything About the Constitution**: Get answers grounded in actual articles and sections.
- **Contextual Retrieval**: Uses vector embeddings to fetch the most relevant constitutional clauses.
- **Generative Responses**: Uses an LLM to generate human-like answers based on retrieved content.
- **Fast and Efficient**: In-memory processing and retrieval ensures quick responses.

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **LLM**: Open-source LLMs (e.g., gemma-2-2b-q5_k_m.gguf)
- **Retrieval Engine**: Qdrant
- **Embeddings**: Sentence Transformers (e.g., `all-MiniLM`)
- **Document Parsing**: Text + Chunking + Preprocessing

## 📝 Getting Started

### Prerequisites

- Python 3.8+
- `pip` for package installation

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rushi15003/RAG_Constitution_of_India.git
   cd RAG Constitution

2. **Create Virtual Environment**:
   ```bash
   python -m venv my_venv
   source my_venv/bin/activate

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   
**Note** : Run all the python files located in modules directory one by one(scraping.py->chunking.py->embedding.py->qdrant_init.py->qdrant_upload.py)  

4. **Run the Application**:
   ```bash
   python app.py

## 🎉 Usage
- **Ask Your Question**: Enter any constitutional query (e.g., "What is Article 21?").
- **Contextual Search**: The system retrieves relevant chunks from the Indian Constitution.
- **Answer Generation**: An LLM answers using the retrieved context.

## 🗂️Dataset Notes
- Data sourced from the Indian Constitution (text format).
- Preprocessing includes chunking by Articles/Parts.
- All content is embedded using Sentence Transformers for vector similarity.
     
