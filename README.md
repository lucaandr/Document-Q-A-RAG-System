# Local RAG System with ChromaDB & Flan-T5

An end-to-end local Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB, and HuggingFace models.

## 📌 Features
- **Data Ingestion:** Web scraping 13 Wikipedia articles (AI/ML, Finance, Sports, Gastronomy) using BeautifulSoup.
- **Chunking Experiments:** Compared performance between `Small` (200), `Medium` (500), and `Large` (1000) chunk sizes.
- **Vector Database:** ChromaDB with `all-MiniLM-L6-v2` embeddings.
- **Local Generation:** `google/flan-t5-base` LLM for constrained Q&A.
- **Safety Evaluation:** Includes Out-of-Domain failure analysis and hallucination prevention tests.

