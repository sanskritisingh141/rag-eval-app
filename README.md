# RAG-Eval: Production Retrieval-Augmented Generation System with Evaluation Dashboard

A production-ready Retrieval-Augmented Generation (RAG) application that enables users to upload documents, ask context-aware questions, inspect source citations, and evaluate response quality using both lightweight local metrics and advanced RAGAS-based evaluation.

Built with Streamlit, LangChain, FAISS, Hugging Face embeddings, and Google Gemini, the project demonstrates an end-to-end RAG pipeline with integrated answer evaluation and explainability features.

---

## Features

### Document Processing

* Upload PDF, DOCX, and TXT files
* Automatic text extraction and preprocessing
* Intelligent document chunking for retrieval

### Retrieval-Augmented Generation

* Semantic search using FAISS vector indexing
* Hugging Face Sentence Transformer embeddings
* Context-aware answer generation using Gemini
* Grounded responses based on retrieved document content
* Source citations with supporting chunks

### Evaluation Dashboard

* Lightweight local evaluation metrics
* Optional RAGAS evaluation
* Question and evaluation history
* Quota-aware evaluation workflow

### User Experience

* Interactive Streamlit interface
* Expandable source citation viewer
* Graceful API quota error handling
* Deployment-ready architecture

---

## System Architecture

```text
User Uploads Document
          │
          ▼
   Text Extraction
          │
          ▼
    Text Chunking
          │
          ▼
Sentence Embeddings
          │
          ▼
  FAISS Vector Store
          │
          ▼
 Similarity Retrieval
          │
          ▼
 Retrieved Context
          │
          ▼
 Gemini Generation
          │
          ▼
 Answer + Citations
          │
          ▼
 Evaluation Dashboard
```

---

## Tech Stack

| Component        | Technology                         |
| ---------------- | ---------------------------------- |
| Frontend         | Streamlit                          |
| LLM              | Google Gemini                      |
| RAG Framework    | LangChain                          |
| Embeddings       | Hugging Face Sentence Transformers |
| Vector Store     | FAISS                              |
| Evaluation       | Local Metrics, RAGAS               |
| Document Parsing | PyPDF, python-docx                 |
| Deployment       | Hugging Face Spaces, Render        |

---

## Evaluation Framework

The application provides two complementary evaluation modes.

### Local Evaluation

The default evaluation mode uses lightweight, quota-free metrics designed for rapid feedback without requiring additional LLM API calls.

These metrics are based on sentence-transformer cosine similarity and lexical overlap heuristics.

| Metric             | Description                                                                              |
| ------------------ | ---------------------------------------------------------------------------------------- |
| Context Similarity | Cosine similarity between the generated answer embedding and retrieved context embedding |
| Answer Relevance   | Cosine similarity between the user question embedding and generated answer embedding     |
| Citation Coverage  | Lexical overlap between answer tokens and retrieved context tokens                       |
| Retrieval Strength | Ratio of non-empty retrieved chunks to total retrieved chunks                            |

#### Advantages

* No API usage
* Fast execution
* Consistent scoring
* Suitable for large-scale testing

### RAGAS Evaluation

For deeper analysis, the application supports optional RAGAS-based evaluation using LLM-as-judge techniques.

| Metric            | Description                                                             |
| ----------------- | ----------------------------------------------------------------------- |
| Faithfulness      | Measures whether the generated answer is supported by retrieved context |
| Answer Relevancy  | Evaluates how effectively the answer addresses the user's question      |
| Context Precision | Assesses whether retrieved chunks are relevant and useful               |

Because RAGAS relies on LLM-based evaluation, it may consume Gemini API quota and require additional processing time.

### Evaluation Workflow

```text
Question
    │
    ▼
Retrieved Context
    │
    ▼
Generated Answer
    │
    ├── Local Evaluation
    │      ├── Context Similarity
    │      ├── Answer Relevance
    │      ├── Citation Coverage
    │      └── Retrieval Strength
    │
    └── Optional RAGAS Evaluation
           ├── Faithfulness
           ├── Answer Relevancy
           └── Context Precision
```

---

## Project Structure

```text
rag-eval-app/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── src/
│   ├── loaders.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── rag_chain.py
│   ├── evaluator.py
│   └── local_evaluator.py
│
└── screenshots/
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-eval-app.git
cd rag-eval-app
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
```

The `.env` file should never be committed to version control.

---

## Running the Application

```bash
streamlit run app.py
```

The application will launch locally in your browser.

---

## Usage

### Step 1: Upload a Document

Supported formats:

* PDF
* DOCX
* TXT

### Step 2: Index the Document

The system extracts text, creates chunks, generates embeddings, and builds a FAISS index.

### Step 3: Ask Questions

Submit natural language questions related to the uploaded document.

### Step 4: Inspect Sources

Review retrieved chunks and supporting citations used to generate the answer.

### Step 5: Evaluate Responses

Choose between:

* Local Evaluation (fast and quota-free)
* RAGAS Evaluation (advanced LLM-based assessment)

---

## Example Queries

### Resume / CV Analysis

```text
What technical skills are mentioned?
What projects has the candidate worked on?
Does the candidate have experience in machine learning?
```

### Research Paper Analysis

```text
What is the primary objective of this paper?
What methodology is used?
What are the key findings?
```

### Hallucination Testing

```text
What is the author's favorite movie?
```

If information is not present in the document, the system is designed to avoid generating unsupported answers.

---

## Screenshots

### Document Upload

![Document Upload](screenshots/upload.png)

### Question Answering with Citations

![RAG Answer](screenshots/answer.png)

### Local Evaluation Dashboard

![Local Evaluation](screenshots/local_eval.png)

### RAGAS Evaluation Dashboard

![RAGAS Evaluation](screenshots/ragas_eval.png)

### Question History

![Question History](screenshots/history.png)

---

## Deployment

### Hugging Face Spaces

1. Create a new Space
2. Select **Streamlit** as the SDK
3. Connect or upload the repository
4. Add the following secret:

```text
GOOGLE_API_KEY
```

5. Deploy the application

Do not upload the `.env` file.

### Render

The application can also be deployed on Render using the same environment configuration.

---

## Requirements

```text
streamlit
langchain==0.2.17
langchain-community==0.2.17
langchain-core==0.2.43
langchain-google-genai==1.0.10
google-generativeai
faiss-cpu
pypdf
python-docx
python-dotenv
sentence-transformers
scikit-learn
ragas==0.1.21
datasets
pandas
```

---

## Limitations

* Gemini free-tier quotas may restrict heavy usage.
* FAISS indexes are stored in memory during active sessions.
* Optimized for single-document workflows.
* RAGAS evaluation can be slower than local evaluation.
* Very large documents may require additional preprocessing.

---

## Future Enhancements

* Multi-document retrieval
* Persistent vector storage
* Hybrid retrieval (semantic + keyword search)
* Retrieval reranking
* PDF page previews for citations
* Batch evaluation datasets
* Docker support
* FastAPI backend
* User authentication and session management

---

## License

This project is intended for educational, research, and portfolio purposes.

---

## Author

Sanskriti Singh

