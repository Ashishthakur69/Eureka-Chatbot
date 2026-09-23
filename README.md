---
title: Eureka Chatbot
emoji: 💡
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# 💡 Eureka Chatbot

> **An AI assistant with document intelligence, RAG-based document question answering, web search, and conversational memory.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![LangChain](https://img.shields.io/badge/LangChain-LLM%20Orchestration-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0467DF)](https://github.com/facebookresearch/faiss)
[![Hugging%20Face](https://img.shields.io/badge/Hugging%20Face-Embeddings-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Groq](https://img.shields.io/badge/Groq-LLM%20API-F55036)](https://groq.com/)

## 🎯 Overview

**Eureka** is a full-stack AI chatbot built to combine everyday conversational AI with **document intelligence**.

Users can chat naturally, upload **PDF or DOCX documents**, ask questions about uploaded content, and request information that may require a web search. For document questions, Eureka uses a retrieval-augmented generation (**RAG**) pipeline so responses are grounded in retrieved document content and accompanied by source/page references when available.

For current or time-sensitive questions, Eureka can use **DuckDuckGo web search** and provide the retrieved information to the language model before generating a response.

## ✨ Key Features

### 🤖 AI Chat
- Natural conversational interaction powered by Groq.
- Uses the <code>openai/gpt-oss-120b</code> model.
- Maintains recent conversation history for contextual responses.

### 📄 Document Intelligence
- Upload **PDF** and **DOCX** files.
- Supports up to **5 documents** in a session.
- Maximum upload request size: **25 MB**.
- Extracts document text, splits it into overlapping chunks, and indexes the chunks for retrieval.

### 🔎 RAG-Based Q&A
- Uses <code>RecursiveCharacterTextSplitter</code>.
- Chunk size: **1000 characters**.
- Chunk overlap: **200 characters**.
- Generates embeddings with <code>sentence-transformers/all-MiniLM-L6-v2</code>.
- Stores embeddings in a **FAISS** vector index.
- Retrieves the **top 5 relevant chunks** for document questions.
- Adds document and page references through an automatic **Sources** section.

### 🌐 Web Search
Eureka detects common signals for current or time-sensitive questions such as:
- latest or recent information
- news
- live information and scores
- weather and forecasts
- stock or share prices
- current-year queries

When triggered, Eureka queries **DuckDuckGo** and supplies the returned search context to the LLM.

### 💬 Conversation Memory
- Keeps recent chat messages in memory.
- Trims history to the latest **10 messages**.

### 🎨 Modern Web Interface
- Responsive chat interface.
- Dark/light theme support.
- Document upload and document management panel.
- Markdown rendering for AI responses.
- Copy-response functionality.
- Online status indicator.
- Responsive layouts for smaller screens.

## 🧠 How Eureka Works

### General Chat

~~~text
User Question
      ↓
Question Analysis
      ↓
Current / Time-Sensitive?
   ↙              ↘
 Yes               No
  ↓                 ↓
DuckDuckGo       Groq LLM
Search               ↓
  ↓              Response
Search Context
  ↓
Groq LLM
  ↓
Response
~~~

### Document Q&A

~~~text
PDF / DOCX Upload
        ↓
Text Extraction
        ↓
Document Chunking
        ↓
Hugging Face Embeddings
        ↓
FAISS Vector Index
        ↓
User Question
        ↓
Similarity Search (Top 5)
        ↓
Retrieved Context
        ↓
Groq LLM
        ↓
Answer + Source References
~~~

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| LLM | Groq — <code>openai/gpt-oss-120b</code> |
| LLM Framework | LangChain |
| Document Loaders | PyPDFLoader, Docx2txtLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| Embeddings | Hugging Face <code>all-MiniLM-L6-v2</code> |
| Vector Store | FAISS |
| Web Search | DuckDuckGo |
| Configuration | python-dotenv |
| Cross-Origin Support | Flask-CORS |

## 📂 Project Structure

~~~text
Eureka-Chatbot/
├── static/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── app.py
├── requirements.txt
├── README.md
└── .gitattributes
~~~

## 🚀 Run Locally

### 1. Clone the repository

~~~bash
git clone https://github.com/Ashishthakur69/Eureka-Chatbot.git
cd Eureka-Chatbot
~~~

### 2. Create a virtual environment

**Windows:**

~~~powershell
python -m venv venv
venv\Scripts\activate
~~~

**macOS / Linux:**

~~~bash
python3 -m venv venv
source venv/bin/activate
~~~

### 3. Install dependencies

~~~bash
pip install -r requirements.txt
~~~

### 4. Configure the Groq API key

Create a local <code>.env</code> file in the project root:

~~~env
GROQ_API_KEY=your_groq_api_key_here
~~~

> **Security:** Never commit <code>.env</code> or API keys to GitHub. Use environment variables or your deployment platform's secret manager.

### 5. Start Eureka

~~~bash
python app.py
~~~

Open the local address shown in the terminal.

## 📄 Supported Documents

| File Type | Supported |
|---|---:|
| PDF | ✅ |
| DOCX | ✅ |
| Other formats | ❌ |

**Limits:** up to 5 documents per session and a maximum upload request size of 25 MB.

## 💬 Example Use Cases

**General AI**
> Explain gradient descent in simple terms.

**Document Q&A**
> Upload a PDF and ask: “What are the main conclusions of this document?”

**Document comparison**
> Upload several documents and ask: “What differences appear between these reports?”

**Current information**
> Ask a time-sensitive question such as a latest-news or current-weather query.

## 🔐 Environment Variables

| Variable | Required | Description |
|---|---:|---|
| <code>GROQ_API_KEY</code> | ✅ | API key used to access the Groq language model |

## 🌐 Live Demo

Try Eureka on Hugging Face:

**[Eureka Chatbot — Hugging Face Space](https://huggingface.co/spaces/ashishthakur69/Eureka-Chatbot)**

Source code:

**[Eureka Chatbot — GitHub](https://github.com/Ashishthakur69/Eureka-Chatbot)**

## 📌 Architecture Notes

- Document embeddings and the FAISS index are maintained in application memory.
- Uploaded files are processed and removed from the upload directory after processing.
- Recent conversation messages are maintained in memory.
- Web search is triggered selectively for recognized current or time-sensitive signals.
- The current implementation is primarily designed as a working AI application/demo rather than a multi-user production backend with persistent storage and authentication.

## 🔮 Future Improvements

- Multi-user session management and authentication.
- Persistent document/vector storage.
- Streaming token generation with richer status updates.
- Hybrid semantic + keyword retrieval.
- Reranking and retrieval evaluation.
- Better source attribution for web results.
- Conversation persistence.
- Production-grade observability and monitoring.
- Containerized deployment and automated CI/CD.

## 👨‍💻 Author

**Ashish Thakur**

BCA | AI & Data Science Enthusiast

**GitHub:** [Ashishthakur69](https://github.com/Ashishthakur69)

---

### ⭐ Eureka

**Ask. Explore. Discover. 💡**
