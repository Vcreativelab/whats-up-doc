# 🩺 What’s Up Doc? — AI Medical Assistant

**What’s Up Doc?** is a multilingual, evidence-based medical assistant built with **Streamlit**, **LangChain**, and **Google Gemini**.  
It can understand medical questions in multiple languages, translate them to English, route them intelligently, generate or summarise medical answers, and translate responses back to the user’s language — all with caching, rate limiting, and safety disclaimers.

## 🎯 Project Goals

This project explores how to build **safe, grounded medical AI assistants**
using retrieval-augmented generation (RAG), multilingual translation,
and strict evidence summarisation.

Key design principles:

- Reduce hallucinations using grounded medical sources
- Maintain transparency through citations
- Provide clear, structured explanations
- Support multilingual medical queries

> ⚠️ This app is for educational purposes only and does not replace professional medical advice.
---

## 🔗 Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://whats-up-doc-beta.streamlit.app/)

---

## ✨ Features

### 🌍 Multilingual support
- Automatic language detection  
- Translates non-English queries to English  
- Translates answers back to the original language  

### 🧠 Medical reasoning with LLMs
- Uses Google Gemini models via LangChain  
- Structured, evidence-based medical answers  
- Clean Markdown formatting with headings and sections  

### 🔀 Smart routing
Decides between:
- Direct LLM response, or  
- Summarised response from verified medical sources

### 🔎 Medical Evidence Retrieval Engine

When external information is needed, the system performs a grounded
medical search pipeline:

- Trusted medical domain filtering (CDC, NIH, Mayo Clinic, etc.)
- Parallel domain search
- Paragraph-level extraction from medical articles
- Semantic reranking using sentence embeddings
- Evidence fusion across multiple sources
- Basic contradiction detection
- Safety firewall preventing weak-evidence answers

### 📚 Summarisation pipeline
- Summarises multi-source medical content  
- Deduplicates and cleans formatting  
- Enforces a single, consistent disclaimer  

### 💾 Caching
- Translation cache  
- Back-translation cache  
- Reduces latency and API costs  

### 🚦 Rate limiting
- Simple token-based limiter to prevent abuse and runaway costs  

### 🧵 Short-term memory
- Keeps conversation context for better follow-up answers  

### 🖥️ Streamlit UI
- Easy API key input (or Streamlit Secrets)  
- Debug mode toggle  
- Clean, interactive interface  

---

## 🏗️ Architecture (High Level)
```
User Query  
↓  
Language Detection & Translation  
↓  
Short-term Memory Injection  
↓  
Router (Search vs Direct Answer)  
↓  

If search is required:

Search Engine  
↓  
Trusted-domain retrieval  
↓  
Paragraph extraction  
↓  
Semantic reranking  
↓  
Evidence fusion  
↓  
Medical summariser  
↓  

Otherwise:

Direct Gemini medical response  

↓  
Response cleanup  
↓  
Back-translation (if needed)  
↓  
Final Answer
```

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **LLM Orchestration:** LangChain  
- **Models:** Google Gemini (`gemini-2.5-flash-lite`, `gemini-2.5-flash`)  
- **Search:** DuckDuckGo  
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)  
- **Web Extraction:** BeautifulSoup  
- **Language:** Python  
- **Caching:** Disk-based caching (diskcache)  
- **Rate Limiting:** Custom token-based limiter  

---

## 📂 Project Structure
```
.
├── core/
│ ├── config.py # App config, constants, API key handling
│ ├── rate_limiter.py # Token-based rate limiting
│ ├── cache_manager.py # Translation caches
│ └── memory_manager.py # Conversation memory
│
├── services/ 
│ ├── medical_agent.py # Main orchestration logic
│ ├── search_engine.py # Medical evidence retrieval engine
│ ├── translator.py # Language detection & translation
│ ├── summariser.py # Medical summarisation logic
│ └── router.py # Routing logic (search vs direct)
│
├── utils/
│ └── formatting.py # Response cleanup & formatting
│
├── app.py # Streamlit entry point
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repo

```
git clone https://github.com/your-username/whats-up-doc.git
cd whats-up-doc
```
### 2. Create a virtual environment (recommended)
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install dependencies
```
pip install -r requirements.txt
```
### 4. Get a Google Gemini API key
- Go to: https://ai.google.dev/

- Create or select a project

- Generate an API key

### 5. Run the app
```
streamlit run app.py
```

---
## 🔐 API Key Configuration
You have two options:

### Option A — Streamlit Secrets (recommended for deployment)
Create .streamlit/secrets.toml:
```
GOOGLE_API_KEY = "your_api_key_here"
```
### Option B — Enter in the UI
- Disable “**Use Streamlit Secrets for API Key**” in the sidebar

- Paste your Gemini API key into the input field
---

## 🚀 Usage
**1.** Enter a medical question (in any language).

**2.** The app will:
- Detect the language

- Translate to English (if needed)

- Generate or summarise a medical answer

- Translate back to your language (if needed)
  
**3.** Read the structured, Markdown-formatted response.

---

## 🛡️ Safety & Disclaimer
This application:

  - Is not a medical device
  
  - Does not provide diagnoses
  
  - Does not replace professional medical advice

**All responses are for educational purposes only.
Always consult a qualified healthcare professional for medical concerns.**
