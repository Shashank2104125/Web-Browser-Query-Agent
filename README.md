# 🔍 Web Browser Query Agent

A smart, AI-powered web query agent, this tool validates user queries, checks for past similar results, fetches and scrapes web content, summarizes it using LLMs, and caches it for future use.

---

## 📌 Features

- ✅ **Query Validation**: Classifies if the input is a valid informational query using an LLM.
- 🔁 **Similarity Detection**: Uses vector embeddings + Pinecone DB to detect and reuse previous results.
- 🌐 **Live Web Search**: Searches the web using Playwright and DuckDuckGo for fresh results.
- 🧠 **Summarization**: Summarizes top web results using Mistral-7B-Instruct via Hugging Face.
- 💾 **Caching**: Embeds and stores query-answer pairs for fast retrieval.
- 🖥️ **Frontend Web App**: Clean and simple interface using Streamlit.
- 🧪 **CLI Interface**: Command-line mode for quick testing.

---

## 📊 Architecture Diagram

        User Input
        ↓
        Query Classifier ───────────────→ [Invalid] → "This is not a valid query."
        ↓
        [Valid]
        ↓
        Convert Query to Embedding
        ↓
        Search in Vector DB (e.g., Pinecone, FAISS, Chroma)
        ↓                           ↓
        [Match Found?]           [No Match]
        ↓                           ↓
        Return cached result        →  Use Playwright to search web
                                    ↓
                                    Scrape Top 5 results (BeautifulSoup/Playwright)
                                    ↓
                                    Summarize each page (LLM API or local model)
                                    ↓
                                    Save summary + embeddings in Vector DB + Doc Store
                                    ↓
                                    Return to user

---

## 🚀 Setup Instructions

1. Clone the Repository
git clone https://github.com/your-username/web-query-agent.git
cd web-query-agent

2. Create a .env File
GOOGLE_API_KEY=your_google_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key

3. Install Requirements
pip install -r requirements.txt
playwright install

4. Usage
Run via CLI: python main.py
Run Streamlit App: streamlit run web_qa_app.py


