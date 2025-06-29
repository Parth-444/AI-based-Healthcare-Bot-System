# 🧠 RAG-Based Medical Chatbot

This is an AI-powered medical chatbot that uses **Retrieval-Augmented Generation (RAG)** to answer health-related questions using your uploaded medical documents.

---

## 📌 Key Features

- ✅ **Symptom Checker:** Describe your symptoms and get possible causes matched from trusted documents.
- ✅ **Treatment Suggestions:** If available, the bot provides treatments or cures mentioned in the uploaded files.
- ✅ **Source Reference:** Each answer shows where the information came from (PDF name or section).
- ✅ **Multi-Document Support:** Upload multiple PDFs — the bot merges them into a single searchable knowledge base.
- ✅ **Chat Memory:** The chatbot remembers your previous questions for a natural conversation flow.

---

## 🚀 How It Works

1. **Upload PDF:** Add your own medical guides, handbooks, or reports.
2. **Generate Embeddings:** The app converts documents into searchable embeddings.
3. **Ask Questions:** Type your symptoms or questions in the chat.
4. **Get Answers:** The bot searches your docs, combines the info, and replies with clear, friendly suggestions.

---

## ⚙️ Tech Stack

- **Python**
- **Streamlit** — simple chat interface.
- **LangChain** — vector store (FAISS) & embedding pipeline.
- **Google Generative AI Embeddings**
- **Custom LLM Client** — your prompt templates & output handling.

---

## 📂 Project Structure (Example)

├─ app.py # Streamlit app
├─ data_loader.py # Loads and splits PDFs
├─ embedding_generator.py # Generates/updates embeddings
├─ llm_client.py # Handles LLM calls
├─ embeddings/ # Folder where FAISS index is saved
└─ README.md # This file


---

## ✅ How to Run

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt

2. **Set you LLM API key**
```bash
export GOOGLE_API_KEY=your_api_key_here
```

3. **Run the app**
 ```bash
streamlit run app.py
```
