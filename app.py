import streamlit as st
from data_loader import DocumentLoader
from embedding_generator import EmbeddingGenerator
from llm_client import LLMClient
import os
import tempfile

current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the embeddings folder
embedding_folder = os.path.join(current_dir, "embeddings")
os.makedirs(embedding_folder, exist_ok=True)

# Initialize components
processor = DocumentLoader()
embedor = EmbeddingGenerator()
llm = LLMClient()

st.title("🧠 RAG-based Medical Chatbot")

# Upload and embed documents
with st.expander("📄 Upload Document to Generate Embeddings"):
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file and st.button("🔄 Convert to Embeddings"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_pdf_path = tmp_file.name
        try:
            chunks = processor.load_and_process(tmp_pdf_path)
            embedor.generate_embeddings(chunks, save_path=embedding_folder)
            st.success("✅ Embeddings generated and saved successfully!")
        finally:
            os.remove(tmp_pdf_path)

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load vector store (once)
try:
    vector_store = embedor.load_vector_store(embedding_folder)
except Exception as e:
    st.error(f"❌ Error loading vector store: {e}")
    st.stop()

# Show existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("💬 Describe your symptoms or ask a question")

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from vector search + LLM
    similar_ones = vector_store.similarity_search(prompt, k=5)
    context = "\n\n".join([doc.page_content for doc in similar_ones])
    answer = llm.generate_output(context, prompt)

    # Show bot response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
