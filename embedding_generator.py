from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
key = os.getenv("GOOGLE_API_KEY")
class EmbeddingGenerator:

    def __init__(self):
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key= key
        )

    def generate_embeddings(self, chunks, save_path):
        # Create vector store from chunks
        vector_store = FAISS.from_documents(chunks, self.embedding_model)
        vector_store.save_local(save_path)
        return save_path

    def load_vector_store(self, load_path):
        return FAISS.load_local(folder_path=load_path, embeddings=self.embedding_model,
                                allow_dangerous_deserialization=True)

