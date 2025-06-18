#load the pdf file
#convert the data into chunks
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embedding_generator import EmbeddingGenerator
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
class DocumentLoader:



    @staticmethod
    def load_and_process(path, chunk_size=1000, chunk_overlap=200):
        loader=PyPDFLoader(path)
        document = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunk_doc = splitter.split_documents(document)
        return chunk_doc


if __name__ == '__main__':
    doc_loader = DocumentLoader()
    main_doc = doc_loader.load_and_process("pragati.pdf")
    generator = EmbeddingGenerator()
    embeddings = generator.generate_embeddings(main_doc, current_dir)
    print(embeddings)
