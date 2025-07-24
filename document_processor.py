import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import Config, set_environment

class Document_Processor:
    def __init__(self, embeddings = None):
        self.embeddings = embeddings or Config.getembeddings()
        self.text_splitter = SemanticChunker(Config.getembeddings())

    def process_pdf(self, uploaded_file):
        temp_path = Config.get_temp_pdf_path()

        with open (temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        #load and process documents
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        #Split documents
        chunks = self.text_splitter.split_documents(documents)

        #Create vector store
        vectorstoredb = Chroma.from_documents(chunks, self.embeddings)

        #Return retriever
        return vectorstoredb.as_retriever()
    
    def cleanup_temp_file(self):
        temp_path = Config.get_temp_pdf_path()
        if os.path.exists(temp_path):
            os.remove(temp_path)




