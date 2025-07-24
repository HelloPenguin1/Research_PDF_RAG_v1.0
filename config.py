import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st

def set_environment():
    secret_keys = [
        "OPENAI_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT",
        "HF_TOKEN",
        "GROQ_API_KEY"
    ]
    for key in secret_keys:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]

class Config:
    @staticmethod
    def getembeddings():
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    @staticmethod   
    def get_temp_pdf_path():
       return "./temp.pdf"
    
       
    
   
   
      
