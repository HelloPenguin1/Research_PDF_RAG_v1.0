import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from config import set_environment

set_environment()

class SessionManager:
    def __init__(self):
        # Initialize session store if not exists
        if 'store' not in st.session_state:
            st.session_state.store = {}
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create chat history for a session"""
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    


    def get_all_sessions(self):
        return st.session_state.store




    def clear_session(self, session_id: str):
        if session_id in st.session_state.store:
            del st.session_state.store[session_id]


   
    
    def clear_all_sessions(self):
        st.session_state.store = {}