import streamlit as st
from langchain_groq import ChatGroq

# Import  custom modules
from config import Config, set_environment
from document_processor import Document_Processor
from rag_pipeline import RAG_pipeline
from session_manager import SessionManager

set_environment()

session_manager = SessionManager()

#Displaying chat history
def display_messages(session_history):

    if hasattr(session_history, 'messages') and session_history.messages:
        for message in session_history.messages:
            if hasattr(message, 'type'):
                if message.type == 'human':
                    with st.chat_message("user"):
                        st.write(message.content)
                elif message.type == "ai":
                    with st.chat_message("assistant"):
                        st.write(message.content)


def main():
    st.title("Research Assistant based on Conversational RAG")
    st.write("Upload your Research PDFs and ask questions!")

    api_key = st.secrets.get('GROQ_API_KEY')

    #Define llm 
    llm = ChatGroq(groq_api_key = api_key, model_name="Gemma2-9b-It")


    session_id = st.text_input("Session ID", value="Default Session")

    uploaded_files = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        accept_multiple_files=False
    )

    if not uploaded_files:
        st.info("Please upload a PDF to start chatting")
        return
    
    with st.spinner("Processing your PDF..."):
        document_processor = Document_Processor() ## embedding and text splitter auto definied/initiated
        retriever = document_processor.process_pdf(uploaded_files)
        document_processor.cleanup_temp_file() 

    rag_pipeline_manager = RAG_pipeline(llm)
    rag_chain = rag_pipeline_manager.create_rag_chain(retriever)

    conversational_rag = rag_pipeline_manager.create_conversational_chain(
        rag_chain,
        session_manager.get_session_history
    )


    chat_container = st.container()

    with chat_container:
        session_history = session_manager.get_session_history(session_id)
        display_messages(session_history)


    #User input
    user_input = st.chat_input("Input your query...:")
    
    if user_input:
        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)

        
        with st.spinner("Generating response..."):
            
            response = conversational_rag.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            
            with chat_container:
                with st.chat_message("assistant"):
                    st.write(response['answer'])
        
    
    with st.sidebar:
        st.header("Session Management")
        if st.button("Clear Current Session"):
            session_manager.clear_session(session_id)
            st.success(f"Cleared session: {session_id}")
        
        if st.button("Clear All Sessions"):
            session_manager.clear_all_sessions()
            st.success("Cleared all sessions")
        
        st.write(f"Total Sessions: {len(session_manager.get_all_sessions())}")



if __name__ == "__main__":
    main()

