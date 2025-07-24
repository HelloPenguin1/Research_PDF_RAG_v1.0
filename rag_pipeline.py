from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from config import set_environment

set_environment()

class RAG_pipeline:
    def __init__(self, llm):
        self.llm = llm
        self.reformulation_prompt = self.create_reformulation_prompt()
        self.answer_prompt = self.create_answer_prompt()

    
    def create_reformulation_prompt(self):
        reform_sys_prompt= """
            Given a chat history and a recent user question which might 
            reference context in the chat history, formulate a standalone
            question which can be understood without the chat history.
            DO NOT answer the question. Just reformulate the question if needed
            else return as it is.
        """

        return ChatPromptTemplate.from_messages([
            ("system", reform_sys_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
    

    
    def create_answer_prompt(self):
        answer_sys_prompt = """
            You are a research assistant for question answering tasks.
            Make sure to answer the questions as accurately as possible without 
            leaving any details using the following retrieved context.
            Give a complete answer to the question.

            Context: {context} 
        """

        return ChatPromptTemplate.from_messages([
            ("system", answer_sys_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        
    def create_rag_chain(self, retriever):
        #create history aware retriever 
        history_aware_retriever = create_history_aware_retriever(self.llm, retriever, self.reformulation_prompt)

        #create question answer chain
        question_answer_chain = create_stuff_documents_chain(self.llm, self.answer_prompt)

        #create a retrieval chain
        rag_pipeline = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
        return rag_pipeline
    

    def create_conversational_chain(self, rag_pipeline, get_session_history_func):
        return RunnableWithMessageHistory(
            rag_pipeline,
            get_session_history_func,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
    

