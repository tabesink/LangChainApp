# ref: https://github.com/alejandro-ao

import streamlit as st 
from dotenv import load_dotenv # access api keys
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings # use openai and huggingface embeddings
from langchain.vectorstores import FAISS # vectorstore: store numeric representation of text (faiss: stores locally) ? use persistance database
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain # enables interactions with vectorstore
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    """loop through each pdf and extract/concatenate all raw text data"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() # extrcat all raw text from pdf
    return text


def get_text_chunks(raw_text):
    """divide text into chunks using langchain"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, # num characters
        chunk_overlap=200, # overlap to prevent context meaning loss 
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

# run locally using huggingfaceinstructembeddings
def get_vectorstore(text_chunks):
    #vectorstore using openai embeddings and faiss
    #embeddings = OpenAIEmbeddings() -- charges money
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings) # create database
    return vectorstore


def get_conversation_chain(vectorstore):
    # initialize an instance of memory from langchain
    llm = ChatOpenAI(temperature=0, openai_api_key="XX")
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_message=True) # ? use buffer memory or entity memory
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    #load_dotenv() # use variables inside of .env
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True) # unsafe_allow_html=True: enables streamlit to see html inside variable, and pasrse html as html  

    # initalize sesssion state to maintain persistant variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question: 
        handle_userinput(user_question)

    st.write(user_template.replace("{{MSG}}", "Hello bot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello human"), unsafe_allow_html=True)

    # add sidebar
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on Process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"): # add spinner 
                # get pdf test 
                raw_text = get_pdf_text(pdf_docs) # take list of pdf files and return single string of text content  
                
                # get text chunks 
                text_chunks = get_text_chunks(raw_text)

                # get create vector store with embeddings using open-ai 
                vectorstore = get_vectorstore(text_chunks)

                # create instance of conv. chain (initalize conv. object)
                st.session_state.conversation = get_conversation_chain(vectorstore) # link variables to session state of the application so streamlit won't re-initialize things


if __name__ == '__main__':
    main()