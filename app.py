from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader    ## read the pdf files
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="My Chat With PDF", page_icon="🤖")

st.header("My Chat With PDF Web Application")

user_question = st.text_input("Ask a question from PDF....")

ask_button = st.button("Ask")

def get_pdf_text(pdf_docs):
    pdf_text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    return pdf_text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectors(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectors = FAISS.from_texts(chunks, embeddings)
    vectors.save_local("faiss_index")

def get_conversional_chain():
    prompt_template = """
    Answer the question as detailed as possible from the context provided. Make sure that the answer 
    is relevant to the context. If the answer is not in the context, say "I don't know, don't give the wrong
    answer. \n\n
    Context : \n {context}?:"\n
    Question : \n {question}?\n
    Answer :
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    prompt_temp = PromptTemplate(template=prompt_template,input_variables=["context", "question"])
    chain = load_qa_chain(model,prompt = prompt_temp)
    return chain

def user_input(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    data = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = data.similarity_search(question)
    chain = get_conversional_chain()
    response = chain({"input_documents":docs, "question": question}, return_only_outputs=True)
    st.write(response["output_text"])

with st.sidebar:
    st.title("Menu")
    pdf_docs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    submit = st.button("Submit & Process")
    if submit:
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vectors(text_chunks)
            st.success("Done")
        
if ask_button:
    user_input(user_question)